# instances/instance.py
"""
instances/instance.py

Instance data class for the TD-DRPTW reproduction project.

This module implements the Instance class and several helper/static methods useful for:
 - holding instance data (nodes, coords, demands, time windows, drone-eligible set D),
 - deterministic instance id / seed derivation,
 - generation helpers (coordinate samplers for the three instance types described in the paper),
 - serialization to/from a JSON-friendly dict and atomic saving/loading,
 - validation of structural and parameter consistency.

Notes and design constraints (followed from the project design and config.yaml):
 - Node indexing: 0 = origin depot, 1..n = customers, n+1 = end depot
 - Coordinates: kilometers (km)
 - Speeds: km/h in params; travel times are in minutes (distance_km / speed_kmph * 60)
 - Time windows and durations: minutes
 - Instance must be self-contained (params snapshot contains Q_t, Q_d, L_t, L_d, speeds, beta, cost params, service_times if present).
 - The generated_seed is computed deterministically from a base_seed and instance descriptors so runs are reproducible.

Public API (as required by design):
 - class Instance:
     - __init__(..., see signature)
     - save(path: str | Path) -> str  # returns path written
     - @classmethod load(path: str | Path) -> "Instance"
     - to_dict() -> dict
     - validate(raise_on_error: bool = True, auto_fix: bool = False) -> bool
     - helper accessors: get_customer_list(), get_depot_indices(), get_coords_array()

 - Several static/class helper methods used by InstanceGenerator or tests:
     - compute_generated_seed(base_seed, instance_type, n, theta, replicate)
     - create_nodes(n)
     - generate_coords_type1, generate_coords_type2, generate_coords_type3
     - select_D
     - generate_time_windows_solomon_style

This module intentionally avoids imposing values for configuration fields that the
paper or config.yaml left unspecified (e.g., service times). If such fields are
missing in params, validate() will either warn or raise depending on caller flags.

Author: Reproducibility codebase
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

# numpy is a project dependency and provides RNG and numeric helpers
import numpy as np

# Try to import yaml to read config if needed; not required for core Instance methods
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore


# ---- Type aliases for clarity ----
Node = int
Coord = Tuple[float, float]
TimeWindow = Tuple[float, float]


# ---- Utility helpers ----
def _atomic_write_text(path: Union[str, Path], text: str, mode: str = "w", encoding: str = "utf-8") -> None:
    """
    Atomically write text content to path by writing to a temp file and moving it into place.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create temp file in same directory to ensure atomic replace works across filesystems
    fd, tmpname = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                # fsync may fail on some platforms; ignore
                pass
        # atomic replace
        os.replace(tmpname, str(path))
    except Exception:
        # cleanup temp file if exists
        try:
            if os.path.exists(tmpname):
                os.remove(tmpname)
        except Exception:
            pass
        raise


def _ensure_list_of_len(seq: Sequence[Any], n: int, name: str) -> None:
    if len(seq) != n:
        raise ValueError(f"{name} must have length {n}, got {len(seq)}")


# ---- Instance class ----
@dataclass
class Instance:
    """
    Data class representing a single TD-DRPTW instance.

    Required fields:
      - id: unique identifier string for the instance
      - n: number of customers (int)
      - coords: mapping node -> (x_km, y_km) for nodes 0..n+1
      - demands: mapping node -> int (0 for depots)
      - time_windows: mapping node -> (earliest, latest) in minutes
      - D: set of customer indices (subset of 1..n) that are drone-eligible
      - params: mapping of problem parameters (snapshot). Should include:
          Q_t, Q_d, L_t, L_d, v_t_kmph, v_d_kmph, beta,
          fixed_vehicle_cost_F (F), c_t, c_d, service_times (may be None)
      - meta: optional metadata dict (generator settings, timestamps, notes)

    The object is JSON-serializable via to_dict(). Use save() to persist atomically.
    """

    id: str
    n: int
    coords: Dict[Node, Coord]
    demands: Dict[Node, int]
    time_windows: Dict[Node, TimeWindow]
    D: Set[Node]
    params: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Basic structural adjustments and checks
        if not isinstance(self.n, int) or self.n < 0:
            raise ValueError("n (number of customers) must be a non-negative integer.")
        # Build nodes list and ensure presence
        expected_nodes = self.create_nodes(self.n)
        # Convert keys to ints if provided as strings (defensive)
        # For coords/demands/time_windows we expect keys as ints already; caller loaders may pass strings
        self.coords = {int(k): tuple(v) for k, v in self.coords.items()}
        self.demands = {int(k): int(v) for k, v in self.demands.items()}
        self.time_windows = {int(k): (float(v[0]), float(v[1])) for k, v in self.time_windows.items()}
        # ensure entries for all expected nodes
        missing_coords = [i for i in expected_nodes if i not in self.coords]
        if missing_coords:
            raise ValueError(f"coords missing entries for nodes: {missing_coords}")
        missing_demands = [i for i in expected_nodes if i not in self.demands]
        if missing_demands:
            raise ValueError(f"demands missing entries for nodes: {missing_demands}")
        missing_tw = [i for i in expected_nodes if i not in self.time_windows]
        if missing_tw:
            raise ValueError(f"time_windows missing entries for nodes: {missing_tw}")

        # Normalize D: ensure ints and subset of customers
        self.D = set(int(x) for x in self.D)
        customers = set(range(1, self.n + 1))
        if not self.D.issubset(customers):
            bad = sorted(list(self.D - customers))
            raise ValueError(f"D contains invalid customer indices (not in 1..n): {bad}")

        # Ensure meta contains generation timestamp if missing
        if "generated_at" not in self.meta:
            self.meta["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # --------------------
    # Serialization
    # --------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable dict representing the instance.

        Keys and layout follow the design plan and serialization expectations so other modules
        (SerializationManager) can read and write these files.
        """
        # Ensure all numeric types are JSON-friendly (int / float)
        coords_out = {str(k): [float(self.coords[k][0]), float(self.coords[k][1])] for k in sorted(self.coords.keys())}
        demands_out = {str(k): int(self.demands[k]) for k in sorted(self.demands.keys())}
        tw_out = {str(k): [float(self.time_windows[k][0]), float(self.time_windows[k][1])] for k in sorted(self.time_windows.keys())}
        d_list = sorted(list(self.D))

        return {
            "id": str(self.id),
            "n": int(self.n),
            "nodes": [i for i in self.create_nodes(self.n)],
            "coords": coords_out,
            "demands": demands_out,
            "time_windows": tw_out,
            "D": d_list,
            "params": self._serialize_params(),
            "meta": dict(self.meta or {}),
        }

    def _serialize_params(self) -> Dict[str, Any]:
        """
        Prepare params for JSON serialization ensuring numeric types are basic Python numbers.
        """
        p = dict(self.params or {})
        # Convert numpy types if any
        def _clean_value(v):
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, (list, tuple)):
                return [_clean_value(x) for x in v]
            if isinstance(v, dict):
                return {str(k): _clean_value(val) for k, val in v.items()}
            return v

        return {str(k): _clean_value(v) for k, v in p.items()}

    def save(self, path: Union[str, Path]) -> str:
        """
        Atomically save the instance JSON to 'path'.

        If 'path' is a directory, a file named '<id>.json' will be created inside it.
        Returns the absolute path to the written file as a string.
        """
        path = Path(path)
        if path.is_dir():
            out_path = path / f"{self.id}.json"
        else:
            # ensure parent exists
            out_path = path
            out_path.parent.mkdir(parents=True, exist_ok=True)

        obj = self.to_dict()
        # write pretty JSON but stable key order for reproducibility
        text = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2)
        _atomic_write_text(out_path, text)
        return str(out_path.resolve())

    # --------------------
    # Loading
    # --------------------
    @classmethod
    def load(cls, path: Union[str, Path]) -> "Instance":
        """
        Load instance JSON from file and construct an Instance object.

        This method will validate the minimal required fields and types but will not run full validate().
        Use validate() for comprehensive checks.

        Raises ValueError or JSONDecodeError on malformed input.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Instance file not found: {path}")

        with open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)

        # basic shape checks
        if not isinstance(data, dict):
            raise ValueError("Instance JSON root must be an object")

        # Extract required fields with defensive handling
        try:
            id_ = str(data["id"])
            n = int(data["n"])
            coords_raw = data["coords"]
            demands_raw = data["demands"]
            tw_raw = data["time_windows"]
            D_raw = data.get("D", [])
            params = data.get("params", {}) or {}
            meta = data.get("meta", {}) or {}
        except KeyError as e:
            raise ValueError(f"Missing required instance top-level key: {e}")

        # Convert coords/demands/tw keys to ints and values to proper tuples
        coords: Dict[int, Coord] = {}
        for k, v in coords_raw.items():
            ik = int(k)
            if not (isinstance(v, (list, tuple)) and len(v) == 2):
                raise ValueError(f"coords[{k}] must be [x, y]")
            coords[ik] = (float(v[0]), float(v[1]))

        demands: Dict[int, int] = {}
        for k, v in demands_raw.items():
            ik = int(k)
            demands[ik] = int(v)

        tw: Dict[int, TimeWindow] = {}
        for k, v in tw_raw.items():
            ik = int(k)
            if not (isinstance(v, (list, tuple)) and len(v) == 2):
                raise ValueError(f"time_windows[{k}] must be [earliest, latest]")
            tw[ik] = (float(v[0]), float(v[1]))

        D_set = set(int(x) for x in D_raw)

        inst = cls(
            id=id_,
            n=n,
            coords=coords,
            demands=demands,
            time_windows=tw,
            D=D_set,
            params=params,
            meta=meta,
        )
        return inst

    # --------------------
    # Validation
    # --------------------
    def validate(self, raise_on_error: bool = True, auto_fix: bool = False) -> bool:
        """
        Validate structural integrity and parameter consistency.

        Parameters:
          - raise_on_error: if True, a failed validation raises ValueError with messages.
          - auto_fix: if True, perform small permitted automatic fixes:
                * remove customers from D whose demand > Q_d (and record in meta["auto_fixed"])
              auto_fix should be used carefully; default False.

        Returns:
          - True if validation passes (or if auto_fix fixed all issues and remaining checks OK).
          - False if validation fails and raise_on_error is False.

        Validation checks performed:
          - Node indexing and presence of coords/demands/time_windows for all nodes 0..n+1
          - Params contain required problem parameters (Q_t, Q_d, L_t, L_d, v_t_kmph, v_d_kmph, beta, F, c_t, c_d)
          - Q_d < Q_t
          - speeds > 0, L_t,L_d > 0, beta >= 1
          - depots time window equals [0, L_t] (warn if not)
          - for each customer: 0 <= e_i <= l_i <= L_t
          - all customer demands <= Q_t
          - if customer in D then demand <= Q_d (auto-fix or error)
          - if service_times missing in params, add a warning in meta
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Node structure
        expected_nodes = self.create_nodes(self.n)
        if sorted(list(self.coords.keys())) != expected_nodes:
            errors.append(f"coords keys mismatch. Expected nodes {expected_nodes}, got {sorted(list(self.coords.keys()))}")
        if sorted(list(self.demands.keys())) != expected_nodes:
            errors.append(f"demands keys mismatch. Expected nodes {expected_nodes}, got {sorted(list(self.demands.keys()))}")
        if sorted(list(self.time_windows.keys())) != expected_nodes:
            errors.append(f"time_windows keys mismatch. Expected nodes {expected_nodes}, got {sorted(list(self.time_windows.keys()))}")

        # Params presence
        required_param_keys = ["Q_t", "Q_d", "L_t", "L_d", "v_t_kmph", "v_d_kmph", "beta", "fixed_vehicle_cost_F", "c_t", "c_d"]
        missing_params = [k for k in required_param_keys if k not in self.params]
        if missing_params:
            errors.append(f"params missing required keys: {missing_params}")

        # validate numeric param ranges if present
        Q_t = self.params.get("Q_t")
        Q_d = self.params.get("Q_d")
        L_t = self.params.get("L_t")
        L_d = self.params.get("L_d")
        v_t = self.params.get("v_t_kmph")
        v_d = self.params.get("v_d_kmph")
        beta = self.params.get("beta")

        try:
            if Q_t is not None:
                if not (float(Q_t) > 0):
                    errors.append("Q_t must be > 0")
            if Q_d is not None:
                if not (float(Q_d) > 0):
                    errors.append("Q_d must be > 0")
            if Q_t is not None and Q_d is not None:
                if not (float(Q_d) < float(Q_t)):
                    errors.append(f"Q_d ({Q_d}) must be strictly less than Q_t ({Q_t}) as assumed in paper")
            if L_t is not None:
                if not (float(L_t) > 0):
                    errors.append("L_t must be > 0")
            if L_d is not None:
                if not (float(L_d) > 0):
                    errors.append("L_d must be > 0")
            if v_t is not None:
                if not (float(v_t) > 0):
                    errors.append("v_t_kmph must be > 0")
            if v_d is not None:
                if not (float(v_d) > 0):
                    errors.append("v_d_kmph must be > 0")
            if beta is not None:
                if not (float(beta) >= 1.0):
                    warnings.append("beta < 1.0 is unusual; paper assumes beta >= 1")
        except (TypeError, ValueError):
            errors.append("One or more numeric params are invalid types")

        # Time windows checks
        if L_t is not None:
            # depot expectation
            depot_tw = self.time_windows.get(0)
            end_depot_tw = self.time_windows.get(self.n + 1)
            if depot_tw is not None:
                if not (math.isclose(float(depot_tw[0]), 0.0, rel_tol=1e-9) and math.isclose(float(depot_tw[1]), float(L_t), rel_tol=1e-9)):
                    warnings.append(f"Depot time window expected [0, L_t={L_t}], got {depot_tw}")
            if end_depot_tw is not None:
                if not (math.isclose(float(end_depot_tw[0]), 0.0, rel_tol=1e-9) and math.isclose(float(end_depot_tw[1]), float(L_t), rel_tol=1e-9)):
                    warnings.append(f"End depot time window expected [0, L_t={L_t}], got {end_depot_tw}")

            # customers
            for i in range(1, self.n + 1):
                e_i, l_i = self.time_windows[i]
                if e_i < 0:
                    errors.append(f"Customer {i} earliest time {e_i} < 0")
                if l_i > float(L_t) + 1e-9:
                    errors.append(f"Customer {i} latest time {l_i} > L_t ({L_t})")
                if e_i > l_i:
                    errors.append(f"Customer {i} earliest time {e_i} > latest {l_i}")

        # Demands vs capacities
        for i in range(1, self.n + 1):
            d_i = int(self.demands[i])
            if Q_t is not None and d_i > float(Q_t):
                errors.append(f"Customer {i} demand {d_i} exceeds truck capacity Q_t={Q_t} - impossible to serve")
        # D vs Q_d
        bad_D = []
        for i in sorted(list(self.D)):
            d_i = int(self.demands[i])
            if Q_d is not None:
                if d_i > float(Q_d):
                    bad_D.append(i)
        if bad_D:
            if auto_fix:
                # remove those from D
                for i in bad_D:
                    self.D.discard(i)
                self.meta.setdefault("auto_fixed", {}).setdefault("removed_from_D_due_to_Qd", []).extend(bad_D)
                # warning instead of error
                warnings.append(f"Removed customers {bad_D} from D because demand > Q_d")
            else:
                errors.append(f"Customers in D whose demand exceed Q_d: {bad_D} (either remove them from D or set auto_fix=True)")

        # Validate D elements are in 1..n
        for d in self.D:
            if not (1 <= d <= self.n):
                errors.append(f"D contains invalid customer index: {d}")

        # Service times presence
        st = self.params.get("service_times", {}) or {}
        truck_st = st.get("truck_service_time_minutes", None)
        drone_st = st.get("drone_service_time_minutes", None)
        if truck_st is None or drone_st is None:
            warnings.append("Service times for truck/drone are not both specified in params['service_times']. The paper "
                            "mentions that travel times include service times; please set these before large experiments.")

        # Geometry checks
        coord_values = list(self.coords.values())
        for (x, y) in coord_values:
            if not (math.isfinite(x) and math.isfinite(y)):
                errors.append(f"Found non-finite coordinate: {(x, y)}")
        # duplicates are allowed but warn
        uniq_coords = set(coord_values)
        if len(uniq_coords) != len(coord_values):
            warnings.append("Some customer/depot coordinates are identical (duplicate coordinates found)")

        # Finalize
        if errors:
            msg = "Instance validation failed with errors:\n" + "\n".join(f"- {e}" for e in errors)
            if raise_on_error:
                raise ValueError(msg)
            else:
                # attach warnings to meta and return False
                self.meta.setdefault("validation", {})["errors"] = errors
                if warnings:
                    self.meta.setdefault("validation", {}).setdefault("warnings", []).extend(warnings)
                return False

        # No hard errors; attach warnings if any
        if warnings:
            self.meta.setdefault("validation", {}).setdefault("warnings", []).extend(warnings)

        return True

    # --------------------
    # Convenience accessors
    # --------------------
    def get_customer_list(self) -> List[int]:
        """Return ordered list of customer indices 1..n."""
        return list(range(1, self.n + 1))

    def get_depot_indices(self) -> Tuple[int, int]:
        """Return (origin_depot_index, end_depot_index)."""
        return 0, self.n + 1

    def get_coords_array(self, node_order: Optional[Sequence[int]] = None) -> np.ndarray:
        """
        Return coordinates as an (m x 2) numpy array in the requested node order.

        If node_order is None, returns array for nodes [0,1,...,n,n+1].
        """
        if node_order is None:
            node_order = self.create_nodes(self.n)
        arr = np.zeros((len(node_order), 2), dtype=float)
        for idx, node in enumerate(node_order):
            c = self.coords.get(int(node))
            if c is None:
                raise KeyError(f"Coordinate for node {node} not found")
            arr[idx, 0] = float(c[0])
            arr[idx, 1] = float(c[1])
        return arr

    # --------------------
    # Deterministic seed/id helpers and generation helpers
    # --------------------
    @staticmethod
    def compute_generated_seed(base_seed: int, instance_type: int, n: int, theta: float, replicate: int) -> int:
        """
        Deterministically compute a per-instance RNG seed given base_seed and instance description.

        Formula (as in reproducibility plan):
            generated_seed = base_seed + instance_type*1000 + n*10 + int(round(theta*100)) + replicate

        Note: returns an int (possibly large). Caller should ensure base_seed is integer.
        """
        if base_seed is None:
            base_seed = 0
        try:
            tb = int(base_seed)
        except Exception:
            tb = int(float(base_seed))
        tpart = int(instance_type) * 1000
        npart = int(n) * 10
        thetap = int(round(float(theta) * 100.0))
        rpart = int(replicate)
        gen = tb + tpart + npart + thetap + rpart
        # ensure non-negative
        return int(gen & 0x7FFFFFFF)

    @staticmethod
    def create_nodes(n: int) -> List[int]:
        """Return list of nodes [0,1,...,n,n+1]."""
        return [0] + list(range(1, n + 1)) + [n + 1]

    @staticmethod
    def generate_coords_type1(n: int, rng: np.random.Generator, grid_start: float = 0.0, grid_end: float = 15.0, grid_step: float = 0.5) -> Tuple[Dict[int, Coord], Coord]:
        """
        Type 1: customers' x,y sampled independently from the discrete grid {grid_start, grid_start+step, ..., grid_end}.
        Depot: randomly chosen on same grid.

        Returns (coords_customers_dict, depot_coord)
        coords_customers_dict keys are 1..n
        """
        # build discrete grid values
        num = int(round((grid_end - grid_start) / grid_step)) + 1
        grid_vals = [round(grid_start + i * grid_step, 8) for i in range(num)]
        # sample customers
        coords = {}
        for i in range(1, n + 1):
            x = float(rng.choice(grid_vals))
            y = float(rng.choice(grid_vals))
            coords[i] = (x, y)
        # depot
        depot = (float(rng.choice(grid_vals)), float(rng.choice(grid_vals)))
        return coords, depot

    @staticmethod
    def generate_coords_type2(n: int, rng: np.random.Generator, grid_start: float = 0.0, grid_end: float = 15.0, grid_step: float = 0.5) -> Tuple[Dict[int, Coord], Coord]:
        """
        Type 2: customers sampled as in type1; depot located at mean of customer coordinates.
        """
        coords_customers, _ = Instance.generate_coords_type1(n=n, rng=rng, grid_start=grid_start, grid_end=grid_end, grid_step=grid_step)
        xs = [coords_customers[i][0] for i in range(1, n + 1)]
        ys = [coords_customers[i][1] for i in range(1, n + 1)]
        mean_coord = (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))
        return coords_customers, mean_coord

    @staticmethod
    def generate_coords_type3(n: int, rng: np.random.Generator, gamma_mu: float = 0.0, gamma_sigma: float = 10.0) -> Tuple[Dict[int, Coord], Coord]:
        """
        Type 3: clustered generation. For each customer:
            - gamma sampled from Normal(gamma_mu, gamma_sigma) and converted to non-negative radial distance by abs()
            - phi sampled Uniform(0, 2*pi)
            - x = gamma * cos(phi), y = gamma * sin(phi)
        Depot is located at origin (0,0).

        The absolute value treatment of gamma is chosen to ensure distances are non-negative and documented
        in the instance.meta during generation.
        """
        coords = {}
        for i in range(1, n + 1):
            gamma = float(rng.normal(loc=gamma_mu, scale=gamma_sigma))
            gamma = abs(gamma)  # enforce non-negative radial distance
            phi = float(rng.uniform(0.0, 2.0 * math.pi))
            x = gamma * math.cos(phi)
            y = gamma * math.sin(phi)
            coords[i] = (float(x), float(y))
        depot = (0.0, 0.0)
        return coords, depot

    @staticmethod
    def select_D(n: int, theta: float, rng: np.random.Generator, demands: Dict[int, int], Q_d: Optional[float]) -> Set[int]:
        """
        Select a subset D of customers that are drone-eligible.

        Rules:
          - Desired size = floor(n * theta)
          - Only customers whose demand <= Q_d are eligible
          - If fewer eligible customers exist than desired size, return all eligible and caller may log a warning
        """
        desired = int(math.floor(float(n) * float(theta)))
        customers = list(range(1, n + 1))
        eligible = [i for i in customers if (Q_d is None) or (int(demands.get(i, 0)) <= int(Q_d))]
        if desired <= 0:
            return set()
        if len(eligible) <= desired:
            # return all eligible
            return set(sorted(eligible))
        # sample without replacement deterministically from RNG
        chosen = list(rng.choice(np.array(eligible, dtype=int), size=desired, replace=False))
        return set(int(x) for x in chosen)

    @staticmethod
    def generate_time_windows_solomon_style(n: int, rng: np.random.Generator, L_t: float, half_width_mu: float = 30.0, half_width_sigma: float = 10.0) -> Dict[int, TimeWindow]:
        """
        Generate time windows for customers using a simple Solomon-like method:
          - For each customer, sample half-width h_i ~ Normal(mu=half_width_mu, sigma=half_width_sigma)
            If sampled h_i <= 1, clamp to 1.0 minute minimal half-width.
          - Choose center c_i ~ Uniform(0, L_t)
          - e_i = max(0, c_i - h_i), l_i = min(L_t, c_i + h_i)
        Depot time windows set to [0, L_t]. Returns dict keyed by nodes 0..n+1.
        """
        tw = {}
        # depots
        tw[0] = (0.0, float(L_t))
        tw[n + 1] = (0.0, float(L_t))
        for i in range(1, n + 1):
            h = float(rng.normal(loc=half_width_mu, scale=half_width_sigma))
            if not math.isfinite(h) or h <= 1.0:
                h = 1.0
            c = float(rng.uniform(0.0, float(L_t)))
            e = max(0.0, c - h)
            l = min(float(L_t), c + h)
            # ensure e <= l
            if e > l:
                # numerical safeguard, set minimal window
                e = max(0.0, min(e, l))
                l = max(e, l)
            tw[i] = (float(e), float(l))
        return tw

    # --------------------
    # Representation
    # --------------------
    def __repr__(self) -> str:
        return f"Instance(id={self.id!r}, n={self.n}, customers=1..{self.n})"
