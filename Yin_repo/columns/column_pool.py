## columns/column_pool.py

"""
columns/column_pool.py

ColumnPool: thread-safe manager for synthetic-route Columns used by the RLMP.

Responsibilities
- Store Columns (route columns) indexed by a deterministic route_id.
- Maintain fast indices:
    - arc -> set(route_id)
    - customer -> set(route_id)
- Provide operations:
    - add(column)
    - add_many(columns)
    - remove_by_predicate(pred)
    - get_all() -> list of active Column objects
    - get_columns_using_arc(i, j) -> list of active Column objects using arc (i,j)
    - find_by_route_id(route_id) -> Optional[Column]
- Optionally persist columns to disk under results_dir/columns when enabled in config.

Design notes
- Thread-safe via an RLock protecting internal structures.
- Route id generation is deterministic: prefer column.route_id if present; otherwise compute
  sha1 hash over a canonical JSON representation of the route (column.serialize()) with an
  optional salt derived from config.reproducibility.global_random_seed for reproducibility.
- Columns are expected to implement the minimal interface (see routes/route.Column):
    - route_id (str) attribute or serializable via column.serialize()
    - a_ir or covers() to obtain covered customers (we accept both)
    - contains_arc(i,j) -> bool
    - cost (float) or route.cost() to compute cost
  The ColumnPool will try to work with such objects flexibly but will raise on grossly invalid inputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

# Try to import yaml to optionally load config.yaml if no config provided
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

# Type alias for arc keys
Arc = Tuple[int, int]


def _now_iso() -> str:
    """Return current UTC time as ISO string."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


class ColumnPool:
    """
    Thread-safe pool of Columns (synthetic-route columns) with indices for arcs and customers.

    Public methods (per design):
      - __init__(config: Optional[dict] = None)
      - add(column: Any) -> None
      - add_many(columns: Iterable[Any]) -> None
      - remove_by_predicate(pred: Callable[[Any], bool]) -> None
      - get_all() -> List[Any]
      - get_columns_using_arc(i: int, j: int) -> List[Any]
      - find_by_route_id(route_id: str) -> Optional[Any]
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize ColumnPool.

        Args:
            config: optional dict-like configuration (as loaded from config.yaml). Expected keys:
                - output.results_dir (str) : base results directory
                - output.save_columns (bool) : whether to persist columns to disk
                - reproducibility.global_random_seed (int) : optional salt for deterministic hashing
        """
        # Load config (accept either dict or path string)
        if config is None:
            cfg = {}
            # attempt to read ./config.yaml if available to honor the reproducibility plan
            try:
                cfg_path = Path("config.yaml")
                if cfg_path.exists() and yaml is not None:
                    with open(cfg_path, "rt", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f) or {}
            except Exception:
                cfg = {}
            self._config: Dict[str, Any] = cfg
        elif isinstance(config, dict):
            self._config = dict(config)
        else:
            # defensive: unsupported config type, coerce to empty dict
            self._config = {}

        output_cfg = self._config.get("output", {}) or {}
        reproduc_cfg = self._config.get("reproducibility", {}) or {}

        # Persistence settings
        self._save_columns: bool = bool(output_cfg.get("save_columns", False))
        results_dir = Path(output_cfg.get("results_dir", "results"))
        self._results_dir: Path = results_dir
        self._columns_dir: Path = self._results_dir / "columns"
        if self._save_columns:
            try:
                self._columns_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                # best-effort; do not fail initialization
                pass

        # Deterministic salt for hashing route representations (optional)
        self._salt: Optional[str] = None
        seed = reproduc_cfg.get("global_random_seed")
        if seed is not None:
            try:
                self._salt = str(int(seed))
            except Exception:
                self._salt = str(seed)

        # Internal storage
        self._lock = threading.RLock()
        # route_id -> column object
        self._columns: Dict[str, Any] = {}
        # arc -> set(route_id)
        self._arc_index: Dict[Arc, Set[str]] = defaultdict(set)
        # customer -> set(route_id)
        self._customer_index: Dict[int, Set[str]] = defaultdict(set)
        # metadata per route_id
        self._meta: Dict[str, Dict[str, Any]] = {}
        # simple counters
        self._total_added: int = 0
        self._total_removed: int = 0
        # logical iteration counter maintained by pool (incremented on adds)
        self._iteration_counter: int = 0

    # -----------------------------
    # Public API
    # -----------------------------

    def add(self, column: Any) -> None:
        """
        Add a Column to the pool.

        If a column with the same deterministic route_id already exists and is active,
        this call is idempotent (it updates last_added_iteration metadata). If it exists
        but was previously deactivated (removed), this will reactivate it.

        The column object is stored as-is (no deep-copy). The Column object must provide
        either a 'route_id' attribute or be serializable via column.serialize() / column.route.

        Arguments:
            column: Column-like object produced by Route.to_column() / pricing.
        """
        # Basic validation and canonical route_id computation outside lock where possible
        route_id = self._compute_route_id_safe(column)

        if route_id is None:
            raise ValueError("ColumnPool.add: unable to determine a deterministic route_id for provided column")

        now_ts = _now_iso()

        with self._lock:
            existing = self._columns.get(route_id)
            if existing is not None and self._meta.get(route_id, {}).get("is_active", False):
                # already present and active: update metadata and return
                self._iteration_counter += 1
                self._meta[route_id]["last_added_iteration"] = int(self._iteration_counter)
                self._meta[route_id]["last_touched_at"] = now_ts
                return

            # Validate minimal expected interface of column
            try:
                # coverage: prefer a_ir (dict) then covers() method
                covers = None
                if hasattr(column, "a_ir") and isinstance(getattr(column, "a_ir"), dict):
                    # a_ir is dict with keys as customer ids (1..n) mapped to 0/1
                    covers = {int(k) for k, v in getattr(column, "a_ir").items() if int(v) == 1}
                elif hasattr(column, "covers") and callable(getattr(column, "covers")):
                    covers = set(getattr(column, "covers")())
                else:
                    # attempt to read 'route' then inspect route
                    covers = None
                    if hasattr(column, "route") and hasattr(column.route, "covers"):
                        covers = set(column.route.covers())
                if covers is None:
                    # fallback: try column.serialize() and look for coverage list
                    if hasattr(column, "serialize") and callable(getattr(column, "serialize")):
                        s = column.serialize()
                        if isinstance(s, dict) and s.get("covered_customers") is not None:
                            covers = set(int(x) for x in s.get("covered_customers") or [])
                    # If still None, set empty set (best-effort)
                    if covers is None:
                        covers = set()
            except Exception as ex:
                raise ValueError(f"ColumnPool.add: failed to extract covers from column: {ex}")

            # Arcs: determine arcs used by column
            try:
                # column should expose truck_arcs and drone_arcs lists in serialize() or attributes
                arcs_set: Set[Arc] = set()
                if hasattr(column, "truck_arcs") and getattr(column, "truck_arcs") is not None:
                    for a in getattr(column, "truck_arcs"):
                        arcs_set.add((int(a[0]), int(a[1])))
                if hasattr(column, "drone_arcs") and getattr(column, "drone_arcs") is not None:
                    for a in getattr(column, "drone_arcs"):
                        arcs_set.add((int(a[0]), int(a[1])))
                # fallback: try column.serialize()
                if not arcs_set:
                    if hasattr(column, "serialize") and callable(getattr(column, "serialize")):
                        s = column.serialize()
                        if isinstance(s, dict):
                            ta = s.get("truck_arcs") or []
                            da = s.get("drone_arcs") or []
                            for a in ta:
                                arcs_set.add((int(a[0]), int(a[1])))
                            for a in da:
                                arcs_set.add((int(a[0]), int(a[1])))
                # final fallback: try column.route.compute_truck_arcs / compute_drone_arcs if available
                if not arcs_set and hasattr(column, "route"):
                    try:
                        if hasattr(column.route, "compute_truck_arcs"):
                            for a in column.route.compute_truck_arcs():
                                arcs_set.add((int(a[0]), int(a[1])))
                        if hasattr(column.route, "compute_drone_arcs"):
                            for a in column.route.compute_drone_arcs():
                                arcs_set.add((int(a[0]), int(a[1])))
                    except Exception:
                        pass
            except Exception:
                arcs_set = set()

            # Metadata setup
            self._iteration_counter += 1
            meta = {
                "created_at": now_ts,
                "last_touched_at": now_ts,
                "last_added_iteration": int(self._iteration_counter),
                "usage_count": int(self._meta.get(route_id, {}).get("usage_count", 0)),
                "is_active": True,
                "removed_at": None,
            }

            # Store column and update indices
            self._columns[route_id] = column
            self._meta[route_id] = meta

            # index by arcs
            for arc in arcs_set:
                self._arc_index[arc].add(route_id)

            # index by covered customers
            for c in covers:
                try:
                    self._customer_index[int(c)].add(route_id)
                except Exception:
                    # ignore invalid customer ids (defensive)
                    continue

            self._total_added += 1

            # Persist column if configured
            if self._save_columns:
                try:
                    self._persist_column(route_id, column, meta)
                except Exception:
                    # do not raise; best-effort persistence
                    pass

    def add_many(self, columns: Iterable[Any]) -> None:
        """
        Add multiple columns in sequence. This is a convenience wrapper that calls add()
        for each column. It acquires the pool lock only once for efficiency.
        """
        with self._lock:
            for col in columns:
                # call internal add logic but keep lock held to reduce contention / repeated locking
                # We call the public add() which reacquires the lock; to avoid deadlock, use internal helper logic:
                # So we create a small local wrapper that uses the same logic as add but without relocking.
                # For simplicity and to avoid duplicating logic, we release the lock and call add() normally.
                # Releasing and reacquiring introduces some overhead but keeps code simpler and safe.
                # (Given typical small batches, this is acceptable.)
                try:
                    # release lock temporarily
                    self._lock.release()
                    try:
                        self.add(col)
                    finally:
                        # re-acquire for next iteration
                        self._lock.acquire()
                except RuntimeError:
                    # In case releasing/acquiring failed, fallback to calling add without managing lock
                    self.add(col)

    def remove_by_predicate(self, pred: Callable[[Any], bool]) -> None:
        """
        Remove (deactivate) all columns for which pred(column) evaluates to True.

        The removal updates internal indices and marks metadata['is_active'] = False.
        The actual Column object is kept in memory for provenance; if long-term memory pressure
        is a concern this method could be extended to fully delete such entries (not done by default).

        Args:
            pred: callable taking a column object and returning True if it should be removed.
        """
        if not callable(pred):
            raise ValueError("ColumnPool.remove_by_predicate: pred must be callable")

        removed_route_ids: List[str] = []
        now_ts = _now_iso()

        with self._lock:
            # iterate over snapshot of route_id keys to allow mutation
            for route_id in list(self._columns.keys()):
                meta = self._meta.get(route_id)
                if meta is None:
                    continue
                if not meta.get("is_active", False):
                    continue
                col = self._columns.get(route_id)
                try:
                    if pred(col):
                        # de-index arcs
                        # determine arcs for column similarly as in add()
                        arcs_set: Set[Arc] = set()
                        try:
                            if hasattr(col, "truck_arcs") and getattr(col, "truck_arcs") is not None:
                                for a in getattr(col, "truck_arcs"):
                                    arcs_set.add((int(a[0]), int(a[1])))
                            if hasattr(col, "drone_arcs") and getattr(col, "drone_arcs") is not None:
                                for a in getattr(col, "drone_arcs"):
                                    arcs_set.add((int(a[0]), int(a[1])))
                            if not arcs_set and hasattr(col, "serialize") and callable(getattr(col, "serialize")):
                                s = col.serialize()
                                ta = s.get("truck_arcs") or []
                                da = s.get("drone_arcs") or []
                                for a in ta:
                                    arcs_set.add((int(a[0]), int(a[1])))
                                for a in da:
                                    arcs_set.add((int(a[0]), int(a[1])))
                        except Exception:
                            arcs_set = set()

                        for arc in arcs_set:
                            s = self._arc_index.get(arc)
                            if s is not None:
                                s.discard(route_id)
                                if not s:
                                    # remove empty key to keep index small
                                    try:
                                        del self._arc_index[arc]
                                    except KeyError:
                                        pass

                        # de-index customers
                        try:
                            covers = None
                            if hasattr(col, "a_ir") and isinstance(getattr(col, "a_ir"), dict):
                                covers = {int(k) for k, v in getattr(col, "a_ir").items() if int(v) == 1}
                            elif hasattr(col, "covers") and callable(getattr(col, "covers")):
                                covers = set(getattr(col, "covers")())
                            elif hasattr(col, "serialize") and callable(getattr(col, "serialize")):
                                s = col.serialize()
                                if isinstance(s, dict) and s.get("covered_customers") is not None:
                                    covers = set(int(x) for x in s.get("covered_customers") or [])
                            if covers is None:
                                covers = set()
                        except Exception:
                            covers = set()
                        for c in covers:
                            s = self._customer_index.get(int(c))
                            if s is not None:
                                s.discard(route_id)
                                if not s:
                                    try:
                                        del self._customer_index[int(c)]
                                    except KeyError:
                                        pass

                        # mark inactive and record removal metadata
                        self._meta[route_id]["is_active"] = False
                        self._meta[route_id]["removed_at"] = now_ts
                        self._meta[route_id]["last_touched_at"] = now_ts
                        removed_route_ids.append(route_id)
                        self._total_removed += 1
                except Exception:
                    # log and continue; do not allow predicate exceptions to abort removal loop
                    # We do a best-effort continue; in production we'd surface logging here
                    continue

        # Method returns None; caller can query pool state to see removals
        return None

    def get_all(self) -> List[Any]:
        """
        Return a list of all active Column objects currently in the pool.

        The returned list is a shallow copy; modifying the returned list does not affect the pool.
        """
        with self._lock:
            result = [col for rid, col in self._columns.items() if self._meta.get(rid, {}).get("is_active", False)]
            return list(result)

    def get_columns_using_arc(self, i: int, j: int) -> List[Any]:
        """
        Return list of active Columns that include arc (i,j).

        Args:
            i: start node index
            j: end node index

        Returns:
            list of Column objects (may be empty)
        """
        with self._lock:
            arc = (int(i), int(j))
            route_ids = self._arc_index.get(arc, set())
            cols = []
            for rid in list(route_ids):
                meta = self._meta.get(rid)
                if meta and meta.get("is_active", False):
                    col = self._columns.get(rid)
                    if col is not None:
                        cols.append(col)
            return cols

    def find_by_route_id(self, route_id: str) -> Optional[Any]:
        """
        Return the Column object with the given route_id if present (active or inactive),
        otherwise return None.
        """
        if route_id is None:
            return None
        with self._lock:
            return self._columns.get(route_id)

    # -----------------------------
    # Internal helpers
    # -----------------------------

    def _compute_route_id_safe(self, column: Any) -> Optional[str]:
        """
        Compute a deterministic route_id for a given column.

        Priority:
          1. If column has attribute 'route_id' (string), use it.
          2. Else, try column.serialize() to get canonical dict and hash it.
          3. Else, try column.route (and inspect route fields) to serialize.
          4. If none succeed, return None.

        The computed id is a short hex SHA1 digest prefixed with 'r_'.
        """
        try:
            # 1) direct attribute
            if hasattr(column, "route_id") and getattr(column, "route_id") not in (None, ""):
                rid = str(getattr(column, "route_id"))
                return rid
        except Exception:
            pass

        # 2) try serialize()
        try:
            if hasattr(column, "serialize") and callable(getattr(column, "serialize")):
                serial = column.serialize()
                # canonical JSON bytes
                payload = json.dumps(serial, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
                if self._salt is not None:
                    payload = payload + b"||salt:" + self._salt.encode("utf-8")
                h = hashlib.sha1(payload).hexdigest()[:16]
                return f"r_{h}"
        except Exception:
            pass

        # 3) try to use column.route structure if available (truck_seq and drone_sorties)
        try:
            if hasattr(column, "route"):
                route = getattr(column, "route")
                route_repr = {}
                if hasattr(route, "truck_seq"):
                    route_repr["truck_seq"] = list(route.truck_seq)
                if hasattr(route, "drone_sorties"):
                    # normalize sorties ordering
                    route_repr["drone_sorties"] = [[int(s), list(cs), int(r)] for (s, cs, r) in route.drone_sorties]
                payload = json.dumps(route_repr, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
                if self._salt is not None:
                    payload = payload + b"||salt:" + self._salt.encode("utf-8")
                h = hashlib.sha1(payload).hexdigest()[:16]
                return f"r_{h}"
        except Exception:
            pass

        return None

    def _persist_column(self, route_id: str, column: Any, meta: Dict[str, Any]) -> None:
        """
        Persist a column to disk as JSON under results_dir/columns/<route_id>.json.

        This method is best-effort and will not raise on failure (caller handles exceptions).
        The JSON includes:
          - meta: provided metadata (created_at, iteration, etc.)
          - route: column.serialize() if available, else a best-effort dict
        """
        # Prepare directory
        try:
            self._columns_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # cannot ensure directory; abort persistence silently
            return

        filename = f"{route_id}.json"
        path = self._columns_dir / filename

        # Build payload
        payload: Dict[str, Any] = {"meta": dict(meta)}
        # try to include cost / reduced cost if available
        try:
            if hasattr(column, "cost"):
                payload["meta"]["cost"] = float(getattr(column, "cost"))
        except Exception:
            pass
        try:
            if hasattr(column, "route") and hasattr(column.route, "serialize"):
                payload["route"] = column.route.serialize()
            elif hasattr(column, "serialize") and callable(getattr(column, "serialize")):
                payload["route"] = column.serialize()
            else:
                # best-effort composition
                payload["route"] = {
                    "truck_arcs": getattr(column, "truck_arcs", []),
                    "drone_arcs": getattr(column, "drone_arcs", []),
                    "covered_customers": getattr(column, "a_ir", {}) or [],
                }
        except Exception:
            # fallback empty
            payload["route"] = {}

        # atomic write
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with open(str(tmp_path), "wt", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True, default=str)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
            os.replace(str(tmp_path), str(path))
        except Exception:
            # best-effort: attempt remove tmp file if exists
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            # swallow exception (persistence is optional)
            return

    # -----------------------------
    # Diagnostics
    # -----------------------------
    def stats(self) -> Dict[str, Any]:
        """
        Return a small diagnostics dict about pool state. Useful for logging.
        """
        with self._lock:
            active = sum(1 for m in self._meta.values() if m.get("is_active", False))
            return {
                "total_columns_stored": len(self._columns),
                "active_columns": int(active),
                "total_added": int(self._total_added),
                "total_removed": int(self._total_removed),
                "unique_arcs_indexed": int(len(self._arc_index)),
                "unique_customers_indexed": int(len(self._customer_index)),
                "iteration_counter": int(self._iteration_counter),
            }


# End of file
