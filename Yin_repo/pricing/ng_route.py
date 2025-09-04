## pricing/ng_route.py

"""
pricing/ng_route.py

Dynamic ng-route relaxation pricing (NgRoutePricer) for the TD-DRPTW reproduction project.

This module implements the NgRoutePricer class that performs the dynamic ng-route
relaxation described in the paper and the reproducibility plan. It repeatedly calls
the exact Labeler in "ng-mode" (Labeler must consult an attribute `ng_neighborhoods`
when present) and, when the best ng-route contains cycles (repeated customer visits),
augments the NG neighborhoods accordingly (Roberti & Mingozzi dynamic augmentation).
NgRoutePricer returns feasible elementary columns (synthetic-routes) with negative
reduced cost to be added to the restricted master problem (RLMP).

Design and behavior highlights:
 - Deterministic NG initialization via k-nearest neighbours on drone (Euclidean) distance.
 - Per-round time management: splits provided time_limit among augmentation rounds,
   respecting labeler-specific time caps if configured.
 - Robust handling of RLMP dual formats:
     - expects duals['pi'] mapping customer -> dual
     - optional duals['sigma'] scalar
     - optional SR dual info in duals['zeta'] and duals['SR_sets'] (several accepted shapes)
 - Careful logging for reproducibility (uses provided logger if available).
 - Conservative defaults provided for unspecified config keys to ensure safe execution.
   These defaults are explicit and documented; users should override via config.yaml.

Public API:
 - class NgRoutePricer
     - __init__(instance, distances, labeler, column_pool, config=None, logger=None, rng=None)
     - price(duals, time_limit, forbidden_arcs=None, forced_arcs=None) -> List[Column]

Dependencies:
 - instances.instance.Instance
 - geometry.distances.DistanceMatrix
 - pricing.labeler.Labeler  (an instance is passed in; Labeler must honor attribute ng_neighborhoods)
 - columns.column_pool.ColumnPool
 - routes.route.Route and routes.route.Column
 - numpy

Author: Reproducibility implementation
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

# Project imports
from instances.instance import Instance
from geometry.distances import DistanceMatrix
from routes.route import Route, Column
from columns.column_pool import ColumnPool

# Labeler is used by PricingManager and passed into NgRoutePricer; import type for annotation only
try:
    from pricing.labeler import Labeler  # type: ignore
except Exception:
    Labeler = Any  # fallback typing if import fails at static-check time

# Logger optional
try:
    from utils.logger import Logger  # type: ignore
except Exception:
    Logger = None  # type: ignore

Arc = Tuple[int, int]


def _now_s() -> float:
    return time.time()


class NgRoutePricer:
    """
    Implements the dynamic ng-route relaxation pricing procedure.

    Parameters:
      - instance: Instance object (problem data)
      - distances: DistanceMatrix (must have compute_all() called)
      - labeler: an instance of pricing.labeler.Labeler (exact bounded bidirectional labeler)
                 NgRoutePricer sets labeler.ng_neighborhoods before calling labeler.price(...)
                 so Labeler must consult that attribute if present.
      - column_pool: ColumnPool for deduplication/persistence (optional but recommended)
      - config: dict-like configuration (can be from config.yaml). Expected keys (with defaults):
           config.pricing.ng_initial_neighborhood_size: int (k nearest neighbors). Default: 1 -> trivial NG_i={i}.
           config.pricing.ng_max_augment_rounds: int. Default: 5
           config.pricing.labeler_time_limit_seconds: float. Default: 10.0
           config.pricing.labeler_per_round_fraction: float in (0,1]. Fraction of overall time_limit to allow per round if labeler_time_limit_seconds not set. Default: 0.9
           config.pricing.ng_return_top_k: int. Limit number of columns returned. Default: 50
           config.solver.lp_tolerance: float. Reduced-cost tolerance (if None default 1e-6)
      - logger: optional Logger instance per utils/logger.Logger
      - rng: numpy.RandomState for deterministic sampling if needed; default seeded from 0 if not provided.

    Usage:
      pricer = NgRoutePricer(instance, distances, labeler, column_pool, config)
      columns = pricer.price(duals, time_limit=30.0, forbidden_arcs=set(), forced_arcs=set())
    """

    def __init__(
        self,
        instance: Instance,
        distances: DistanceMatrix,
        labeler: "Labeler",
        column_pool: ColumnPool,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[Logger] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        if instance is None or distances is None or labeler is None or column_pool is None:
            raise ValueError("NgRoutePricer requires non-null instance, distances, labeler and column_pool.")

        self.instance = instance
        self.distances = distances
        self.labeler = labeler
        self.column_pool = column_pool
        self.config = dict(config or {})
        self.logger = logger
        self.rng = rng if rng is not None else np.random.RandomState(0)

        # Read config defaults (explicit defaults as required)
        pr = self.config.get("pricing", {}) or {}
        heuristic = pr.get("heuristic_params", {}) or {}
        solver_cfg = self.config.get("solver", {}) or {}

        self.k_ng_initial = int(pr.get("ng_initial_neighborhood_size", heuristic.get("ng_initial_neighborhood_size", 1)))
        # ensure at least 1
        if self.k_ng_initial < 1:
            self.k_ng_initial = 1

        self.max_augment_rounds = int(pr.get("ng_max_augment_rounds", heuristic.get("ng_max_augment_rounds", 5)))
        if self.max_augment_rounds < 1:
            self.max_augment_rounds = 1

        # per-labeler cap (seconds) used as min(per-round limit, remaining_time)
        self.labeler_time_limit_seconds = float(pr.get("labeler_time_limit_seconds", heuristic.get("labeler_time_limit_seconds", 10.0)))
        if self.labeler_time_limit_seconds <= 0.0:
            self.labeler_time_limit_seconds = 10.0

        # Fraction of remaining time to give to labeler if explicit labeler_time_limit_seconds not enforced
        self.per_round_fraction = float(pr.get("labeler_per_round_fraction", heuristic.get("labeler_per_round_fraction", 0.9)))
        if not (0.0 < self.per_round_fraction <= 1.0):
            self.per_round_fraction = 0.9

        # How many columns to return at most
        self.return_top_k = int(pr.get("ng_return_top_k", heuristic.get("ng_return_top_k", 50)))
        if self.return_top_k <= 0:
            self.return_top_k = 50

        # reduced cost tolerance
        self.rc_tol = float(solver_cfg.get("lp_tolerance", 1e-6))

        # safety minimal per-round labeler time (seconds)
        self.min_labeler_time = float(pr.get("labeler_min_time", heuristic.get("labeler_min_time", 0.5)))
        if self.min_labeler_time <= 0.0:
            self.min_labeler_time = 0.5

        # store last NG sets for inspection
        self.last_ng_sets: Dict[int, Set[int]] = {}

        # Ensure distances computed
        if not getattr(self.distances, "_computed", False):
            self.distances.compute_all()

    # -------------------------
    # Utility helpers
    # -------------------------
    def _log(self, event: str, payload: Dict[str, Any]) -> None:
        """Log via provided logger if available; otherwise silent."""
        if self.logger is not None:
            try:
                self.logger.log(event, payload)
            except Exception:
                # don't fail due to logging issues
                pass

    def _init_ng_sets(self) -> Dict[int, Set[int]]:
        """
        Initialize NG_sets: Dict[node -> set of neighbor customers]. Deterministic.

        Policy:
          - For each customer i in 1..n, NG_i is the set of k nearest customers by Euclidean drone distance,
            including i itself. Ties broken by node id ascending.
          - If k >= n then NG_i = all customers.
        Returns dict mapping customer id -> set of customer ids.
        """
        n = self.instance.n
        customers = list(range(1, n + 1))
        NG: Dict[int, Set[int]] = {}
        k = min(self.k_ng_initial, n) if n > 0 else 0
        # Precompute distance matrix for customers using DistanceMatrix.d_drone
        # Use deterministic ordering
        for i in customers:
            # compute distances to all customers j
            dlist = []
            for j in customers:
                try:
                    d = float(self.distances.d_drone(i, j))
                except Exception:
                    # fallback large distance
                    d = float("inf")
                dlist.append((d, j))
            # sort by distance then node id
            dlist.sort(key=lambda x: (float(x[0]), int(x[1])))
            # pick k nearest (include i)
            chosen = [int(j) for (_, j) in dlist[:k]]
            # ensure i in chosen
            if i not in chosen:
                # replace last with i deterministically
                if len(chosen) < k:
                    chosen.append(i)
                else:
                    chosen[-1] = i
            NG[i] = set(chosen)
        return NG

    @staticmethod
    def _detect_cycles_in_sequence(seq: List[int]) -> List[Set[int]]:
        """
        Detect cycles (repeated nodes) in a visit sequence seq (list of customer node ids).
        Returns list of cycle node sets. Each cycle set is a set of distinct node ids appearing
        in the subsequence from one occurrence to the next occurrence of that node (inclusive).

        Example:
          seq = [2,5,3,7,5,8] -> repeated node 5 occurs at indices 1 and 4 -> cycle nodes {5,3,7}
        Multiple overlapping repeats produce multiple cycle sets.
        The returned list contains cycle sets for each first-to-next occurrence pair.
        """
        cycles = []
        last_pos: Dict[int, int] = {}
        for idx, node in enumerate(seq):
            if node in last_pos:
                start = last_pos[node]
                end = idx
                # Extract nodes between start and end (inclusive)
                cycle_nodes = set(seq[start:end + 1])
                cycles.append(cycle_nodes)
            last_pos[node] = idx
        # Remove duplicates (if identical sets appear)
        uniq = []
        seen = set()
        for s in cycles:
            key = tuple(sorted(s))
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        return uniq

    @staticmethod
    def _combine_schedule_sequence(route: Route) -> List[int]:
        """
        Build combined chronological service sequence S for the given Route.
        We rely on Route.is_feasible() having been called so route._cached_feasibility contains 'schedule'.
        The schedule format from routes/route.is_feasible():
          schedule = {"truck_schedule": [...], "drone_sorties": [...]}
        - truck_schedule entries: dict with keys 'node', 'service_start' (start), 'service_end'
        - drone_sorties entries: dict with keys 'customers' -> list of dicts with keys 'node', 'start' etc.
        Combine all customer-service events (excluding depots) using event time chosen as:
          - truck customer: service_start
          - drone customer: start (drone service start)
        Sort events by event time asc; ties by node id to be deterministic.
        Return list of customer node ids in chronological service order.
        """
        seq: List[int] = []
        # Attempt to access cached schedule; if not available, try route.serialize() which may include schedule
        try:
            feasible, reason = route.is_feasible()
            if not feasible:
                # Even if infeasible, try to use serialize schedule if present
                ser = route.serialize()
                schedule = ser.get("schedule", {}) or {}
            else:
                # route.is_feasible sets _cached_feasibility third element as schedule
                _, _, sched = route._cached_feasibility
                schedule = sched or {}
        except Exception:
            ser = route.serialize()
            schedule = ser.get("schedule", {}) or {}

        events = []
        # truck schedule
        truck_sched = schedule.get("truck_schedule", []) or []
        for ev in truck_sched:
            node = ev.get("node")
            if node is None:
                continue
            if not (1 <= int(node) <= route.instance.n):
                continue
            # choose service_start if present else arrival
            t = ev.get("service_start", ev.get("arrival", 0.0))
            try:
                t = float(t)
            except Exception:
                t = 0.0
            events.append((t, int(node), "truck"))
        # drone sorties
        dsorts = schedule.get("drone_sorties", []) or []
        for s in dsorts:
            custs = s.get("customers", []) or []
            for cust_ev in custs:
                node = cust_ev.get("node")
                if node is None:
                    continue
                if not (1 <= int(node) <= route.instance.n):
                    continue
                t = cust_ev.get("start", cust_ev.get("arrival", 0.0))
                try:
                    t = float(t)
                except Exception:
                    t = 0.0
                events.append((t, int(node), "drone"))
        # sort by time then node id for determinism
        events.sort(key=lambda x: (float(x[0]), int(x[1])))
        seq = [int(ev[1]) for ev in events]
        return seq

    def _compute_reduced_cost_from_duals(self, col: Column, duals: Dict[str, Any]) -> float:
        """
        Compute reduced cost for a column using duals:
          rc = c_r - sum_i a_ir * pi_i - sum_S (1/2 * (sum_{i in S} a_ir) * zeta_S) - sigma

        Accepts multiple dual formats:
          - duals['pi']: mapping int->float OR string keys convertible to int
          - duals['sigma']: scalar
          - SR duals:
              - duals['zeta']: mapping where keys may be ints (indices) or tuple/list or stringified tuple;
                if keys are ints then duals['SR_sets'] must be provided mapping index->iterable of nodes in S
              - duals['SR_sets']: mapping index->iterable nodes (optional)
        If SR info not found, SR term is zero.
        """
        # cost c_r
        try:
            c_r = float(col.cost)
        except Exception:
            try:
                c_r = float(col.route.cost())
            except Exception:
                c_r = float("inf")

        # pi extraction
        pi_raw = duals.get("pi", {}) or {}
        pi_map: Dict[int, float] = {}
        for k, v in pi_raw.items():
            try:
                ik = int(k)
                pi_map[ik] = float(v)
            except Exception:
                # skip non-int keys
                continue

        sigma = float(duals.get("sigma", 0.0))

        # base sum pi
        sum_pi = 0.0
        for i, a in col.a_ir.items():
            if int(a) != 0:
                sum_pi += float(pi_map.get(int(i), 0.0))

        # SR handling
        sr_term = 0.0
        zeta_raw = duals.get("zeta", {}) or {}
        sr_sets = duals.get("SR_sets", {}) or {}
        # If zeta_raw keys are ints and sr_sets provides sets mapping, use that
        if isinstance(zeta_raw, dict) and len(zeta_raw) > 0:
            for k, zval in zeta_raw.items():
                # determine S nodes
                nodes = None
                if isinstance(k, int):
                    # look up in sr_sets
                    nodes = sr_sets.get(k)
                elif isinstance(k, (list, tuple)):
                    nodes = tuple(int(x) for x in k)
                elif isinstance(k, str):
                    s = k.strip()
                    if s.startswith("(") and s.endswith(")"):
                        s = s[1:-1]
                    parts = [p.strip() for p in s.split(",") if p.strip()]
                    try:
                        nodes = tuple(int(x) for x in parts)
                    except Exception:
                        nodes = None
                else:
                    nodes = None
                if nodes is None:
                    # if sr_sets contains mapping for string key
                    possible = sr_sets.get(k)
                    if possible:
                        nodes = tuple(int(x) for x in possible)
                if nodes is None:
                    # cannot interpret, skip
                    continue
                # count how many nodes in S visited by col
                cnt = sum(1 for node in nodes if int(node) in col.a_ir and int(col.a_ir[int(node)]) != 0)
                if cnt > 0:
                    sr_term += 0.5 * float(cnt) * float(zval)
        else:
            # no zeta info; sr_term remains 0
            sr_term = 0.0

        rc = float(c_r) - float(sum_pi) - float(sr_term) - float(sigma)
        return rc

    # -------------------------
    # Core dynamic-ng procedure
    # -------------------------
    def price(
        self,
        duals: Dict[str, Any],
        time_limit: Optional[float],
        forbidden_arcs: Optional[Set[Arc]] = None,
        forced_arcs: Optional[Set[Arc]] = None,
    ) -> List[Column]:
        """
        Execute dynamic ng-route relaxation pricing.

        Arguments:
          - duals: RLMP duals (dict) with at least 'pi' mapping
          - time_limit: seconds allowed for this entire pricing call (float) or None
          - forbidden_arcs: set of arcs forbidden by branching
          - forced_arcs: set of arcs forced by branching

        Returns:
          - List[Column] (feasible elementary synthetic-route columns) with negative reduced cost.
        """
        t_start = _now_s()
        deadline = t_start + (float(time_limit) if time_limit is not None else float("inf"))
        forbidden_arcs = set(forbidden_arcs or set())
        forced_arcs = set(forced_arcs or set())

        # quick sanity: if time budget too small, exit early
        remaining_time = max(0.0, deadline - _now_s())
        if remaining_time < self.min_labeler_time:
            self._log("ng_price_short_time", {"remaining_time_s": remaining_time})
            return []

        # initialize NG_sets deterministically
        NG_sets = self._init_ng_sets()
        # expose to Labeler via attribute; Labeler is expected to check labeler.ng_neighborhoods when present
        try:
            setattr(self.labeler, "ng_neighborhoods", NG_sets)
        except Exception:
            # cannot set attribute; log and abort ng-route attempt
            self._log("ng_labeler_attr_fail", {"msg": "labeler.ng_neighborhoods attribute could not be set."})
            return []

        self.last_ng_sets = {k: set(v) for k, v in NG_sets.items()}
        self._log("ng_init", {"k": self.k_ng_initial, "ng_stats": {k: len(v) for k, v in NG_sets.items()}})

        # storage for candidate elementary columns
        candidate_columns: Dict[str, Column] = {}
        # Save best non-elementary routes for potential repair fallback
        best_non_elem_routes: List[Column] = []

        # dynamic augmentation loop
        round_idx = 0
        any_augmented = False
        while round_idx < self.max_augment_rounds and _now_s() < deadline:
            round_idx += 1
            # compute per-round time budget
            remaining_time = max(0.0, deadline - _now_s())
            # prefer labeler_time_limit_seconds but also cannot exceed remaining_time
            per_round_limit = min(self.labeler_time_limit_seconds, remaining_time * self.per_round_fraction)
            if per_round_limit < self.min_labeler_time and remaining_time >= self.min_labeler_time:
                per_round_limit = min(self.min_labeler_time, remaining_time)
            if per_round_limit < self.min_labeler_time:
                # insufficient time to run labeler
                break

            round_t0 = _now_s()
            self._log("ng_round_start", {"round": round_idx, "per_round_limit_s": per_round_limit, "remaining_time_s": remaining_time})

            # Set labeler.ng_neighborhoods to current NG_sets just before invocation (ensure updated)
            try:
                setattr(self.labeler, "ng_neighborhoods", NG_sets)
            except Exception:
                # best-effort; if failing we still try to call labeler
                pass

            # Call labeler.price(...) in ng-mode. Labeler must honor ng_neighborhoods attribute.
            try:
                cols: List[Column] = self.labeler.price(duals=duals, forbidden_arcs=forbidden_arcs, forced_arcs=forced_arcs, time_limit=per_round_limit)
            except Exception as ex:
                # If Labeler fails, log and stop ng-route pricing
                self._log("ng_labeler_exception", {"round": round_idx, "error": str(ex)})
                break

            round_t1 = _now_s()
            used = round_t1 - round_t0
            self._log("ng_round_labeler_return", {"round": round_idx, "n_cols": len(cols), "time_used_s": used})

            if not cols:
                # No columns found in this round; stop augmentation
                break

            # Analyze columns returned by labeler: sort by reduced cost (if attribute present) else compute
            def col_reduced_cost(col: Column) -> float:
                # try attribute first
                rc = None
                rc = getattr(col, "reduced_cost", None)
                if rc is None:
                    try:
                        rc = float(self._compute_reduced_cost_from_duals(col, duals))
                    except Exception:
                        rc = float("inf")
                return float(rc)

            cols = sorted(cols, key=lambda c: col_reduced_cost(c))
            # Consider up to some number of best columns to analyze for cycles (limit to avoid explosion)
            analyze_limit = min(len(cols), max(1, int(self.return_top_k / 5)))
            augmented_this_round = False

            for idx, col in enumerate(cols[:analyze_limit]):
                rc_val = col_reduced_cost(col)
                # Only consider columns with sufficiently negative reduced cost
                if rc_val > -self.rc_tol:
                    continue

                # Ensure route has schedule (Route.is_feasible() populates schedule). If not, call is_feasible
                try:
                    feasible, reason = col.route.is_feasible()
                except Exception:
                    feasible = False
                    reason = "is_feasible_exception"
                if not feasible:
                    # If Labeler returned infeasible route, skip but record it
                    self._log("ng_labeler_returned_infeasible", {"route_id": getattr(col, "route_id", None), "reason": reason})
                    continue

                # Build combined visit sequence S
                S = self._combine_schedule_sequence(col.route)
                if not S:
                    # route does not serve any customers? skip
                    continue

                # detect cycles
                cycles = self._detect_cycles_in_sequence(S)
                if not cycles:
                    # elementary route found -> accept and add to candidate columns
                    rid = getattr(col, "route_id", None) or f"auto_{hash(tuple(S))}"
                    if rid not in candidate_columns:
                        # ensure full feasibility and compute rc again for safety
                        try:
                            if not col.route.is_feasible()[0]:
                                continue
                        except Exception:
                            continue
                        # attach reduced_cost attribute if not present
                        try:
                            setattr(col, "reduced_cost", float(rc_val))
                        except Exception:
                            pass
                        candidate_columns[rid] = col
                        self.column_pool.add(col)  # best-effort persistence/dedup
                        self._log("ng_accept_elementary", {"round": round_idx, "route_id": rid, "reduced_cost": float(rc_val)})
                    # Optionally we can stop early upon finding enough columns
                    if len(candidate_columns) >= self.return_top_k:
                        break
                else:
                    # record non-elementary route for fallback repair
                    best_non_elem_routes.append(col)
                    # Perform augmentation: for each cycle, add all cycle nodes into NG sets of nodes in cycle
                    for cyc in cycles:
                        # ensure all nodes in cycle are customers (should be)
                        cyc_customers = set(int(x) for x in cyc if 1 <= int(x) <= self.instance.n)
                        if not cyc_customers:
                            continue
                        for v in cyc_customers:
                            before = set(NG_sets.get(v, set()))
                            unioned = before.union(cyc_customers)
                            if unioned != before:
                                NG_sets[v] = set(unioned)
                                augmented_this_round = True
                                any_augmented = True
                    # log cycles summary
                    self._log("ng_detected_cycles", {"round": round_idx, "route_id": getattr(col, "route_id", None), "num_cycles": len(cycles), "cycles": [sorted(list(x)) for x in cycles]})
                    # update labeler.ng_neighborhoods for next run
                    try:
                        setattr(self.labeler, "ng_neighborhoods", NG_sets)
                    except Exception:
                        pass

            # update last_ng_sets
            self.last_ng_sets = {k: set(v) for k, v in NG_sets.items()}
            # if we found enough elementary columns, break
            if len(candidate_columns) >= self.return_top_k:
                break

            # if augmented this round, continue to next round (if time remains). Otherwise stop.
            if not augmented_this_round:
                # No augmentation: either columns were elementary or no cycles/changing needed.
                break

            # check remaining time to decide whether to continue
            if _now_s() >= deadline:
                break

        # After augmentation rounds: if no elementary columns found but we have best_non_elem_routes, attempt simple repair (shortcutting)
        if not candidate_columns and best_non_elem_routes:
            # attempt repair for up to a limited number of best non-elementary routes
            repairs_tried = 0
            for col in sorted(best_non_elem_routes, key=lambda c: float(getattr(c, "reduced_cost", self._compute_reduced_cost_from_duals(c, duals)))):
                if repairs_tried >= 5 or _now_s() >= deadline:
                    break
                repaired = self._try_shortcut_repair(col, duals)
                repairs_tried += 1
                if repaired is not None:
                    rc_repaired = float(getattr(repaired, "reduced_cost", self._compute_reduced_cost_from_duals(repaired, duals)))
                    if rc_repaired < -self.rc_tol:
                        rid = getattr(repaired, "route_id", None) or f"repaired_{repairs_tried}"
                        candidate_columns[rid] = repaired
                        self.column_pool.add(repaired)
                        self._log("ng_repair_accepted", {"repaired_route_id": rid, "reduced_cost": rc_repaired})
                        # if enough, stop
                        if len(candidate_columns) >= self.return_top_k:
                            break

        # finalize: sort candidate columns by reduced cost (most negative first) and return up to return_top_k
        final_cols = list(candidate_columns.values())
        final_cols.sort(key=lambda c: float(getattr(c, "reduced_cost", self._compute_reduced_cost_from_duals(c, duals))))
        final_cols = final_cols[: self.return_top_k]
        total_time = _now_s() - t_start
        self._log("ng_price_completed", {"rounds": round_idx, "num_returned": len(final_cols), "time_s": total_time, "any_augmented": any_augmented})
        return final_cols

    # -------------------------
    # Light-weight repair heuristics
    # -------------------------
    def _try_shortcut_repair(self, col: Column, duals: Dict[str, Any]) -> Optional[Column]:
        """
        Attempt a simple deterministic shortcut repair on a non-elementary route (Column).
        Strategy:
          - Build combined visit sequence S
          - For the first detected repeated node i with occurrences at positions (p,q), attempt to remove
            the subsequence S[p+1:q] (i.e., remove the cycle interior) and reconstruct a candidate route
            by removing corresponding customer service calls from the original Route:
              - If removed customers were served by drone sorties, remove them from the drone sortie
                sequence and attempt to keep route feasible.
              - If removed customers were served by truck, remove them from truck_seq.
            - Reconstruct Route and validate feasibility; if feasible compute reduced cost and return Column.
        This shortcut is conservative and simple; it may fail often but is tractable and deterministic.
        """
        try:
            route = col.route
        except Exception:
            return None

        # get combined sequence
        S = self._combine_schedule_sequence(route)
        if not S:
            return None
        # detect first repeated node pair
        last_pos = {}
        first_pair = None
        for idx, node in enumerate(S):
            if node in last_pos:
                first_pair = (last_pos[node], idx, node)
                break
            last_pos[node] = idx
        if first_pair is None:
            return None
        p, q, repeated_node = first_pair
        # nodes to remove = S[p+1 : q] (interior)
        to_remove = set(S[p + 1 : q])
        if not to_remove:
            return None

        # Attempt to build new truck_seq and new drone_sorties by removing customers in to_remove
        try:
            old_truck_seq = list(route.truck_seq)
            old_drone_sorties = list(route.drone_sorties)
        except Exception:
            return None

        # remove to_remove from truck_seq
        new_truck_seq = [n for n in old_truck_seq if n not in to_remove]
        # ensure start and end depots present
        if new_truck_seq[0] != 0:
            new_truck_seq = [0] + new_truck_seq
        if new_truck_seq[-1] != self.instance.n + 1:
            new_truck_seq = new_truck_seq + [self.instance.n + 1]

        # rebuild drone_sorties by removing customers in to_remove; if any sortie becomes empty, collapse it
        new_drone_sorties = []
        for sep, custs, r in old_drone_sorties:
            new_custs = [c for c in custs if c not in to_remove]
            # If sep or rendezvous removed or sortie empty, skip that sortie (drone not used for it)
            if sep in to_remove or r in to_remove:
                continue
            if len(new_custs) == 0:
                # convoying case: if both sep==r maybe okay, but skip empty sortie
                continue
            new_drone_sorties.append((sep, new_custs, r))

        # Build candidate Route
        try:
            repaired_route = Route(truck_seq=new_truck_seq, drone_sorties=new_drone_sorties, instance=self.instance, distances=self.distances, config=self.config)
        except Exception:
            return None

        # validate feasibility
        try:
            feasible, reason = repaired_route.is_feasible()
        except Exception:
            feasible = False
            reason = "is_feasible_exception"
        if not feasible:
            return None

        # compute reduced cost
        try:
            # build Column
            new_col = repaired_route.to_column()
        except Exception:
            # attempt to create Column manually
            try:
                route_id = f"repair_{hash(tuple(new_truck_seq))}"
                a_ir = {i: 1 if i in repaired_route.covers() else 0 for i in range(1, self.instance.n + 1)}
                truck_arcs = repaired_route.compute_truck_arcs()
                drone_arcs = repaired_route.compute_drone_arcs()
                new_col = Column(route_id=route_id, route=repaired_route, a_ir=a_ir, cost=float(repaired_route.cost()), truck_arcs=truck_arcs, drone_arcs=drone_arcs)
            except Exception:
                return None

        # compute reduced cost and attach
        try:
            rc = float(self._compute_reduced_cost_from_duals(new_col, duals))
            setattr(new_col, "reduced_cost", rc)
        except Exception:
            return None

        return new_col

