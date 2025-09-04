## cuts/sr_separator.py

"""
cuts/sr_separator.py

Subset-Row (SR) inequality separator for the TD-DRPTW Branch-and-Price-and-Cut.

This module provides the SRSeparator class which, given the current restricted master
primal solution (lambda values for columns), identifies violated subset-row (SR)
inequalities of the form (for |S|=3, p=2):

    sum_{r in R_s} (1/2 * sum_{i in S} a_{i,r}) * lambda_r  <=  1

If the left-hand side (LHS) of an SR exceeds 1 + eps, the SR (S,p) is returned as
a violated cut. The separator enumerates combinations S (complete enumeration for
|S|=3 by default) and returns violated cuts sorted by violation magnitude.

Design / interface (per reproducibility plan):
  - SRSeparator(rlmp_solver)
      rlmp_solver: instance of RLMP_Solver (expected to expose .column_pool and .instance and .config)
  - separate(lambda_primal: Dict[str, float]) -> List[Tuple[List[int], int]]
      lambda_primal: mapping from column_id (route_id) -> lambda value (float)
      returns: list of tuples (S_list, p) where S_list is a list of customer indices (ints) and p is int (2)

Behavior notes:
  - Uses configuration from rlmp_solver.config if present; otherwise loads ./config.yaml if available.
  - Default parameters (when config missing) match the paper:
        sr_cardinality = 3
        sr_p = 2
        sr_enumeration = True
        sr_violation_eps = 1e-8
        sr_max_add_per_iteration = None  (means add all violated SRs)
  - Avoids proposing SRs that were already proposed by this SRSeparator instance (internal cache).
  - Efficient computation: precomputes per-customer aggregated lambda contributions so LHS(S) =
        0.5 * (contrib[i1] + contrib[i2] + contrib[i3])

Robustness:
  - Extracts coverage for each column by trying column.covers(), column.a_ir, column.serialize()['coverage'] variants.
  - If rlmp_solver.instance is not available, falls back to the union of customers observed in the column pool.
  - If no columns or all lambdas are zero, returns empty list quickly.

Author: Reproducibility codebase
"""

from __future__ import annotations

import itertools
import logging
import os
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Try importing yaml for fallback config loading; it's optional but present in requirements.
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml expected but handle gracefully
    yaml = None  # type: ignore

# Type aliases
Arc = Tuple[int, int]


_logger = logging.getLogger(__name__)
_DEFAULT_SR_CARDINALITY = 3
_DEFAULT_SR_P = 2
_DEFAULT_SR_ENUMERATION = True
_DEFAULT_SR_EPS = 1e-8
_DEFAULT_SR_MAX_ADD = None  # means add all violated SRs


class SRSeparator:
    """
    Subset-Row (SR) inequality separator.

    Usage:
        srsep = SRSeparator(rlmp_solver)
        cuts = srsep.separate(lambda_primal)  # returns list of (S_list, p)
        if cuts:
            rlmp_solver.add_sr_cuts(cuts)

    The RLMP solver is expected to provide:
      - rlmp_solver.column_pool with method get_all() returning Column objects
      - rlmp_solver.instance (optional) to obtain number of customers n
      - rlmp_solver.config (optional) a dict-like configuration loaded from config.yaml
    """

    def __init__(self, rlmp_solver: object) -> None:
        if rlmp_solver is None:
            raise ValueError("SRSeparator requires a non-null rlmp_solver instance.")
        self.rlmp = rlmp_solver

        # column_pool expected per design
        try:
            self.col_pool = getattr(self.rlmp, "column_pool")
        except Exception:
            self.col_pool = None

        # instance if available
        self.instance = getattr(self.rlmp, "instance", None)

        # load configuration with safe defaults
        cfg = getattr(self.rlmp, "config", None)
        if cfg is None:
            # attempt to load config.yaml from working directory
            cfg = {}
            try:
                if yaml is not None and os.path.exists("config.yaml"):
                    with open("config.yaml", "rt", encoding="utf-8") as f:
                        loaded = yaml.safe_load(f) or {}
                        if isinstance(loaded, dict):
                            cfg = loaded
            except Exception:
                cfg = {}

        # drill down into column_generation config with safe defaults
        cg = {}
        try:
            cg = cfg.get("column_generation", {}) or {}
        except Exception:
            cg = {}

        self.sr_cardinality: int = int(cg.get("sr_cardinality", _DEFAULT_SR_CARDINALITY))
        self.sr_p: int = int(cg.get("sr_p", _DEFAULT_SR_P))
        self.sr_enumeration: bool = bool(cg.get("sr_enumeration", _DEFAULT_SR_ENUMERATION))
        self.sr_violation_eps: float = float(cg.get("sr_violation_eps", _DEFAULT_SR_EPS))
        # may be None in config -> means add all
        self.sr_max_add_per_iteration: Optional[int] = cg.get("sr_max_add_per_iteration", _DEFAULT_SR_MAX_ADD)
        if self.sr_max_add_per_iteration is not None:
            try:
                self.sr_max_add_per_iteration = int(self.sr_max_add_per_iteration)
            except Exception:
                self.sr_max_add_per_iteration = None

        # internal cache to avoid re-adding same SRs repeatedly from this separator instance
        # store canonical tuples (i1,i2,...,ik)
        self._added_srs: Set[Tuple[int, ...]] = set()

        _logger.debug(
            "SRSeparator initialized: sr_cardinality=%s sr_p=%s sr_enumeration=%s sr_eps=%s sr_max_add=%s",
            self.sr_cardinality,
            self.sr_p,
            self.sr_enumeration,
            self.sr_violation_eps,
            self.sr_max_add_per_iteration,
        )

    def _extract_columns_and_lambda(self, lambda_primal: Dict[str, float]) -> Tuple[List[Tuple[str, float, Set[int]]], Set[int]]:
        """
        Build a list of (col_id, lambda_value, coverage_set) for active columns in column pool.

        Returns:
            cols_info: list of tuples
            customers_set: set of customer indices observed (union of coverage sets)
        """
        cols_info: List[Tuple[str, float, Set[int]]] = []
        customers_set: Set[int] = set()

        if self.col_pool is None:
            _logger.warning("SRSeparator: rlmp_solver has no column_pool attribute; returning no cuts.")
            return cols_info, customers_set

        try:
            columns = self.col_pool.get_all()
        except Exception as ex:
            _logger.exception("SRSeparator: failed to retrieve columns from column_pool: %s", ex)
            return cols_info, customers_set

        for col in columns:
            # determine column id / key used in lambda_primal
            col_id = None
            # preferred attribute names: route_id or route.route_id or id
            if hasattr(col, "route_id"):
                try:
                    col_id = str(getattr(col, "route_id"))
                except Exception:
                    col_id = None
            if col_id is None and hasattr(col, "route") and getattr(col, "route") is not None and hasattr(col.route, "id"):
                try:
                    col_id = str(getattr(col.route, "id"))
                except Exception:
                    col_id = None
            if col_id is None and hasattr(col, "route") and getattr(col, "route") is not None and hasattr(col.route, "_route_id"):
                try:
                    col_id = str(getattr(col.route, "_route_id"))
                except Exception:
                    col_id = None
            if col_id is None and hasattr(col, "serialize") and callable(getattr(col, "serialize")):
                try:
                    ser = col.serialize()
                    # common keys: 'route_id' or 'route_id' inside meta
                    if isinstance(ser, dict):
                        if "route_id" in ser:
                            col_id = str(ser["route_id"])
                        elif "meta" in ser and isinstance(ser["meta"], dict) and ser["meta"].get("route_id"):
                            col_id = str(ser["meta"].get("route_id"))
                except Exception:
                    col_id = None
            if col_id is None:
                # last resort use object's id
                try:
                    col_id = f"obj_{id(col)}"
                except Exception:
                    col_id = "unknown_col"

            # lambda value: if not present default 0.0
            lam = 0.0
            try:
                lam = float(lambda_primal.get(col_id, 0.0))
            except Exception:
                # if key types differ (e.g., lambda_primal uses different id strings),
                # attempt to match by comparing possible alternative ids present in column serialize()
                lam = 0.0
                try:
                    ser = col.serialize() if hasattr(col, "serialize") else {}
                    if isinstance(ser, dict):
                        meta = ser.get("meta", {}) or {}
                        alt_id = meta.get("route_id") or meta.get("id")
                        if alt_id is not None:
                            lam = float(lambda_primal.get(str(alt_id), 0.0))
                except Exception:
                    lam = 0.0

            # extract coverage set
            coverage: Set[int] = set()
            # prefer col.covers() if available
            try:
                if hasattr(col, "covers") and callable(getattr(col, "covers")):
                    cov = col.covers()
                    if isinstance(cov, (set, list, tuple)):
                        coverage = set(int(x) for x in cov)
                elif hasattr(col, "a_ir") and isinstance(getattr(col, "a_ir"), dict):
                    a_ir = getattr(col, "a_ir")
                    for k, v in a_ir.items():
                        try:
                            if int(v) != 0:
                                coverage.add(int(k))
                        except Exception:
                            # keys might be strings - try convert
                            try:
                                ik = int(k)
                                if int(v) != 0:
                                    coverage.add(ik)
                            except Exception:
                                continue
                else:
                    # try serialized representation
                    if hasattr(col, "serialize") and callable(getattr(col, "serialize")):
                        ser = col.serialize()
                        if isinstance(ser, dict):
                            covlist = ser.get("covered_customers") or ser.get("coverage") or ser.get("coverage_list") or ser.get("coverage_set")
                            if isinstance(covlist, (list, tuple, set)):
                                coverage = set(int(x) for x in covlist)
                            else:
                                # sometimes 'route' contains 'covered_customers'
                                route_part = ser.get("route", {}) or {}
                                covlist2 = route_part.get("covered_customers") or route_part.get("coverage")
                                if isinstance(covlist2, (list, tuple, set)):
                                    coverage = set(int(x) for x in covlist2)
                # ensure coverage contains only customer indices (1..n)
                if coverage:
                    # filter out depot indices 0 and n+1 if erroneously present
                    n_val = getattr(self.instance, "n", None)
                    if isinstance(n_val, int):
                        coverage = set(i for i in coverage if 1 <= int(i) <= n_val)
                    else:
                        # best-effort keep positive indices
                        coverage = set(i for i in coverage if int(i) >= 1)
            except Exception:
                # fallback: empty coverage
                coverage = set()

            # accumulate
            cols_info.append((col_id, lam, coverage))
            customers_set.update(coverage)

        return cols_info, customers_set

    def separate(self, lambda_primal: Dict[str, float]) -> List[Tuple[List[int], int]]:
        """
        Perform SR separation on the current RLMP primal solution.

        Parameters:
            lambda_primal: dict mapping column_id (string) -> lambda value (float)

        Returns:
            List of SR cuts as tuples: (S_list, p), where S_list is a list of customer indices (ints)
            and p is integer (e.g., 2). The returned list is ordered by violation magnitude (largest first).
        """
        if not self.sr_enumeration:
            _logger.debug("SRSeparator.separate: SR enumeration disabled by configuration.")
            return []

        # gather column info and observed customers
        cols_info, customers_obs = self._extract_columns_and_lambda(lambda_primal)

        # quick exit: no columns or no customers present
        if not cols_info or not customers_obs:
            _logger.debug("SRSeparator.separate: no columns or no customers observed -> no SRs.")
            return []

        # If instance available, define customers as 1..n; else fallback to observed customers sorted
        customers_list: List[int]
        if getattr(self.instance, "n", None) is not None:
            n_customers = int(self.instance.n)
            customers_list = list(range(1, n_customers + 1))
        else:
            customers_list = sorted(customers_obs)

        # Precompute per-customer aggregated lambda contributions:
        # customer_contrib[i] = sum_{r covers i} lambda_r
        customer_contrib: Dict[int, float] = {i: 0.0 for i in customers_list}
        # populate contributions
        for (_col_id, lam, coverage) in cols_info:
            if not lam:
                # lam==0.0 skip
                continue
            for cust in coverage:
                # only accumulate for customers within expected list
                if cust in customer_contrib:
                    customer_contrib[cust] += float(lam)

        # If all contributions zero, no SR violated
        if all(abs(v) <= 0.0 for v in customer_contrib.values()):
            _logger.debug("SRSeparator.separate: all customer contributions zero -> no SRs.")
            return []

        # enumerate combinations of customers of size k
        k = int(self.sr_cardinality)
        p = int(self.sr_p)
        eps = float(self.sr_violation_eps)

        violations: List[Tuple[Tuple[int, ...], float, float]] = []  # (S_tuple, violation_amount, LHS_value)

        # iterate combinations in deterministic order
        # To reduce work, optionally skip combinations with customers whose contributions are all zero,
        # but simplest is to compute LHS quickly using customer_contrib sums.
        custs = customers_list

        # If number customers less than k, nothing to do
        if len(custs) < k:
            return []

        # Precompute a small list of customers that have non-zero contrib; if many zeros, limit combos
        nonzero_customers = [i for i, v in customer_contrib.items() if float(v) != 0.0]
        # If number of nonzero customers is small and less than k, no violation possible
        if len(nonzero_customers) < k:
            # but even if some customers have zero, combinations with some nonzero may create LHS >1, so we must still check all combinations
            # However if all contributions small enough, can early exit; we'll proceed with enumeration for correctness.
            pass

        # iterate combinations
        # compute LHS(S) = 0.5 * sum(customer_contrib[i] for i in S)
        # RHS = 1.0
        rhs = 1.0
        # itertools.combinations yields tuples in lexicographic order; good determinism
        for S in itertools.combinations(custs, k):
            # skip if we've already added this SR previously (canonical tuple)
            S_key = tuple(sorted(int(x) for x in S))
            if S_key in self._added_srs:
                continue
            # compute LHS using precomputed contributions
            lhs_sum = 0.0
            # minor optimization: sum only non-zero contr but customers small we do direct
            for node in S_key:
                lhs_sum += float(customer_contrib.get(node, 0.0))
            lhs = 0.5 * lhs_sum
            if lhs > rhs + eps:
                violation_amount = lhs - rhs
                violations.append((S_key, violation_amount, lhs))

        if not violations:
            _logger.debug("SRSeparator.separate: no violated SRs found (checked %d combinations).", 0 if not custs else (len(custs) if False else None))
            # For clarity compute number of combinations checked
            try:
                comb_count = 0
                # Avoid expensive computation if many customers; compute exact via math.comb if available
                import math as _math  # local import

                comb_count = int(_math.comb(len(custs), k))
            except Exception:
                comb_count = -1
            _logger.debug("SRSeparator.separate: checked combinations ~= %s; no violations.", comb_count)
            return []

        # sort violations by violation_amount desc to return most violated first
        violations.sort(key=lambda t: float(t[1]), reverse=True)

        # limit number of added SRs if configured
        if self.sr_max_add_per_iteration is not None and isinstance(self.sr_max_add_per_iteration, int) and self.sr_max_add_per_iteration > 0:
            chosen = violations[: self.sr_max_add_per_iteration]
        else:
            chosen = violations

        cuts: List[Tuple[List[int], int]] = []
        for S_key, viol_amt, lhs_val in chosen:
            # prepare cut as list[int]
            S_list = list(S_key)
            cuts.append((S_list, p))
            # record as added to avoid duplicates in subsequent calls
            try:
                self._added_srs.add(S_key)
            except Exception:
                # if adding to set fails for some reason, continue without caching
                pass

        # logging summary
        try:
            top_info = [{"S": list(v[0]), "violation": float(v[1]), "lhs": float(v[2])} for v in violations[: min(len(violations), 5)]]
        except Exception:
            top_info = []
        _logger.info(
            "SRSeparator.separate: found %d violated SRs (returning %d). Top violations: %s",
            len(violations),
            len(cuts),
            top_info,
        )
        return cuts
