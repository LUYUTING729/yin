## tests/test_small_instances.py

"""
Unit and integration tests for small instances of the TD-DRPTW reproduction project.

These tests validate core building blocks on tiny instances (n <= 6) so brute-force
or lightweight checks are feasible.

Tests included:
 - config fixture: loads config.yaml and sets conservative defaults for missing
   service times and heuristic hyperparameters so tests run.
 - instance fixture: produces a small instance deterministically.
 - test_distance_matrix_relation: checks d_truck ≈ beta * d_drone and time computations.
 - test_truck_only_route_feasibility_and_column: builds a simple truck-only route and
   checks feasibility, cost and coverage.
 - test_column_pool_basic_operations: add/get/remove columns and arc lookups.
 - test_rlmp_seeding_and_sr_separator: seeds RLMP with single-customer routes, solves LP,
   computes SR violations by enumeration and checks SRSeparator outputs (skipped if CPLEX/docplex not installed).

Notes:
 - These tests attempt to be robust: they check for availability of optional backends
   (docplex) and skip SR-related tests if solver is not available.
 - They do not assume global modification of files; they pass a modified config dict
   into constructors that accept `config` arguments.
"""

from __future__ import annotations

import itertools
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pytest
import yaml

# Project imports (these must be importable when running tests)
from instances.instance import Instance
from instances.instance_generator import InstanceGenerator
from geometry.distances import DistanceMatrix
from routes.route import Route, Column
from columns.column_pool import ColumnPool

# Optional imports for RLMP/SR tests; importorskip if not available
docplex = pytest.importorskip("docplex", reason="docplex (CPLEX) required for RLMP/SR tests; skipping those tests if unavailable")

from master.rlmp_solver import RLMP_Solver
from cuts.sr_separator import SRSeparator


# Constants / tolerances
EPS = 1e-6


# ----------------------------
# Fixtures
# ----------------------------
@pytest.fixture(scope="session")
def config_dict() -> Dict:
    """
    Load config.yaml from repository root and apply conservative defaults for:
     - service_times.truck_service_time_minutes
     - service_times.drone_service_time_minutes
     - pricing.heuristic_params.* (few defaults to allow heuristics to run)
    The tests MUST NOT modify the on-disk config.yaml; they operate on an in-memory copy.
    """
    cfg_path = Path("config.yaml")
    if not cfg_path.exists():
        pytest.skip("config.yaml not found in repository root; skipping tests that need config.")

    with cfg_path.open("rt", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Ensure problem_parameters exist with minimal required keys (fall back to paper defaults if missing)
    problem = cfg.get("problem_parameters", {}) or {}
    problem.setdefault("Q_t", 100)
    problem.setdefault("Q_d", 20)
    problem.setdefault("L_t", 480)
    problem.setdefault("L_d", 30)
    problem.setdefault("v_t_kmph", 40.0)
    problem.setdefault("v_d_kmph", 40.0)
    problem.setdefault("beta", 2.0)
    cfg["problem_parameters"] = problem

    # Cost parameters
    costs = cfg.get("cost_parameters", {}) or {}
    costs.setdefault("fixed_vehicle_cost_F", 20.0)
    costs.setdefault("truck_cost_per_min_c_t", 0.083)
    costs.setdefault("drone_cost_per_min_c_d", 0.021)
    cfg["cost_parameters"] = costs

    # Service times: tests require explicit numeric values. Provide conservative defaults
    svc = cfg.get("service_times", {}) or {}
    svc.setdefault("truck_service_time_minutes", 10.0)
    svc.setdefault("drone_service_time_minutes", 5.0)
    cfg["service_times"] = svc

    # Pricing heuristic params: ensure minimal sensible defaults to run heuristics
    pricing = cfg.get("pricing", {}) or {}
    heur = pricing.get("heuristic_params", {}) or {}
    heur.setdefault("greedy_random_restarts", 5)
    heur.setdefault("greedy_random_top_k", 5)
    heur.setdefault("tabu_tenure", 7)
    heur.setdefault("tabu_max_iterations", 50)
    heur.setdefault("tabu_no_improve_limit", 10)
    heur.setdefault("ng_initial_neighborhood_size", 1)
    heur.setdefault("labeler_time_limit_seconds", 5.0)
    pricing["heuristic_params"] = heur
    cfg["pricing"] = pricing

    # Column generation defaults
    cg = cfg.get("column_generation", {}) or {}
    cg.setdefault("sr_cardinality", 3)
    cg.setdefault("sr_p", 2)
    cg.setdefault("sr_violation_eps", 1e-8)
    cfg["column_generation"] = cg

    # Time limits (light for tests)
    times = cfg.get("time_limits", {}) or {}
    times.setdefault("per_pricing_call_seconds", 5.0)
    times.setdefault("performance_small_seconds", 60.0)
    cfg["time_limits"] = times

    return cfg


@pytest.fixture(scope="session")
def small_instance(config_dict) -> Instance:
    """
    Generate a small instance (n=5) deterministically using InstanceGenerator with provided config.
    Returns an Instance object.
    """
    gen = InstanceGenerator(config=config_dict)
    # Use type 1, n=5, theta=0.4, replicate 0 for deterministic small instance
    inst = gen.generate_one(type_index=1, n=5, theta=0.4, replicate=0)
    # validate instance explicitly
    inst.validate(raise_on_error=True, auto_fix=False)
    return inst


@pytest.fixture(scope="session")
def distances(small_instance, config_dict) -> DistanceMatrix:
    dm = DistanceMatrix(small_instance)
    dm.compute_all()
    return dm


@pytest.fixture
def column_pool() -> ColumnPool:
    return ColumnPool(config={})


# ----------------------------
# Tests
# ----------------------------
def test_distance_matrix_relation(small_instance: Instance, distances: DistanceMatrix, config_dict):
    """
    Validate that d_truck ≈ beta * d_drone for several random node pairs
    and that travel times use speeds and service times in config.
    """
    beta_cfg = float(config_dict["problem_parameters"]["beta"])
    v_t = float(config_dict["problem_parameters"]["v_t_kmph"])
    v_d = float(config_dict["problem_parameters"]["v_d_kmph"])
    truck_svc = float(config_dict["service_times"]["truck_service_time_minutes"])
    drone_svc = float(config_dict["service_times"]["drone_service_time_minutes"])

    nodes = list(range(0, small_instance.n + 2))
    # sample some pairs
    pairs = [(0, 1), (1, 2), (2, 3), (3, small_instance.n + 1)]
    for (i, j) in pairs:
        d_d = distances.d_drone(i, j)
        d_t = distances.d_truck(i, j)
        # check relation within relative tolerance
        assert math.isfinite(d_d) and math.isfinite(d_t)
        assert abs(d_t - beta_cfg * d_d) <= 1e-6, f"d_truck vs beta*d_drone mismatch for pair {(i,j)}: {d_t} vs {beta_cfg*d_d}"
        # travel times
        t_d = distances.t_drone(i, j)
        t_t = distances.t_truck(i, j)
        # travel-only components
        travel_only_d = (d_d / v_d) * 60.0
        travel_only_t = (d_t / v_t) * 60.0
        # t_d should equal travel_only_d + drone service time at j (depots have 0)
        if j == 0 or j == small_instance.n + 1:
            expected_td = travel_only_d
            expected_tt = travel_only_t
        else:
            expected_td = travel_only_d + drone_svc
            expected_tt = travel_only_t + truck_svc
        assert abs(t_d - expected_td) <= 1e-6, f"t_drone mismatch for ({i},{j}): got {t_d}, expected {expected_td}"
        assert abs(t_t - expected_tt) <= 1e-6, f"t_truck mismatch for ({i},{j}): got {t_t}, expected {expected_tt}"


def test_truck_only_route_feasibility_and_column(small_instance: Instance, distances: DistanceMatrix):
    """
    Build a simple truck-only route visiting all customers in index order and assert feasibility,
    cost finite and coverage includes all customers exactly once.
    """
    n = small_instance.n
    truck_seq = [0] + list(range(1, n + 1)) + [n + 1]
    drone_sorties: List[Tuple[int, List[int], int]] = []
    route = Route(truck_seq=truck_seq, drone_sorties=drone_sorties, instance=small_instance, distances=distances, config=None)
    feasible, reason = route.is_feasible()
    assert feasible, f"Truck-only route unexpectedly infeasible: {reason}"
    col = route.to_column()
    assert isinstance(col, Column)
    # coverage
    covered = route.covers()
    expected_customers = set(range(1, n + 1))
    assert covered == expected_customers, f"Route covers {covered} expected {expected_customers}"
    # cost finite
    c = route.cost()
    assert math.isfinite(c) and c > 0.0, f"Route cost invalid: {c}"


def test_column_pool_basic_operations(small_instance: Instance, distances: DistanceMatrix, column_pool: ColumnPool):
    """
    Test adding columns, retrieving by arc, and removing by predicate.
    Create two simple partitions of customers into two truck-only routes.
    """
    n = small_instance.n
    # split customers roughly in two groups
    mid = max(1, n // 2)
    seq1 = [0] + list(range(1, mid + 1)) + [n + 1]
    seq2 = [0] + list(range(mid + 1, n + 1)) + [n + 1]
    r1 = Route(truck_seq=seq1, drone_sorties=[], instance=small_instance, distances=distances, config=None)
    r2 = Route(truck_seq=seq2, drone_sorties=[], instance=small_instance, distances=distances, config=None)
    assert r1.is_feasible()[0] and r2.is_feasible()[0]
    c1 = r1.to_column()
    c2 = r2.to_column()
    column_pool.add(c1)
    column_pool.add(c2)
    all_cols = column_pool.get_all()
    assert len(all_cols) >= 2
    # find an arc from first route and query
    arcs1 = c1.truck_arcs
    assert len(arcs1) >= 1
    arc0 = arcs1[0]
    cols_using_arc = column_pool.get_columns_using_arc(arc0[0], arc0[1])
    assert any(getattr(col, "route_id", None) == getattr(c1, "route_id", None) for col in cols_using_arc)
    # remove columns that serve customer 1 via predicate
    def pred(col):
        try:
            return 1 in col.route.covers()
        except Exception:
            return False
    column_pool.remove_by_predicate(pred)
    remaining = column_pool.get_all()
    # ensure columns with customer 1 are removed/inactive
    for col in remaining:
        assert 1 not in col.route.covers()


@pytest.mark.skipif(not hasattr(RLMP_Solver, "solve_lp"), reason="RLMP_Solver with docplex required for SR test")
def test_rlmp_seeding_and_sr_separator(config_dict):
    """
    Seed RLMP with single-customer truck routes, solve LP, and verify SRSeparator.separate finds
    violated |S|=3 cuts by enumeration.
    """
    # create a tiny instance with n=6 to enumerate triplets easily
    gen = InstanceGenerator(config=config_dict)
    inst = gen.generate_one(type_index=1, n=6, theta=0.5, replicate=1)
    inst.validate(raise_on_error=True, auto_fix=False)
    dm = DistanceMatrix(inst)
    dm.compute_all()

    cp = ColumnPool(config=config_dict)
    rlmp = RLMP_Solver(instance=inst, column_pool=cp, config=config_dict)
    # seed initial columns (single-customer truck routes)
    rlmp.seed_initial_columns()
    # solve LP to get primal lambdas
    lp_obj, primal_lambda, duals = rlmp.solve_lp(time_limit_seconds=10.0)
    # primal_lambda is mapping from column_id -> lambda
    # run SRSeparator
    sr_sep = SRSeparator(rlmp)
    cuts = sr_sep.separate(primal_lambda)

    # manual enumeration: compute LHS for each triplet S of size 3
    # build a list of columns with their coverage and primal lambda
    cols = cp.get_all()
    # map column id to lambda value (use string keys)
    lambda_map = {str(k): float(v) for k, v in (primal_lambda or {}).items()}
    # helper to get coverage set for a column
    def coverage_set(col: Column) -> Set[int]:
        try:
            return set(col.route.covers())
        except Exception:
            # fallback a_ir
            if hasattr(col, "a_ir") and isinstance(col.a_ir, dict):
                return set(int(i) for i, v in col.a_ir.items() if int(v) != 0)
            return set()
    # compute per-S LHS
    n = inst.n
    violated_manual = []
    # precompute per-column lambda val
    col_entries = []
    for col in cols:
        cid = getattr(col, "route_id", str(id(col)))
        lam = float(lambda_map.get(str(cid), 0.0))
        cov = coverage_set(col)
        col_entries.append((cid, lam, cov))
    # enumerate triplets
    for S in itertools.combinations(range(1, n + 1), 3):
        lhs = 0.0
        for (_cid, lam, cov) in col_entries:
            cnt = sum(1 for i in S if i in cov)
            lhs += 0.5 * float(cnt) * float(lam)
        if lhs > 1.0 + float(config_dict["column_generation"]["sr_violation_eps"]):
            violated_manual.append((tuple(sorted(S)), lhs))
    # The separator should return at least the manually violated ones (order might differ)
    manual_set = set(tuple(sorted(x[0])) for x in violated_manual)
    returned_set = set(tuple(sorted(x[0])) for x in cuts)
    # Assert returned_set is subset of manual_set (no false positives)
    for s in returned_set:
        assert s in manual_set, f"SRSeparator returned non-violated cut {s}"
    # Also ensure all manual violations are returned unless sr_max_add_per_iteration limited
    max_add = config_dict.get("column_generation", {}).get("sr_max_add_per_iteration", None)
    if max_add is None:
        # expect equality
        assert returned_set == manual_set, f"SRSeparator missed violations: manual={manual_set} returned={returned_set}"
    else:
        # returned_set should be subset and size <= max_add
        assert returned_set.issubset(manual_set)
        assert len(returned_set) <= int(max_add)


# End of file
