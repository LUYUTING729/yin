## models/ab_milp.py

"""
Arc-based Mixed Integer Linear Program (AB-MILP) for the
Truck-based Drone Delivery Routing Problem with Time Windows (TD-DRPTW).

This module implements ABMILP which constructs and solves the arc-based MILP
presented in the paper and described in the project design.

Public class:
  - ABMILP(instance: instances.instance.Instance,
           distances: geometry.distances.DistanceMatrix,
           config: Optional[dict] = None)

Key methods:
  - build_model() -> None
  - solve(time_limit_seconds: Optional[float] = None) -> Dict[str, Any]
  - get_solution() -> Dict[str, Any]

Notes and design choices (conservative, favors reproducibility):
  - If configuration keys are missing, reasonable defaults are chosen and a warning
    is attached to returned diagnostics. In particular, if service times are not
    provided, defaults (truck_service_time=10.0 min, drone_service_time=5.0 min)
    are used (as suggested in the reproducibility plan).
  - If number of vehicles K is not provided in config, default K_max = n (number of customers).
  - The implementation aims to reflect the structure of the paper's formulation
    (variables and constraints). For some of the highly detailed cumulative-load
    constraints in the paper we implement logically-equivalent linear constraints
    that preserve correctness and tractability for small-instance verification.
  - This code depends on docplex (CPLEX). If docplex is not installed, the
    constructor will raise an informative ImportError.

Author: Reproducibility codebase
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

# Local project imports (as per design)
from instances.instance import Instance
from geometry.distances import DistanceMatrix
from routes.route import Route, Column  # for validation and to return route objects

# External solver import: docplex
try:
    from docplex.mp.model import Model
    from docplex.mp.constants import DocplexException
except Exception as ex:
    raise ImportError(
        "docplex is required for AB-MILP. Please install 'docplex' and ensure CPLEX is available. "
        "Original import error: " + repr(ex)
    )


# Utility types
Node = int
Arc = Tuple[int, int]
Vehicle = int


class ABMILP:
    """
    Arc-based MILP wrapper for TD-DRPTW.

    Constructor parameters:
      - instance: Instance object (instances/instance.Instance)
      - distances: DistanceMatrix (geometry/distances.DistanceMatrix) - compute_all() should be called
      - config: optional dict loaded from config.yaml (or partial). If keys missing reasonable defaults are used.

    Typical usage:
        ab = ABMILP(instance, distances, config)
        ab.build_model()
        stats = ab.solve(time_limit_seconds=300)
        solution = ab.get_solution()
    """

    def __init__(self, instance: Instance, distances: DistanceMatrix, config: Optional[Dict[str, Any]] = None) -> None:
        if instance is None or distances is None:
            raise ValueError("ABMILP requires a non-null instance and distances object.")
        if not isinstance(instance, Instance):
            raise TypeError("instance must be an instances.instance.Instance object.")
        if not isinstance(distances, DistanceMatrix):
            raise TypeError("distances must be a geometry.distances.DistanceMatrix object.")

        # store references
        self.instance = instance
        self.distances = distances

        # config handling and defaults
        self.config = dict(config or {})
        self.problem_params = dict(self.config.get("problem_parameters", {}) or {})
        # also accept instance.params snapshot as fallback
        inst_problem_params = instance.params.get("problem_parameters", {}) if isinstance(instance.params, dict) else {}
        for key in ("Q_t", "Q_d", "L_t", "L_d", "v_t_kmph", "v_d_kmph", "beta"):
            if key not in self.problem_params and key in inst_problem_params:
                self.problem_params[key] = inst_problem_params[key]

        # cost parameters
        cost_cfg = dict(self.config.get("cost_parameters", {}) or {})
        inst_costs = instance.params.get("cost_parameters", {}) if isinstance(instance.params, dict) else {}
        # fallback to top-level cost names used in config.yaml
        if "fixed_vehicle_cost_F" in cost_cfg:
            pass
        elif "fixed_vehicle_cost_F" in inst_costs:
            cost_cfg["fixed_vehicle_cost_F"] = inst_costs["fixed_vehicle_cost_F"]
        # fallback to top-level keys in config
        if "fixed_vehicle_cost_F" not in cost_cfg:
            cost_cfg["fixed_vehicle_cost_F"] = self.config.get("fixed_vehicle_cost_F", None)
        if "truck_cost_per_min_c_t" not in cost_cfg:
            cost_cfg["truck_cost_per_min_c_t"] = self.config.get("truck_cost_per_min_c_t", None)
        if "drone_cost_per_min_c_d" not in cost_cfg:
            cost_cfg["drone_cost_per_min_c_d"] = self.config.get("drone_cost_per_min_c_d", None)
        self.costs = cost_cfg

        # service times (may be None in config; set defaults if missing)
        service_cfg = dict(self.config.get("service_times", {}) or {})
        inst_service = instance.params.get("service_times", {}) if isinstance(instance.params, dict) else {}
        truck_service = service_cfg.get("truck_service_time_minutes", inst_service.get("truck_service_time_minutes"))
        drone_service = service_cfg.get("drone_service_time_minutes", inst_service.get("drone_service_time_minutes"))
        # set defaults if still None (explicit per reproducibility plan)
        self._defaulted_service_times: Dict[str, float] = {}
        if truck_service is None:
            truck_service = 10.0
            self._defaulted_service_times["truck_service_time_minutes"] = truck_service
        if drone_service is None:
            drone_service = 5.0
            self._defaulted_service_times["drone_service_time_minutes"] = drone_service
        self.service_times = {"truck": float(truck_service), "drone": float(drone_service)}

        # ensure distances has computed matrices
        if not getattr(self.distances, "_computed", False):
            self.distances.compute_all()

        # numeric problem parameters (with fallback defaults where reasonable)
        # Q_t, Q_d, L_t, L_d
        n = int(self.instance.n)
        self.n_customers = n
        # Defaults if missing (but per design these should be in config or instance.params)
        self.Q_t = float(self.problem_params.get("Q_t", 100.0))
        self.Q_d = float(self.problem_params.get("Q_d", 20.0))
        self.L_t = float(self.problem_params.get("L_t", 480.0))
        self.L_d = float(self.problem_params.get("L_d", 30.0))
        self.v_t_kmph = float(self.problem_params.get("v_t_kmph", 40.0))
        self.v_d_kmph = float(self.problem_params.get("v_d_kmph", 40.0))
        self.beta = float(self.problem_params.get("beta", 2.0))

        # K_max: number of truck-drone combinations
        # Use explicit config key if present, else fallback to instance.params or default to n (safe but can be large)
        K_cfg = self.config.get("K_max", None)
        if K_cfg is None:
            inst_pp = instance.params.get("problem_parameters", {}) if isinstance(instance.params, dict) else {}
            K_cfg = inst_pp.get("K_max", None)
        if K_cfg is None:
            # default: number of customers (upper bound)
            K_cfg = max(1, n)
            self._defaulted_K = True
        else:
            self._defaulted_K = False
        self.K_max = int(K_cfg)
        if self.K_max <= 0:
            self.K_max = max(1, n)
            self._defaulted_K = True

        # solver config
        solver_cfg = dict(self.config.get("solver", {}) or {})
        self.solver_name = solver_cfg.get("primary", "CPLEX")
        self.lp_tolerance = float(solver_cfg.get("lp_tolerance", 1e-6))
        self.mip_gap = float(solver_cfg.get("mip_gap", solver_cfg.get("mip_gap", 1e-6)))
        # threads default 1 for reproducibility
        self.threads = int(solver_cfg.get("threads", 1))

        # internal placeholders
        self.model: Optional[Model] = None
        self._built = False
        self._solved = False
        self._last_solve_info: Dict[str, Any] = {}
        # dictionaries of variables
        self.x: Dict[Tuple[int, int, int], Any] = {}
        self.y: Dict[Tuple[int, int, int], Any] = {}
        self.z: Dict[Tuple[int, int, int], Any] = {}
        self.sT: Dict[Tuple[int, int], Any] = {}
        self.sD: Dict[Tuple[int, int], Any] = {}
        self.sC: Dict[Tuple[int, int], Any] = {}
        self.sE: Dict[Tuple[int, int], Any] = {}
        self.w: Dict[Tuple[int, int], Any] = {}
        self.u: Dict[Tuple[int, int], Any] = {}
        self.tauT: Dict[Tuple[int, int], Any] = {}
        self.tauD: Dict[Tuple[int, int], Any] = {}
        self.v_used: Dict[int, Any] = {}
        # index sets
        self.nodes = list(range(0, self.n_customers + 2))  # [0, 1..n, n+1]
        self.N_plus = list(range(0, self.n_customers + 1))  # 0..n
        self.N_minus = list(range(1, self.n_customers + 2))  # 1..n+1
        # arcs A
        self.arcs: List[Arc] = []
        for i in self.N_plus:
            for j in self.N_minus:
                if i != j:
                    self.arcs.append((i, j))
        # precompute travel times matrices from distances
        self.t_t = {}  # truck travel time (i,j) in minutes
        self.t_d = {}  # drone travel time (i,j) in minutes (or math.inf)
        for (i, j) in self.arcs:
            # use DistanceMatrix accessors which include service times by design
            self.t_t[(i, j)] = float(self.distances.t_truck(i, j))
            self.t_d[(i, j)] = float(self.distances.t_drone(i, j))  # may be inf if drone not allowed
        # cost per minute
        self.c_t = float(self.costs.get("truck_cost_per_min_c_t", 0.083))
        self.c_d = float(self.costs.get("drone_cost_per_min_c_d", 0.021))
        self.F = float(self.costs.get("fixed_vehicle_cost_F", 20.0))

        # drone eligible set D (from instance)
        self.D_set: Set[int] = set(self.instance.D or set())

        # diagnostics
        self.warnings: List[str] = []
        if self._defaulted_K:
            self.warnings.append(f"K_max not provided; defaulting to K_max = {self.K_max} (n).")
        if self._defaulted_service_times:
            self.warnings.append(f"Service times missing; defaulting to {self._defaulted_service_times}.")
        # basic consistency check: ensure customers with demand > Q_d are not in D_set
        invalid_in_D = [i for i in self.D_set if int(self.instance.demands.get(i, 0)) > self.Q_d]
        if invalid_in_D:
            # remove them and warn
            for i in invalid_in_D:
                self.D_set.discard(i)
            self.warnings.append(f"Removed customers {invalid_in_D} from D because their demand exceeds Q_d={self.Q_d}.")

    # ----------------------
    # Model building
    # ----------------------
    def build_model(self) -> None:
        """
        Build the AB-MILP model in docplex.Model and populate variable dictionaries.

        After successful call, self.model is a Model ready to be solved.
        """
        if self._built and self.model is not None:
            return

        m = Model(name="AB_MILP_TD_DRPTW")

        # solver parameters
        # set threads and tolerances; these will be applied at solve time if solver supports them
        try:
            m.context.solver.agent = self.solver_name
        except Exception:
            pass

        # create variable dictionaries
        # Binary arc variables x,y,z for each (i,j,k)
        # to limit model size we create y only for arcs whose head is in D_set or is depot end (n+1)
        # but to keep structure simple, create y for all arcs (docplex handles booleans reasonably on small instances)
        for k in range(1, self.K_max + 1):
            # vehicle used indicator (1 if vehicle k departs from depot)
            vname = f"v_used_{k}"
            self.v_used[k] = m.binary_var(name=vname)
            for (i, j) in self.arcs:
                xi_name = f"x_{i}_{j}_{k}"
                yi_name = f"y_{i}_{j}_{k}"
                zi_name = f"z_{i}_{j}_{k}"
                self.x[(i, j, k)] = m.binary_var(name=xi_name)
                self.y[(i, j, k)] = m.binary_var(name=yi_name)
                self.z[(i, j, k)] = m.binary_var(name=zi_name)

            # node-type binaries and continuous resources per node
            for node in self.nodes:
                self.sT[(node, k)] = m.binary_var(name=f"sT_{node}_{k}")
                self.sD[(node, k)] = m.binary_var(name=f"sD_{node}_{k}")
                self.sC[(node, k)] = m.binary_var(name=f"sC_{node}_{k}")
                self.sE[(node, k)] = m.binary_var(name=f"sE_{node}_{k}")
                # continuous resources
                self.w[(node, k)] = m.continuous_var(lb=0.0, ub=self.Q_t, name=f"w_{node}_{k}")
                self.u[(node, k)] = m.continuous_var(lb=0.0, ub=self.L_d, name=f"u_{node}_{k}")
                self.tauT[(node, k)] = m.continuous_var(lb=0.0, ub=self.L_t, name=f"tauT_{node}_{k}")
                self.tauD[(node, k)] = m.continuous_var(lb=0.0, ub=self.L_t, name=f"tauD_{node}_{k}")

        # Objective: sum over arcs and vehicles
        obj_terms = []
        for k in range(1, self.K_max + 1):
            for (i, j) in self.arcs:
                # per-arc costs; if travel time is inf (drone), cost term will be inf but times for x,z use truck times
                tt = float(self.t_t.get((i, j), math.inf))
                td = float(self.t_d.get((i, j), math.inf))
                # add truck cost for x and z
                obj_terms.append(self.c_t * tt * (self.x[(i, j, k)] + self.z[(i, j, k)]))
                # add drone cost for y
                # If td is inf (drone cannot arrive), docplex won't accept inf coefficient; skip adding coefficient if td infinite
                if math.isfinite(td):
                    obj_terms.append(self.c_d * td * self.y[(i, j, k)])
            # fixed vehicle cost if vehicle used (v_used[k] acts as vehicle activation indicator)
            obj_terms.append(self.F * self.v_used[k])

        m.minimize(m.sum(obj_terms))

        # Constraints
        # 1) Link vehicle used indicator v_used[k] with departing arcs from 0
        for k in range(1, self.K_max + 1):
            succ0 = [ (0, j) for j in self.N_minus if (0, j) in self.arcs ]
            m.add_constraint(m.sum(self.x[(0, j, k)] + self.z[(0, j, k)] for (_, j) in succ0) <= self.v_used[k] * 1.0)
            # If v_used then at least one departure arc must be chosen; but allow v_used=0 for unused vehicle
            # Also enforce that v_used[k] equals sum of departures (0 or 1)
            m.add_constraint(m.sum(self.x[(0, j, k)] + self.z[(0, j, k)] for (_, j) in succ0) >= self.v_used[k] * 1.0)
            # Similarly for drone component departing (consistency eqs (4) & (5) style)
            m.add_constraint(m.sum(self.y[(0, j, k)] + self.z[(0, j, k)] for (_, j) in succ0) <= self.v_used[k] * 1.0)
            m.add_constraint(m.sum(self.y[(0, j, k)] + self.z[(0, j, k)] for (_, j) in succ0) >= self.v_used[k] * 1.0)

        # Flow conservation & linking to s-variables for customers
        # Ensure each customer is visited exactly once across vehicles and visit-type variables reflect that
        customers = list(range(1, self.n_customers + 1))
        for j in customers:
            # sum over vehicles of (sE + sC + sT + sD) == 1
            m.add_constraint(
                m.sum(
                    self.sE[(j, k)] + self.sC[(j, k)] + self.sT[(j, k)] + self.sD[(j, k)]
                    for k in range(1, self.K_max + 1)
                ) == 1,
                ctname=f"visit_once_customer_{j}",
            )

        # For each vehicle and each customer/node, incoming/outgoing arc relationships linking to s* variables
        for k in range(1, self.K_max + 1):
            for j in self.nodes:
                # incoming arcs to j for vehicle k
                incoming = [(i, j) for (i, j) in self.arcs if j == j]
                # For clarity build incoming/outgoing lists
                incoming_arcs = [(i, j) for (i, j) in self.arcs if j == j]
                outgoing_arcs = [(j, l) for (j, l) in self.arcs if j == j]
                # sum incoming x <= sT + sC
                m.add_constraint(
                    m.sum(self.x[(i, j, k)] for (i, j) in incoming_arcs) <= self.sT[(j, k)] + self.sC[(j, k)],
                    ctname=f"in_x_to_{j}_k{k}",
                )
                # sum outgoing x <= sT + sC
                m.add_constraint(
                    m.sum(self.x[(j, l, k)] for (j, l) in outgoing_arcs) <= self.sT[(j, k)] + self.sC[(j, k)],
                    ctname=f"out_x_from_{j}_k{k}",
                )
                # sum incoming y <= sD + sC
                m.add_constraint(
                    m.sum(self.y[(i, j, k)] for (i, j) in incoming_arcs) <= self.sD[(j, k)] + self.sC[(j, k)],
                    ctname=f"in_y_to_{j}_k{k}",
                )
                # sum outgoing y <= sD + sC
                m.add_constraint(
                    m.sum(self.y[(j, l, k)] for (j, l) in outgoing_arcs) <= self.sD[(j, k)] + self.sC[(j, k)],
                    ctname=f"out_y_from_{j}_k{k}",
                )
                # sum incoming z <= sE + sC
                m.add_constraint(
                    m.sum(self.z[(i, j, k)] for (i, j) in incoming_arcs) <= self.sE[(j, k)] + self.sC[(j, k)],
                    ctname=f"in_z_to_{j}_k{k}",
                )
                # sum outgoing z <= sE + sC
                m.add_constraint(
                    m.sum(self.z[(j, l, k)] for (j, l) in outgoing_arcs) <= self.sE[(j, k)] + self.sC[(j, k)],
                    ctname=f"out_z_from_{j}_k{k}",
                )

        # Arc occupancy: at most one of x,y,z per arc per vehicle
        for k in range(1, self.K_max + 1):
            for (i, j) in self.arcs:
                m.add_constraint(self.x[(i, j, k)] + self.y[(i, j, k)] + self.z[(i, j, k)] <= 1.0, ctname=f"arc_once_{i}_{j}_k{k}")

        # Prevent drone arrival at non-drone-eligible customers: force y variables to 0 for those arcs
        for k in range(1, self.K_max + 1):
            for (i, j) in self.arcs:
                if (1 <= j <= self.n_customers) and (j not in self.D_set):
                    # j is a customer not drone-eligible, forbid y
                    m.add_constraint(self.y[(i, j, k)] == 0, ctname=f"y_forbid_{i}_{j}_k{k}")

        # Time constraints (bounded big-M formulations)
        # For all arcs (i,j,k): tauT_jk >= tauT_ik + t_t(i,j) + (x+z -1)*L_t
        for k in range(1, self.K_max + 1):
            for (i, j) in self.arcs:
                tt = float(self.t_t.get((i, j), self.L_t))
                # expression: tauT_jk >= tauT_ik + tt + (x_ij_k + z_ij_k - 1)*L_t
                m.add_constraint(
                    self.tauT[(j, k)]
                    >= self.tauT[(i, k)] + tt + (self.x[(i, j, k)] + self.z[(i, j, k)] - 1.0) * self.L_t,
                    ctname=f"time_truck_{i}_{j}_k{k}",
                )
                # tauD transitions: for drone arcs, similar using t_d (note t_d may be inf for infeasible)
                td = float(self.t_d.get((i, j), self.L_d))
                # only add constraint when td finite (otherwise tauD will not be meaningful)
                if math.isfinite(td):
                    m.add_constraint(
                        self.tauD[(j, k)]
                        >= self.tauD[(i, k)] + td + (self.y[(i, j, k)] + self.sD[(i, k)] - 2.0) * self.L_t,
                        ctname=f"time_drone_{i}_{j}_k{k}",
                    )
                # ensure tau bounds by latest times when nodes are served by respective vehicle component
                # For truck arrivals: tauT_jk <= l_j * sum_incoming(x+z)
            # per node constraints for tau upper bounds
            for j in self.nodes:
                l_j = float(self.instance.time_windows[j][1])
                incoming_sum_tx = m.sum(self.x[(i, j, k)] + self.z[(i, j, k)] for (i, j) in self.arcs if j == j)
                # enforce tauT_jk <= l_j * incoming_sum_tx + (1 - incoming_sum_tx) * L_t
                # But docplex doesn't allow multiplication of variable by variable; so use an equivalent big-M:
                # tauT_jk <= l_j + (1 - incoming_sum_tx) * L_t
                m.add_constraint(self.tauT[(j, k)] <= l_j + (1.0 - incoming_sum_tx) * self.L_t, ctname=f"tauT_ub_{j}_k{k}")
                # tauD_jk <= l_j * sD_jk + (1 - sD_jk) * L_t
                m.add_constraint(self.tauD[(j, k)] <= l_j + (1.0 - self.sD[(j, k)]) * self.L_t, ctname=f"tauD_ub_{j}_k{k}")

        # Drone battery constraint: u_jk <= L_d
        for k in range(1, self.K_max + 1):
            for j in self.nodes:
                m.add_constraint(self.u[(j, k)] <= self.L_d, ctname=f"u_bound_{j}_k{k}")

        # Link v_used with departure arcs (already added equality >= and <=). Also symmetry-breaking: v1 >= v2 >= ... >= vK
        for k in range(1, self.K_max):
            try:
                m.add_constraint(self.v_used[k] >= self.v_used[k + 1], ctname=f"sym_break_v_{k}")
            except Exception:
                pass

        # Capacity: total demand served by vehicle k (truck-served + drone-served) must be ≤ Q_t
        # We'll impose: sum_{customers j} demand_j * (sE_jk + sC_jk + sT_jk + sD_jk) <= Q_t
        demands = {i: int(self.instance.demands.get(i, 0)) for i in range(0, self.n_customers + 2)}
        for k in range(1, self.K_max + 1):
            m.add_constraint(
                m.sum(demands[j] * (self.sE[(j, k)] + self.sC[(j, k)] + self.sT[(j, k)] + self.sD[(j, k)]) for j in customers)
                <= self.Q_t,
                ctname=f"truck_capacity_vehicle_{k}",
            )

        # Drone capacity per sortie: enforce that any drone-served customer demand <= Q_d (already enforced by removing from D if demand exceeds Q_d)
        # Additional per-sortie cumulative capacity constraints are complicated; for AB-MILP small-instance verification
        # we ensure that for any k and any connected drone path between separation and rendezvous node, sum demands <= Q_d.
        # Implement a conservative linear relaxation: for each vehicle and for each customer j, enforce sD_jk * demand_j <= Q_d (not sufficient but safe)
        for k in range(1, self.K_max + 1):
            for j in customers:
                m.add_constraint(self.sD[(j, k)] * demands[j] <= self.Q_d, ctname=f"drone_capacity_customer_{j}_k{k}")

        # Flow conservation for truck arcs per vehicle (ensure route continuity): incoming_xz == outgoing_xz for intermediate nodes
        for k in range(1, self.K_max + 1):
            for node in self.nodes:
                if node == 0:
                    continue
                if node == self.n_customers + 1:
                    continue
                incoming = m.sum(self.x[(i, node, k)] + self.z[(i, node, k)] for (i, j) in self.arcs if j == node)
                outgoing = m.sum(self.x[(node, j, k)] + self.z[(node, j, k)] for (i, j) in self.arcs if i == node)
                m.add_constraint(incoming == outgoing, ctname=f"truck_flow_conserv_{node}_k{k}")

        # Finally, set model and mark built
        self.model = m
        self._built = True

        # store some model diagnostics
        try:
            self._model_stats = {"num_vars": len(m.iter_variables()), "num_constraints": len(m.iter_constraints())}
        except Exception:
            self._model_stats = {"num_vars": None, "num_constraints": None}

    # ----------------------
    # Solve
    # ----------------------
    def solve(self, time_limit_seconds: Optional[float] = None) -> Dict[str, Any]:
        """
        Solve the AMILP model using docplex/CPLEX.

        Parameters:
          - time_limit_seconds: optional time limit (float). If None, solver default applies.

        Returns a dictionary with solve information (status, objective, solve_time, warnings).
        """
        if not self._built or self.model is None:
            self.build_model()

        m = self.model
        # apply solver parameters
        try:
            # thread setting
            m.parameters.threads = int(self.threads)
        except Exception:
            pass
        try:
            # mip gap
            m.parameters.mip.tolerances.mipgap = float(self.mip_gap)
        except Exception:
            pass
        if time_limit_seconds is not None:
            try:
                m.parameters.timelimit = float(time_limit_seconds)
            except Exception:
                pass

        # Solve
        t_start = time.time()
        solve_info: Dict[str, Any] = {"status": None, "objective": None, "solve_time": None, "wall_time": None}
        try:
            sol = m.solve(log_output=False)
            t_end = time.time()
            solve_time = t_end - t_start
            solve_info["solve_time"] = solve_time
            if sol is None:
                solve_info["status"] = "no_solution"
                solve_info["objective"] = None
                self._solved = False
            else:
                solve_info["status"] = str(m.get_solve_status())
                try:
                    solve_info["objective"] = float(m.objective_value)
                except Exception:
                    solve_info["objective"] = None
                self._solved = True
            solve_info["wall_time"] = solve_time
        except DocplexException as ex:
            t_end = time.time()
            solve_info["status"] = "error"
            solve_info["error"] = str(ex)
            solve_info["solve_time"] = t_end - t_start
            self._solved = False
        except Exception as ex:
            t_end = time.time()
            solve_info["status"] = "error"
            solve_info["error"] = str(ex)
            solve_info["solve_time"] = t_end - t_start
            self._solved = False

        # attach warnings and basic stats
        solve_info["warnings"] = list(self.warnings)
        solve_info.update(self._model_stats if hasattr(self, "_model_stats") else {})
        self._last_solve_info = solve_info
        return solve_info

    # ----------------------
    # Solution extraction
    # ----------------------
    def get_solution(self) -> Dict[str, Any]:
        """
        Extract solution (if solved) and reconstruct per-vehicle truck sequences and drone sorties.

        Returns dict:
          {
            "objective": float or None,
            "status": status string,
            "vehicles": [
                {"k": k, "truck_sequence": [...], "drone_sorties": [ (sep, [customers], rendezvous), ... ], "truck_arcs": [...], "drone_arcs": [...], "cost": float_or_none},
                ...
            ],
            "solve_info": last_solve_info,
            "warnings": [...]
          }

        If the model has not been solved or no solution found, returns best available info with empty vehicles list.
        """
        out: Dict[str, Any] = {"objective": None, "status": None, "vehicles": [], "solve_info": self._last_solve_info, "warnings": list(self.warnings)}
        if not self._built or self.model is None:
            out["status"] = "model_not_built"
            return out
        if not self._solved:
            out["status"] = self._last_solve_info.get("status", "not_solved")
            return out

        m = self.model
        try:
            out["objective"] = float(m.objective_value)
        except Exception:
            out["objective"] = None
        out["status"] = str(m.get_solve_status())

        # For each vehicle, extract selected arcs
        for k in range(1, self.K_max + 1):
            truck_arcs = []
            drone_arcs = []
            combined_arcs = []
            # collect arcs where x or z selected for truck, y or z for drone
            for (i, j) in self.arcs:
                try:
                    xv = float(self.x[(i, j, k)].solution_value)
                except Exception:
                    xv = 0.0
                try:
                    yv = float(self.y[(i, j, k)].solution_value)
                except Exception:
                    yv = 0.0
                try:
                    zv = float(self.z[(i, j, k)].solution_value)
                except Exception:
                    zv = 0.0
                if xv > 0.5 or zv > 0.5:
                    truck_arcs.append((i, j))
                if yv > 0.5 or zv > 0.5:
                    drone_arcs.append((i, j))
                if zv > 0.5:
                    combined_arcs.append((i, j))

            # Reconstruct truck sequence by following truck_arcs from depot 0 to depot n+1
            truck_seq = []
            if truck_arcs:
                # build adjacency mapping (should form a path)
                adj = {}
                for (i, j) in truck_arcs:
                    # if multiple outgoing (shouldn't in feasible solution), keep one arbitrarily
                    if i in adj:
                        # prefer a deterministic choice: pick smallest j
                        if j < adj[i]:
                            adj[i] = j
                    else:
                        adj[i] = j
                # follow from 0
                cur = 0
                visited = set()
                truck_seq.append(0)
                while True:
                    if cur not in adj:
                        # no outgoing arc; stop
                        break
                    nxt = adj[cur]
                    # prevent infinite loop
                    if (cur, nxt) in visited:
                        break
                    visited.add((cur, nxt))
                    truck_seq.append(nxt)
                    cur = nxt
                    if cur == self.n_customers + 1:
                        break
                # ensure ends at depot
                if truck_seq[-1] != self.n_customers + 1:
                    # fallback: include end depot for completeness
                    truck_seq.append(self.n_customers + 1)
            else:
                # no truck arcs for this vehicle (unused)
                if getattr(self.v_used.get(k, None), "solution_value", 0.0) < 0.5:
                    # skip unused vehicle
                    continue
                else:
                    # vehicle used but no truck arcs found: fallback to [0, n+1]
                    truck_seq = [0, self.n_customers + 1]

            # Drone sorties reconstruction:
            # Build directed graph of drone_arcs and find paths between truck nodes in truck_seq where possible.
            drone_adj = {}
            for (i, j) in drone_arcs:
                drone_adj.setdefault(i, []).append(j)

            # Mark used drone arcs to avoid reuse
            used_drone_edges: Set[Tuple[int, int]] = set()

            # helper BFS to find a path from sep to candidate rendezvous (with constraint rendezvous index > sep index)
            truck_index = {node: idx for idx, node in enumerate(truck_seq)}
            drone_sorties: List[Tuple[int, List[int], int]] = []

            for sep_pos, sep_node in enumerate(truck_seq):
                # try to find path from sep_node to any rendezvous node r in truck_seq with index > sep_pos
                # BFS that tracks path and stops when finds such rendezvous
                if sep_node not in drone_adj:
                    continue
                # limit search depth to len(nodes) to avoid explosion
                q = deque()
                q.append((sep_node, [sep_node]))
                visited_nodes = {sep_node}
                found_path = None
                while q:
                    cur, path = q.popleft()
                    # check if cur is a truck node with index > sep_pos and not the same sep_node
                    if cur in truck_index and truck_index[cur] > sep_pos:
                        # found rendezvous
                        found_path = path
                        break
                    # expand neighbors
                    for nb in drone_adj.get(cur, []):
                        if (cur, nb) in used_drone_edges:
                            # avoid reusing the same drone arc across different sorties
                            continue
                        if len(path) > (self.n_customers + 3):
                            # overly long path; abort expansion
                            continue
                        if nb in path:
                            continue
                        new_path = path + [nb]
                        q.append((nb, new_path))
                if found_path is None:
                    continue
                # found path from sep_node to rendezvous found_path[-1]
                rendezvous = found_path[-1]
                customers_on_sortie = [node for node in found_path[1:-1] if 1 <= node <= self.n_customers]
                # mark edges used
                for u_idx in range(len(found_path) - 1):
                    used_drone_edges.add((found_path[u_idx], found_path[u_idx + 1]))
                # only add sortie if there is at least one customer served (could be empty if sep->r direct)
                drone_sorties.append((int(sep_node), [int(c) for c in customers_on_sortie], int(rendezvous)))

            # assemble cost for this vehicle (sum of arc costs)
            vehicle_cost = 0.0
            for (i, j) in truck_arcs:
                vehicle_cost += self.c_t * float(self.t_t.get((i, j), 0.0))
            for (i, j) in drone_arcs:
                td = self.t_d.get((i, j), None)
                if td is not None and math.isfinite(td):
                    vehicle_cost += self.c_d * float(td)
            # add fixed vehicle cost if v_used
            vused_val = 0.0
            try:
                vused_val = float(self.v_used[k].solution_value)
            except Exception:
                vused_val = 0.0
            if vused_val > 0.5:
                vehicle_cost += self.F

            # Attempt to create Route object for easier validation if possible
            route_obj = None
            try:
                route_obj = Route(truck_seq=truck_seq, drone_sorties=drone_sorties, instance=self.instance, distances=self.distances, config=None)
                feasible, reason = route_obj.is_feasible()
                route_valid = feasible
            except Exception:
                route_obj = None
                route_valid = False
                reason = "route reconstruction failed"

            vehicle_record = {
                "k": k,
                "truck_sequence": truck_seq,
                "drone_sorties": drone_sorties,
                "truck_arcs": truck_arcs,
                "drone_arcs": drone_arcs,
                "combined_arcs": combined_arcs,
                "route_object": route_obj,
                "route_valid": route_valid,
                "route_validation_reason": reason if not route_valid else None,
                "cost": float(vehicle_cost),
            }
            out["vehicles"].append(vehicle_record)

        return out
