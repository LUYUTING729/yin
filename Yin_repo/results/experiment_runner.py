## results/experiment_runner.py

"""
results/experiment_runner.py

ExperimentRunner orchestrates experiments described in the reproducibility plan.
It relies on the project's modules (Instance, DistanceMatrix, ColumnPool, RLMP_Solver,
PricingManager, SRSeparator, BranchAndBound) and the provided config.yaml.

Primary responsibilities implemented here:
 - Load and validate configuration (config.yaml).
 - Discover or accept instance list to run experiments on.
 - For each instance: create solver objects, run root column-and-cut generation,
   optionally run full BPC (branch-and-bound), collect and save per-instance results.
 - Aggregate per-experiment results into CSVs and optionally plot simple charts.

Notes:
 - This orchestrator performs conservative config validation and will abort if
   required numeric configuration values are null or missing (per plan).
 - It assumes the other modules in the project are present and implement the
   interfaces described in the design.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Project imports (these modules are part of the designed package)
try:
    from utils.logger import Logger
except Exception:
    Logger = None  # type: ignore

from utils.serialization import SerializationManager
from instances.instance import Instance
from instances.instance_generator import InstanceGenerator
from geometry.distances import DistanceMatrix
from columns.column_pool import ColumnPool
from master.rlmp_solver import RLMP_Solver
from pricing.pricing_manager import PricingManager
from cuts.sr_separator import SRSeparator
from bnb.branch_and_bound import BranchAndBound

# typing aliases
Arc = Tuple[int, int]

# Default results folder names (can be overridden by config)
_DEFAULT_RESULTS_DIR = Path("results")
_DEFAULT_LOGS_DIR = Path("logs")
_DEFAULT_INSTANCES_DIR = Path("instances")


@dataclass
class ExperimentResult:
    """Container for per-instance result data to be serialized."""
    instance_id: str
    seed: Optional[int]
    run_start: str
    run_end: Optional[str] = None
    total_time_s: Optional[float] = None
    LB: Optional[float] = None
    UB: Optional[float] = None
    gap_percent: Optional[float] = None
    solved: bool = False
    num_columns: int = 0
    num_sr_cuts: int = 0
    nodes_explored: Optional[int] = None
    pricing_counts: Dict[str, int] = field(default_factory=dict)
    cg_iterations: List[Dict[str, Any]] = field(default_factory=list)
    node_logs: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    """
    Orchestrates execution of experiment batches.

    Usage:
      runner = ExperimentRunner(config_path="config.yaml")
      runner.run_experiment_batch(experiment_key="enhancement_strategies_test")

    Public methods:
      - run_experiment_batch(experiment_key: str, instance_paths: Optional[List[Path]] = None)
      - run_instance(instance_path: Path, experiment_spec: dict, time_limit: Optional[float] = None)
      - aggregate_results(results: List[ExperimentResult], experiment_key: str) -> pd.DataFrame
      - plot_results(aggregated_df: pd.DataFrame, experiment_key: str)
    """

    def __init__(self, config_path: Union[str, Path] = "config.yaml"):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        with open(self.config_path, "rt", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

        # Setup output folders from config or defaults
        out = self.config.get("output", {}) or {}
        self.results_dir = Path(out.get("results_dir", _DEFAULT_RESULTS_DIR)).resolve()
        self.logs_dir = Path(out.get("logs_dir", _DEFAULT_LOGS_DIR)).resolve()
        self.instances_dir = Path(out.get("instances_dir", _DEFAULT_INSTANCES_DIR)).resolve()

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.instances_dir, exist_ok=True)

        # Setup logger
        run_id = f"experiment_runner_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
        try:
            self.logger = Logger(run_id=run_id, instance_id=None, config=self.config) if Logger is not None else None
        except Exception:
            self.logger = None

        # serialization manager
        self.serializer = SerializationManager(config=self.config, logger=self.logger)

        # Validate critical config
        self._validate_config()

    def _log(self, event: str, payload: Dict[str, Any]) -> None:
        """Helper for structured logging to configured logger or fallback print."""
        payload = dict(payload or {})
        payload["timestamp"] = datetime.utcnow().isoformat() + "Z"
        if self.logger is not None:
            try:
                self.logger.log(event, payload)
            except Exception:
                # fallback
                print(f"[{event}] {payload}")
        else:
            print(f"[{event}] {payload}")

    def _validate_config(self) -> None:
        """
        Validate that essential configuration values exist and are non-null.
        If any required keys are missing or null, raise ValueError listing them.
        """
        required_paths = [
            ("problem_parameters", "Q_t"),
            ("problem_parameters", "Q_d"),
            ("problem_parameters", "L_t"),
            ("problem_parameters", "L_d"),
            ("problem_parameters", "v_t_kmph"),
            ("problem_parameters", "v_d_kmph"),
            ("problem_parameters", "beta"),
            ("cost_parameters", "fixed_vehicle_cost_F"),
            ("cost_parameters", "truck_cost_per_min_c_t"),
            ("cost_parameters", "drone_cost_per_min_c_d"),
        ]

        missing = []
        for path in required_paths:
            top = self.config.get(path[0], {}) or {}
            if path[1] not in top or top.get(path[1]) is None:
                missing.append(".".join(path))

        # service times must be provided explicitly (per reproducibility plan)
        svc = self.config.get("service_times", {}) or {}
        if svc.get("truck_service_time_minutes") is None:
            missing.append("service_times.truck_service_time_minutes")
        if svc.get("drone_service_time_minutes") is None:
            missing.append("service_times.drone_service_time_minutes")

        # experiments block must exist
        if "experiments" not in self.config or not isinstance(self.config["experiments"], dict):
            missing.append("experiments")

        # heuristics hyperparameters (pricing) - ensure at least some exist (we require greedy params exist)
        pricing = self.config.get("pricing", {}) or {}
        hp = pricing.get("heuristic_params", {}) or {}
        # We require presence of greedy_random_restarts and greedy_random_top_k and tabu params per plan.
        # If absent, signal them: the plan mandates user confirm; but to be pragmatic we allow defaults.
        # However instruction stated: If null, must be set; so enforce they are present.
        heuristic_required = [
            ("pricing", "heuristic_params", "greedy_random_restarts"),
            ("pricing", "heuristic_params", "greedy_random_top_k"),
            ("pricing", "heuristic_params", "tabu_tenure"),
            ("pricing", "heuristic_params", "tabu_max_iterations"),
            ("pricing", "heuristic_params", "tabu_no_improve_limit"),
        ]
        for p in heuristic_required:
            level = self.config.get(p[0], {}) or {}
            sub = level.get(p[1], {}) or {}
            if p[2] not in sub or sub.get(p[2]) is None:
                missing.append(".".join(p))

        # solver availability check: CPLEX (docplex) expected
        solver_cfg = self.config.get("solver", {}) or {}
        primary_solver = solver_cfg.get("primary", "CPLEX")
        solver_ok = False
        solver_err_msg = None
        if primary_solver and "CPLEX" in str(primary_solver).upper():
            try:
                import docplex  # noqa: F401
                solver_ok = True
            except Exception as ex:
                solver_err_msg = "docplex (CPLEX) not importable: " + str(ex)
                solver_ok = False
        elif primary_solver and "GUROBI" in str(primary_solver).upper():
            try:
                import gurobipy  # noqa: F401
                solver_ok = True
            except Exception as ex:
                solver_err_msg = "gurobipy not importable: " + str(ex)
                solver_ok = False
        else:
            # unknown solver configured, attempt docplex anyway
            try:
                import docplex  # noqa: F401
                solver_ok = True
            except Exception:
                solver_ok = False
                solver_err_msg = "Neither CPLEX nor Gurobi available."

        if not solver_ok:
            missing.append(f"solver availability: {solver_err_msg or 'solver not available'}")

        if missing:
            # Compose informative error message
            msg = "ExperimentRunner configuration validation failed. The following required config entries are missing or null:\n"
            for m in missing:
                msg += f" - {m}\n"
            msg += "Please set them in config.yaml before running experiments."
            raise ValueError(msg)

    def _discover_instance_paths(self) -> List[Path]:
        """
        Discover instance JSON files in instances_dir. Returns list of Paths.
        """
        inst_paths = sorted(self.instances_dir.glob("*.json"))
        return inst_paths

    def run_experiment_batch(self, experiment_key: str, instance_paths: Optional[Iterable[Union[str, Path]]] = None) -> pd.DataFrame:
        """
        Run the named experiment (as configured in config.yaml) over the specified instances.
        If instance_paths omitted, discover instances in config.output.instances_dir.

        Returns aggregated pandas DataFrame for the experiment.
        """
        experiments_cfg = self.config.get("experiments", {}) or {}
        if experiment_key not in experiments_cfg:
            raise KeyError(f"Experiment '{experiment_key}' not found in config.experiments")

        spec = experiments_cfg[experiment_key] or {}
        time_limit = float(spec.get("time_limit_seconds", 3600))

        # prepare instance list
        paths: List[Path] = []
        if instance_paths:
            for p in instance_paths:
                paths.append(Path(p))
        else:
            paths = self._discover_instance_paths()

        results: List[ExperimentResult] = []
        aggregated_rows: List[Dict[str, Any]] = []

        # per-experiment output directory
        exp_out_dir = self.results_dir / experiment_key
        exp_out_dir.mkdir(parents=True, exist_ok=True)

        # save copy of config used
        cfg_copy_path = exp_out_dir / "config_used.yaml"
        with open(cfg_copy_path, "wt", encoding="utf-8") as cf:
            yaml.safe_dump(self.config, cf)

        # iterate instances
        for inst_path in paths:
            try:
                res = self.run_instance(inst_path, spec, time_limit, exp_out_dir)
            except Exception as ex:
                # record error summary
                self._log("instance_run_failed", {"instance": str(inst_path), "error": str(ex)})
                # create result with error
                res = ExperimentResult(
                    instance_id=str(inst_path.stem),
                    seed=None,
                    run_start=datetime.utcnow().isoformat() + "Z",
                    run_end=datetime.utcnow().isoformat() + "Z",
                    total_time_s=0.0,
                    LB=None,
                    UB=None,
                    gap_percent=None,
                    solved=False,
                    num_columns=0,
                    num_sr_cuts=0,
                    nodes_explored=0,
                    pricing_counts={},
                    cg_iterations=[],
                    node_logs=[],
                    error=str(ex),
                    meta={},
                )
                # persist minimal JSON
                outf = exp_out_dir / f"{inst_path.stem}_error.json"
                with open(outf, "wt", encoding="utf-8") as f:
                    json.dump(res.__dict__, f, indent=2, default=str)
            # append to results list
            results.append(res)

            # prepare a simple aggregated row for CSV: many experiments have different metrics,
            # include broadly useful columns
            aggregated_rows.append(
                {
                    "instance_id": res.instance_id,
                    "seed": res.seed,
                    "run_start": res.run_start,
                    "run_end": res.run_end,
                    "total_time_s": res.total_time_s,
                    "LB": res.LB,
                    "UB": res.UB,
                    "gap_percent": res.gap_percent,
                    "solved": res.solved,
                    "num_columns": res.num_columns,
                    "num_sr_cuts": res.num_sr_cuts,
                    "nodes_explored": res.nodes_explored,
                }
            )

        # Save aggregated CSV
        df = pd.DataFrame(aggregated_rows)
        csv_path = exp_out_dir / f"{experiment_key}_aggregated.csv"
        df.to_csv(csv_path, index=False)

        # Also save full JSON results list
        json_full = exp_out_dir / f"{experiment_key}_results.json"
        with open(json_full, "wt", encoding="utf-8") as jf:
            # serialize ExperimentResult objects
            out_list = [r.__dict__ for r in results]
            json.dump(out_list, jf, indent=2, default=str)

        # Optionally produce plots if configured
        try:
            plot_cfg = self.config.get("output", {}) or {}
            if plot_cfg.get("produce_plots", False):
                self.plot_results(df, experiment_key, out_dir=exp_out_dir)
        except Exception:
            pass

        return df

    def run_instance(self, instance_path: Union[str, Path], spec: Dict[str, Any], time_limit: Optional[float], exp_out_dir: Path) -> ExperimentResult:
        """
        Run experiment for a single instance path. This orchestrates:
         - load instance
         - build DistanceMatrix, ColumnPool, RLMP_Solver, PricingManager, SRSeparator
         - root column-and-cut generation (CCG)
         - optionally run full Branch-and-Price-and-Cut (BPC)
         - collect metrics and save per-instance JSON file into exp_out_dir

        Returns ExperimentResult
        """
        t_total_start = time.time()
        inst_path = Path(instance_path)
        run_start_iso = datetime.utcnow().isoformat() + "Z"

        # default ExperimentResult
        result = ExperimentResult(instance_id=inst_path.stem, seed=None, run_start=run_start_iso)

        try:
            instance = Instance.load(inst_path)
        except Exception as ex:
            result.error = f"Failed to load instance: {ex}"
            result.run_end = datetime.utcnow().isoformat() + "Z"
            return result

        # set RNG seeds deterministically from instance meta if available
        seed = None
        try:
            seed = int(instance.meta.get("generator_seed") or instance.meta.get("seed") or self.config.get("reproducibility", {}).get("global_random_seed", 0))
        except Exception:
            seed = int(self.config.get("reproducibility", {}).get("global_random_seed", 0) or 0)
        result.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Build distances
        try:
            distances = DistanceMatrix(instance)
            distances.compute_all()
        except Exception as ex:
            result.error = f"DistanceMatrix failure: {ex}"
            result.run_end = datetime.utcnow().isoformat() + "Z"
            return result

        # Create column pool and RLMP solver and pricing manager and SR separator
        column_pool = ColumnPool(config=self.config)
        rlmp = RLMP_Solver(instance=instance, column_pool=column_pool, config=self.config, logger=self.logger)
        pricing_mgr = PricingManager(instance=instance, distances=distances, column_pool=column_pool, config=self.config, logger=self.logger, rng=np.random.RandomState(seed))
        sr_sep = SRSeparator(rlmp)
        bnb = BranchAndBound(instance=instance, rlmp_solver=rlmp, pricing_mgr=pricing_mgr, sr_separator=sr_sep, config=self.config, logger=self.logger)

        # Root column-and-cut generation (CCG)
        cg_start = time.time()
        cg_iterations = []
        total_columns_added = 0
        total_sr_added = 0
        pricing_counts = {"labeler": 0, "greedy": 0, "tabu": 0, "ng_route": 0}

        # Seed initial columns
        try:
            rlmp.seed_initial_columns()
        except Exception as ex:
            # If seeding fails, proceed but log
            self._log("seed_initial_columns_failed", {"instance": instance.id, "error": str(ex)})
        # Now loop
        root_time_limit = float(spec.get("time_limit_seconds", time_limit or self.config.get("time_limits", {}).get("cg_root_enhancement_test_seconds", 1800)))
        cg_deadline = time.time() + root_time_limit
        max_ccg_iters = int(self.config.get("column_generation", {}).get("max_iterations", 1000) or 1000)
        iter_count = 0

        try:
            while True:
                iter_count += 1
                if iter_count > max_ccg_iters:
                    self._log("ccg_iter_limit_reached", {"instance": instance.id, "iters": iter_count})
                    break
                if time.time() >= cg_deadline:
                    self._log("ccg_time_limit_reached", {"instance": instance.id, "elapsed_s": time.time() - cg_start})
                    break

                # Solve RLMP LP
                lp_time_budget = min(self.config.get("time_limits", {}).get("per_pricing_call_seconds", 60.0), max(1.0, cg_deadline - time.time()))
                lp_obj, primal_lambda, duals = rlmp.solve_lp(time_limit_seconds=lp_time_budget)
                iter_entry = {"iteration": iter_count, "lp_obj": lp_obj, "time": time.time(), "new_columns": 0, "new_sr": 0}
                # Pricing: attempt to generate columns
                pricing_time_budget = float(self.config.get("time_limits", {}).get("per_pricing_call_seconds", 60.0))
                new_columns = pricing_mgr.generate_columns(duals, forbidden_arcs=set(), forced_arcs=set(), time_budget=pricing_time_budget)
                # update counts
                pricing_counts["labeler"] += 0  # heuristics/pricing manager already logs counts internally; we keep zeros as placeholder
                if new_columns:
                    rlmp.add_columns(new_columns)
                    iter_entry["new_columns"] = len(new_columns)
                    total_columns_added += len(new_columns)
                    # continue iteration
                    cg_iterations.append(iter_entry)
                    continue

                # No columns found, attempt SR separation
                sr_cuts = sr_sep.separate(primal_lambda)
                if sr_cuts:
                    rlmp.add_sr_cuts(sr_cuts)
                    iter_entry["new_sr"] = len(sr_cuts)
                    total_sr_added += len(sr_cuts)
                    cg_iterations.append(iter_entry)
                    continue

                # Neither columns nor SRs -> stop
                cg_iterations.append(iter_entry)
                break
        except Exception as ex:
            # record error but try to continue to saving
            self._log("ccg_exception", {"instance": instance.id, "error": str(ex)})
            result.error = f"CCG failure: {ex}"

        cg_end = time.time()
        result.cg_iterations = cg_iterations
        result.num_columns = len(column_pool.get_all())
        result.num_sr_cuts = total_sr_added
        result.pricing_counts = pricing_counts

        # After root CCG, collect LB and possibly integer solution at root
        try:
            lp_obj, primal_lambda, duals = rlmp.solve_lp(time_limit_seconds=10.0)
            result.LB = float(lp_obj) if lp_obj is not None else None
        except Exception:
            result.LB = None

        # If experiment requires running full BPC, run it
        run_bpc = bool(spec.get("run_full_bpc", True))
        remaining_time = (time_limit or float(self.config.get("time_limits", {}).get("performance_small_seconds", 7200))) - (time.time() - t_total_start)
        if remaining_time < 0:
            remaining_time = 0.0

        if run_bpc and remaining_time > 1e-6:
            try:
                bnb_result = bnb.run(time_limit=remaining_time)
                # bnb_result contains incumbent solution and stats
                inc = bnb_result.get("incumbent_solution", None)
                inc_obj = bnb_result.get("incumbent_obj", None)
                result.UB = float(inc_obj) if inc_obj is not None and math.isfinite(inc_obj) else None
                result.solved = bool(inc is not None and inc_obj is not None)
                result.nodes_explored = bnb_result.get("stats", {}).get("nodes_explored")
                # record statistics
                result.meta["bnb_stats"] = bnb_result.get("stats", {})
            except Exception as ex:
                self._log("bnb_exception", {"instance": instance.id, "error": str(ex)})
                result.error = f"BPC run failure: {ex}"

        else:
            # Try to recover a feasible integer solution by solving sub-RLMP IP if any columns exist
            try:
                ip_obj, ip_sol = rlmp.solve_ip(time_limit=30.0)
                if ip_obj is not None:
                    result.UB = float(ip_obj)
                    result.solved = True
            except Exception:
                pass

        # Finalize LB/UB/gap
        if result.LB is None:
            # attempt to get last LP objective
            try:
                lp_obj, _, _ = rlmp.solve_lp(time_limit_seconds=5.0)
                result.LB = float(lp_obj) if lp_obj is not None else None
            except Exception:
                result.LB = result.LB

        if result.UB is None:
            # best effort: try to extract best integer via rlmp.solve_ip quickly
            try:
                ip_obj, ip_sol = rlmp.solve_ip(time_limit=10.0)
                if ip_obj is not None:
                    result.UB = float(ip_obj)
            except Exception:
                pass

        if (result.UB is not None) and (result.LB is not None) and result.UB > 0:
            result.gap_percent = 100.0 * (result.UB - result.LB) / max(1e-8, result.UB)
        else:
            result.gap_percent = None

        result.run_end = datetime.utcnow().isoformat() + "Z"
        result.total_time_s = time.time() - t_total_start

        # write per-instance JSON result
        inst_out_path = exp_out_dir / f"{instance.id}_result.json"
        with open(inst_out_path, "wt", encoding="utf-8") as outf:
            json.dump(result.__dict__, outf, indent=2, default=str)

        # also log short summary
        self._log("instance_run_complete", {"instance": instance.id, "LB": result.LB, "UB": result.UB, "gap_percent": result.gap_percent, "total_time_s": result.total_time_s})

        return result

    def aggregate_results(self, results: List[ExperimentResult], experiment_key: str) -> pd.DataFrame:
        """
        Aggregate per-instance ExperimentResult objects into a pandas DataFrame.

        Returns DataFrame.
        """
        rows = []
        for r in results:
            rows.append(
                {
                    "instance_id": r.instance_id,
                    "seed": r.seed,
                    "run_start": r.run_start,
                    "run_end": r.run_end,
                    "total_time_s": r.total_time_s,
                    "LB": r.LB,
                    "UB": r.UB,
                    "gap_percent": r.gap_percent,
                    "solved": r.solved,
                    "num_columns": r.num_columns,
                    "num_sr_cuts": r.num_sr_cuts,
                    "nodes_explored": r.nodes_explored,
                }
            )
        df = pd.DataFrame(rows)
        out_csv = self.results_dir / experiment_key / f"{experiment_key}_aggregated_summary.csv"
        df.to_csv(out_csv, index=False)
        return df

    def plot_results(self, aggregated_df: pd.DataFrame, experiment_key: str, out_dir: Optional[Path] = None) -> None:
        """
        Produce a couple of simple diagnostic plots from aggregated DataFrame and
        save them into experiment results directory.

        - Cost (UB) histogram
        - Gap percent distribution
        """
        out_dir = Path(out_dir or (self.results_dir / experiment_key))
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            if "UB" in aggregated_df.columns and aggregated_df["UB"].notna().any():
                plt.figure(figsize=(8, 4))
                aggregated_df["UB"].dropna().hist(bins=20)
                plt.title(f"{experiment_key} - Distribution of UB")
                plt.xlabel("UB (operating cost)")
                plt.ylabel("Count")
                fpath = out_dir / f"{experiment_key}_UB_hist.png"
                plt.tight_layout()
                plt.savefig(fpath)
                plt.close()

            if "gap_percent" in aggregated_df.columns and aggregated_df["gap_percent"].notna().any():
                plt.figure(figsize=(8, 4))
                aggregated_df["gap_percent"].dropna().hist(bins=20)
                plt.title(f"{experiment_key} - Gap % Distribution")
                plt.xlabel("Gap %")
                plt.ylabel("Count")
                fpath = out_dir / f"{experiment_key}_gap_hist.png"
                plt.tight_layout()
                plt.savefig(fpath)
                plt.close()
        except Exception as ex:
            self._log("plotting_error", {"experiment": experiment_key, "error": str(ex)})


# If executed as script, run a quick discovery and print config summary
if __name__ == "__main__":
    runner = ExperimentRunner("config.yaml")
    print("Config loaded. Available experiments:", list((runner.config.get("experiments") or {}).keys()))
    # Example: run first experiment with default instance discovery (may be heavy)
    exps = list((runner.config.get("experiments") or {}).keys())
    if exps:
        key = exps[0]
        print(f"Running experiment '{key}' on discovered instances (this may be long).")
        try:
            df = runner.run_experiment_batch(key)
            print("Aggregated results written. Preview:")
            print(df.head())
        except Exception as e:
            print("Experiment run failed:", e)
    else:
        print("No experiments configured in config.yaml.")
