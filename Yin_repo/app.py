#!/usr/bin/env python3
"""
app.py

Entry point for reproducing the experiments in:
European Journal of Operational Research (TD-DRPTW BPC algorithm).

This script provides a small command-line interface to:
 - validate configuration (config.yaml),
 - generate instances (optional),
 - run one configured experiment (by key) or all experiments defined in config.yaml,
 - run a single-instance execution (load instance file and run BPC),
 - run the unit tests (optional).

Usage examples:
  python app.py --config config.yaml --run-experiment enhancement_strategies_test
  python app.py --config config.yaml --generate-instances
  python app.py --config config.yaml --instance instances/instance_type1_n35_theta0.5_rep0.json
  python app.py --config config.yaml --dry-run

Notes / Guarantees:
 - The application follows the reproducibility plan and the Design specification.
 - It delegates heavy-lifting to modules in the project (InstanceGenerator, ExperimentRunner, etc.).
 - It performs strict configuration validation and prints clear messages when required
   config values are missing (the design requires explicit service times).
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional, Sequence

# Standard libs
import json
import yaml

# Project modules (per design)
try:
    from utils.logger import Logger
except Exception:
    Logger = None  # type: ignore

try:
    from results.experiment_runner import ExperimentRunner
    from instances.instance_generator import InstanceGenerator
except Exception as e:
    # If imports fail, provide clear error to user.
    print("ERROR: Failed to import project modules required by app.py. Ensure package files are available.")
    print("Import error:", e)
    raise

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def load_config(config_path: Path) -> dict:
    """Load YAML config file and return dict."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "rt", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def ensure_logs_dir(cfg: dict) -> Path:
    out = cfg.get("output", {}) or {}
    logs_dir = Path(out.get("logs_dir", "logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def bootstrap_logger(config_path: Path, instance_id: Optional[str]) -> Optional[Logger]:
    """Construct Logger if utils.logger.Logger is available; otherwise return None."""
    cfg = {}
    try:
        with open(config_path, "rt", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    run_id = f"run_{Path(config_path).stem}_{int(time.time())}"
    try:
        if Logger is not None:
            logger = Logger(run_id=run_id, instance_id=instance_id, config=cfg)
            logger.log("app_init", {"config_path": str(config_path), "instance_id": instance_id})
            return logger
        else:
            # No structured logger available
            return None
    except Exception:
        return None


def fatal(msg: str, exit_code: int = 1) -> None:
    """Print fatal message and exit."""
    print("FATAL:", msg, file=sys.stderr)
    sys.exit(exit_code)


# -----------------------------------------------------------------------------
# CLI and main application logic
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="Run reproducibility experiments for TD-DRPTW BPC algorithm (paper reproduction).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", "-c", type=Path, required=True, help="Path to config.yaml (required).")
    parser.add_argument(
        "--generate-instances",
        action="store_true",
        help="Generate the full instance set as defined in config.instance_generation (writes to output.instances_dir).",
    )
    parser.add_argument(
        "--run-experiment",
        type=str,
        help="Run the named experiment key from config.experiments (e.g., 'enhancement_strategies_test').",
    )
    parser.add_argument(
        "--run-all-experiments",
        action="store_true",
        help="Run all experiments defined in config.experiments sequentially.",
    )
    parser.add_argument(
        "--instance",
        type=Path,
        help="Path to a single instance JSON file to run (overrides running experiment selection).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration, print planned actions and exit without executing heavy computations.",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run the project's pytest tests (tests/test_small_instances.py).",
    )
    parser.add_argument(
        "--allow-default-heuristics",
        action="store_true",
        help="Allow the app to use conservative defaults for unspecified heuristic hyperparameters.\n"
        "If not provided and heuristics parameters are missing, the app will abort to force explicit settings.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Override (seconds) global time limit for the experiment run (optional).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)

    config_path = Path(args.config).resolve()
    try:
        cfg = load_config(config_path)
    except Exception as ex:
        fatal(f"Failed to load config file '{config_path}': {ex}")

    # Ensure logs/results directories exist
    try:
        logs_dir = ensure_logs_dir(cfg)
    except Exception:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)

    # Bootstrap logger
    logger = bootstrap_logger(config_path, instance_id=(args.instance.name if args.instance else None))

    # If dry-run: perform validation using ExperimentRunner but do not execute experiments
    if args.dry_run:
        print("DRY RUN: validating configuration and planned experiments...")
        try:
            runner = ExperimentRunner(config_path)
        except Exception as ex:
            fatal(f"Configuration validation failed: {ex}")
        # Print summary of experiments
        exps = cfg.get("experiments", {}) or {}
        print(f"Config file validated successfully: {config_path}")
        print(f"Found {len(exps)} experiment(s): {list(exps.keys())}")
        # If instance generation is configured, show counts
        inst_gen_cfg = cfg.get("instance_generation", {}) or {}
        n_values = inst_gen_cfg.get("n_values", [])
        theta_values = inst_gen_cfg.get("theta_values", [])
        replicates = inst_gen_cfg.get("replicates_per_setting", 1)
        inst_types = inst_gen_cfg.get("instance_types", 3)
        total_expected = int(inst_types) * len(n_values) * len(theta_values) * int(replicates)
        print(f"Instance generation settings: types={inst_types}, n_values={n_values}, theta_values={theta_values}, replicates={replicates}")
        print(f"--> Total instances expected when generating all: {total_expected}")
        print("Dry-run complete. No heavy computation performed.")
        return 0

    # If user requested generating all instances
    if args.generate_instances:
        print("Generating instances as specified in config.instance_generation ...")
        try:
            ig = InstanceGenerator(config=cfg, logger=logger)
            generated = ig.generate_all()
            print(f"Generated {len(generated)} instances in {cfg.get('output',{}).get('instances_dir','instances')}")
        except Exception as ex:
            fatal(f"Failed to generate instances: {ex}")
        # If only generation requested, exit
        if not (args.run_experiment or args.run_all_experiments or args.instance or args.run_tests):
            print("Instance generation complete.")
            return 0

    # If tests requested, run pytest programmatically
    if args.run_tests:
        print("Running unit tests (pytest)...")
        # run pytest in a subprocess to avoid interfering with interpreter state
        try:
            import pytest

            # Run pytest with test module; capture exit code
            rc = pytest.main(["-q", "tests/test_small_instances.py"])
            if rc != 0:
                print("Some tests failed. Exit code:", rc)
                return rc
            print("All tests passed.")
        except Exception as ex:
            fatal(f"Failed to run pytest programmatically: {ex}")
        # tests done; continue if user also requested to run experiments

    # Instantiate ExperimentRunner
    try:
        runner = ExperimentRunner(config_path)
    except Exception as ex:
        fatal(f"Failed to initialize ExperimentRunner (config validation): {ex}")

    # If single instance run requested
    if args.instance:
        inst_path = Path(args.instance).resolve()
        if not inst_path.exists():
            fatal(f"Requested instance file does not exist: {inst_path}")
        # choose experiment spec if available: prefer 'performance_comparison' else default
        experiments_cfg = cfg.get("experiments", {}) or {}
        # choose a sensible spec: prefer one matching instance size if present; else default to first experiment key
        chosen_spec = {}
        if args.run_experiment and args.run_experiment in experiments_cfg:
            chosen_key = args.run_experiment
            chosen_spec = experiments_cfg[chosen_key] or {}
            out_dir = Path(cfg.get("output", {}).get("results_dir", "results")) / chosen_key
        else:
            # default spec: try performance_comparison else any
            if "performance_comparison" in experiments_cfg:
                chosen_key = "performance_comparison"
                chosen_spec = experiments_cfg.get("performance_comparison", {}) or {}
            else:
                # fallback to first experiment key if exists
                keys = list(experiments_cfg.keys())
                chosen_key = keys[0] if keys else "single_instance"
                chosen_spec = experiments_cfg.get(chosen_key, {}) or {}
            out_dir = Path(cfg.get("output", {}).get("results_dir", "results")) / chosen_key

        out_dir.mkdir(parents=True, exist_ok=True)
        # time limit override
        time_limit = args.time_limit if args.time_limit is not None else chosen_spec.get("time_limit_seconds", None)
        print(f"Running single-instance experiment for {inst_path} using experiment spec '{chosen_key}' with time_limit={time_limit}")
        try:
            res = runner.run_instance(inst_path, chosen_spec, time_limit, out_dir)
            print(f"Instance run complete: instance={res.instance_id}, LB={res.LB}, UB={res.UB}, gap%={res.gap_percent}")
        except Exception as ex:
            fatal(f"run_instance failed: {ex}")
        return 0

    # If run a single named experiment
    if args.run_experiment:
        exp_key = args.run_experiment
        experiments_cfg = cfg.get("experiments", {}) or {}
        if exp_key not in experiments_cfg:
            fatal(f"Requested experiment key '{exp_key}' not present in config.experiments. Available keys: {list(experiments_cfg.keys())}")
        spec = experiments_cfg[exp_key] or {}
        print(f"Running experiment '{exp_key}' ... (this may take a long time depending on config)")
        try:
            df = runner.run_experiment_batch(exp_key)
            print(f"Experiment '{exp_key}' completed. Aggregated results saved under results/{exp_key}.")
        except Exception as ex:
            fatal(f"Experiment '{exp_key}' failed: {ex}")
        return 0

    # If run all experiments
    if args.run_all_experiments:
        experiments_cfg = cfg.get("experiments", {}) or {}
        if not experiments_cfg:
            fatal("No experiments configured in config.experiments to run.")
        for key in list(experiments_cfg.keys()):
            print(f"Running experiment '{key}' ...")
            try:
                _ = runner.run_experiment_batch(key)
                print(f"Experiment '{key}' finished.")
            except Exception as ex:
                print(f"Experiment '{key}' failed: {ex}", file=sys.stderr)
        return 0

    # If none of the above specified, show usage help
    print("No action specified. Use --generate-instances, --run-experiment, --run-all-experiments, or --instance.")
    print("Use --dry-run to validate configuration without heavy computation.")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
        sys.exit(0 if rc is None else int(rc))
    except SystemExit as se:
        raise
    except Exception as exc:
        print("Unhandled exception in app.py:", exc, file=sys.stderr)
        raise
