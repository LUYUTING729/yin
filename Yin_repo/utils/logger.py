"""utils/logger.py

Structured, thread-safe logger for the BPC reproduction project.

This module implements the Logger class described in the design. The Logger writes
either newline-delimited JSON (JSON Lines) or human-readable text lines to a per-run
log file under the configured logs directory. It automatically attaches an envelope
to each event with timestamp, level, run_id, instance_id and elapsed time.

Public API (as required by the design interface):
- Logger.log(event: str, payload: Dict) -> None
- Logger.save_run(summary: Optional[Dict]=None, path: Optional[str]=None) -> None

Usage:
    logger = Logger(run_id="run123", instance_id="inst_A", config=config_data)
    logger.log("experiment_start", {"config_snapshot_path": "results/run123/config.json", "seed": 123})
    ...
    logger.save_run(summary={"status": "completed", "final_objective": 123.45})
"""

from __future__ import annotations

import json
import os
import threading
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Union, IO
import datetime

try:
    import yaml
except Exception:  # pragma: no cover - yaml should be available per requirements
    yaml = None  # type: ignore

# Default configuration keys and default fallback values (used when config is incomplete)
_DEFAULT_LOG_LEVEL = "INFO"
_DEFAULT_STRUCTURED_JSON = True
_DEFAULT_CG_ITER_LOG = True
_DEFAULT_PER_NODE_LOG = True
_DEFAULT_LOGS_DIR = "logs"
_DEFAULT_RESULTS_DIR = "results"
_DEFAULT_SAVE_RLMP_SNAPSHOTS = False
_DEFAULT_GLOBAL_SEED = None
_DEFAULT_DETERMINISTIC_HASHING = True


def _now_iso_utc() -> str:
    """Return current UTC time in ISO8601 with Z suffix and millisecond precision."""
    return datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def _safe_makedirs(path: Union[str, Path]) -> None:
    p = Path(path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)


def _hash_config(obj: Any) -> str:
    """Return a short SHA256 hex digest for a Python object by JSON-serializing it."""
    try:
        data = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        # Fallback: convert to str
        data = str(obj).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:12]


class Logger:
    """
    Thread-safe structured logger.

    Public methods:
      - log(event: str, payload: Dict[str, Any], level: Optional[str] = None) -> None
      - save_run(summary: Optional[Dict[str, Any]] = None, path: Optional[str] = None) -> None

    Initialization:
      Logger(run_id: Optional[str], instance_id: Optional[str], config: Union[Dict,str,None])

    Parameters:
      - run_id: Unique identifier for the run. If None, will be deterministically generated
                from timestamp, instance_id and config reproducibility seed/hash.
      - instance_id: Optional instance identifier string.
      - config: Either a dict-like configuration or a path to a YAML config file. If None,
                default values will be used for relevant logging keys.
    """

    def __init__(
        self,
        run_id: Optional[str],
        instance_id: Optional[str],
        config: Optional[Union[Dict[str, Any], str]] = None,
    ) -> None:
        # Load config if a path is provided
        self._config: Dict[str, Any] = {}
        if isinstance(config, str):
            # treat as path to YAML
            if yaml is None:
                raise RuntimeError("PyYAML is required to load config from YAML path")
            try:
                with open(config, "rt", encoding="utf-8") as f:
                    self._config = yaml.safe_load(f) or {}
            except Exception:
                # fall back to empty config
                self._config = {}
        elif isinstance(config, dict):
            self._config = dict(config)
        else:
            self._config = {}

        # Extract logging-related config with safe defaults
        logging_cfg = self._config.get("logging", {}) or {}
        output_cfg = self._config.get("output", {}) or {}
        reproducibility_cfg = self._config.get("reproducibility", {}) or {}

        self._level = str(logging_cfg.get("level", _DEFAULT_LOG_LEVEL)).upper()
        self._structured = bool(logging_cfg.get("structured_json", _DEFAULT_STRUCTURED_JSON))
        self._cg_iteration_log = bool(logging_cfg.get("cg_iteration_log", _DEFAULT_CG_ITER_LOG))
        self._per_node_log = bool(logging_cfg.get("per_node_log", _DEFAULT_PER_NODE_LOG))

        self._logs_dir = Path(output_cfg.get("logs_dir", _DEFAULT_LOGS_DIR))
        self._results_dir = Path(output_cfg.get("results_dir", _DEFAULT_RESULTS_DIR))
        self._save_rlmp_snapshots = bool(
            output_cfg.get("save_rlmp_snapshots", _DEFAULT_SAVE_RLMP_SNAPSHOTS)
        )

        self._global_seed = reproducibility_cfg.get("global_random_seed", _DEFAULT_GLOBAL_SEED)
        self._deterministic_hashing = bool(
            reproducibility_cfg.get("deterministic_hashing", _DEFAULT_DETERMINISTIC_HASHING)
        )

        # Ensure directories exist
        _safe_makedirs(self._logs_dir)
        _safe_makedirs(self._results_dir)

        # Build deterministic run_id if not provided
        if run_id and isinstance(run_id, str) and run_id.strip():
            self.run_id = run_id
        else:
            # deterministic run id using timestamp + instance_id + config hash + seed
            ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            inst = instance_id or "global"
            cfg_hash = _hash_config(self._config) if self._deterministic_hashing else "na"
            seed_part = f"seed-{self._global_seed}" if self._global_seed is not None else "noseed"
            self.run_id = f"{ts}__{inst}__{seed_part}__{cfg_hash}"

        self.instance_id = instance_id

        # Create file handle
        if self._structured:
            fname = f"{self.run_id}.log.jsonl"
            mode = "a"
            self._log_path = self._logs_dir / fname
            # open in text mode with buffering disabled for immediate flush behavior
            self._fh: IO[str] = open(self._log_path, mode, encoding="utf-8")
        else:
            fname = f"{self.run_id}.log.txt"
            mode = "a"
            self._log_path = self._logs_dir / fname
            self._fh = open(self._log_path, mode, encoding="utf-8")

        # Internal lock for thread safety
        self._lock = threading.Lock()

        # record start time for elapsed_time calculations
        self._start_time = time.time()

        # counters and minimal aggregates (kept here for convenience)
        self._counters: Dict[str, int] = {}
        # keep a minimal list of saved artifacts to record in summary if needed
        self._artifacts: Dict[str, str] = {}

        # Write initial header entry (not imposing experimental_start payload; caller may log it)
        header_payload = {
            "config_snapshot_hash": _hash_config(self._config),
            "config_keys": list(self._config.keys()),
            "repro_seed": self._global_seed,
            "deterministic_hashing": self._deterministic_hashing,
        }
        # Try to save a copy of the config to results_dir for reproducibility
        try:
            config_snapshot_path = self._results_dir / f"{self.run_id}.config.json"
            with open(config_snapshot_path, "wt", encoding="utf-8") as cf:
                json.dump(self._config, cf, indent=2, default=str)
            header_payload["config_snapshot_path"] = str(config_snapshot_path)
        except Exception:
            # ignore snapshot failures but include that it failed
            header_payload["config_snapshot_path"] = None

        # Log header (use INFO)
        self.log("logger_initialised", header_payload, level="INFO")

    # Public API ------------------------------------------------------------

    def log(self, event: str, payload: Dict[str, Any], level: Optional[str] = None) -> None:
        """
        Log an event with a payload.

        Parameters:
            event: event name string (e.g., "cg_iteration", "rlmp_solve")
            payload: dictionary with event-specific keys (serializable by JSON)
            level: optional verbosity level for this event (e.g., "INFO", "DEBUG").
                   If omitted, defaults to INFO.
        """
        if level is None:
            level = "INFO"
        level = level.upper()

        # Respect filter for cg_iteration and per_node events
        if event == "cg_iteration" and not self._cg_iteration_log:
            return
        if event == "bn_branch_node" and not self._per_node_log:
            return

        # Level filtering: simple ordinal check
        if not self._level_allowed(level):
            return

        envelope = {
            "timestamp": _now_iso_utc(),
            "level": level,
            "run_id": self.run_id,
            "instance_id": self.instance_id,
            "node_id": payload.get("node_id") if isinstance(payload, dict) else None,
            "elapsed_time_s": round(time.time() - self._start_time, 6),
            "event": event,
            "payload": payload,
        }

        # Write to file in thread-safe manner
        try:
            line: str
            if self._structured:
                # compact JSON line
                line = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"), default=str)
            else:
                # human readable line
                ts = envelope["timestamp"]
                ev = envelope["event"]
                lvl = envelope["level"]
                # pretty payload
                payload_str = json.dumps(envelope["payload"], ensure_ascii=False, default=str)
                line = f"[{ts}] {lvl} {ev} | {payload_str}"
            with self._lock:
                self._fh.write(line + "\n")
                # flush to disk immediately to preserve progress in long-running runs
                try:
                    self._fh.flush()
                    os.fsync(self._fh.fileno())
                except Exception:
                    # If fsync not available or fails, ignore but do not raise to caller
                    pass
        except Exception:
            # Attempt to fallback to stderr without raising
            try:
                import sys

                err_msg = f"Logger write failure for event={event}; payload={repr(payload)}"
                sys.stderr.write(err_msg + "\n")
            except Exception:
                pass

    def save_run(self, summary: Optional[Dict[str, Any]] = None, path: Optional[Union[str, Path]] = None) -> None:
        """
        Finalize logging for the run: flush and close handles, write a summary JSON.

        Parameters:
            summary: optional dictionary with summary keys (status, final_objective, etc.)
            path: optional explicit path to save the summary JSON. If None, a default path
                  results_dir/<run_id>.summary.json is used.
        """
        # Build a minimal summary if none provided
        end_time = time.time()
        total_time = round(end_time - self._start_time, 6)
        default_summary = {
            "run_id": self.run_id,
            "instance_id": self.instance_id,
            "start_time": datetime.datetime.utcfromtimestamp(self._start_time).isoformat() + "Z",
            "end_time": datetime.datetime.utcfromtimestamp(end_time).isoformat() + "Z",
            "total_time_s": total_time,
            "num_logged_events": None,
            "artifacts": self._artifacts,
        }
        if summary:
            # merge provided summary (provided keys override defaults)
            merged = dict(default_summary)
            merged.update(summary)
            summary = merged
        else:
            summary = default_summary

        # close file handle safely
        try:
            with self._lock:
                try:
                    self._fh.flush()
                    os.fsync(self._fh.fileno())
                except Exception:
                    # ignore fsync/flush failures
                    pass
                try:
                    self._fh.close()
                except Exception:
                    pass
        except Exception:
            # best-effort close; continue to write summary
            pass

        # Determine path
        if path:
            summary_path = Path(path)
        else:
            summary_path = Path(self._results_dir) / f"{self.run_id}.summary.json"

        # Write summary JSON
        try:
            _safe_makedirs(summary_path.parent)
            with open(summary_path, "wt", encoding="utf-8") as sf:
                json.dump(summary, sf, indent=2, default=str)
            # Also log that we saved the summary (open a fresh handle to append a small entry)
            try:
                with open(self._logs_dir / f"{self.run_id}.log.meta.json", "wt", encoding="utf-8") as mf:
                    meta = {
                        "summary_path": str(summary_path),
                        "written_at": _now_iso_utc(),
                        "run_id": self.run_id,
                        "instance_id": self.instance_id,
                    }
                    json.dump(meta, mf, indent=2)
            except Exception:
                # non-critical
                pass
        except Exception as ex:
            # If we cannot save the summary, attempt to write to stderr
            try:
                import sys

                sys.stderr.write(f"Failed to write logger summary to {summary_path}: {ex}\n")
            except Exception:
                pass

    # Internal helpers -----------------------------------------------------

    def _level_allowed(self, event_level: str) -> bool:
        """
        Determine if an event at event_level should be logged given the configured minimum level.

        Supported levels (ordered): DEBUG < INFO < WARNING < ERROR < CRITICAL
        """
        order = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
        min_lvl = self._level if isinstance(self._level, str) else _DEFAULT_LOG_LEVEL
        min_val = order.get(min_lvl.upper(), 20)
        ev_val = order.get(event_level.upper(), 20)
        return ev_val >= min_val

    # Convenience methods for internal bookkeeping (kept private to respect design)
    def _inc_counter(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + int(amount)

    def _get_counter(self, name: str) -> int:
        return int(self._counters.get(name, 0))

    def _register_artifact(self, key: str, path: str) -> None:
        with self._lock:
            self._artifacts[key] = path


# Module-level basic sanity check when run as script (demonstration only)
if __name__ == "__main__":  # pragma: no cover - demonstration
    # Simple demo of using the Logger
    demo_config = {
        "logging": {"level": "DEBUG", "structured_json": True, "cg_iteration_log": True, "per_node_log": True},
        "output": {"logs_dir": "logs_demo", "results_dir": "results_demo", "save_rlmp_snapshots": False},
        "reproducibility": {"global_random_seed": 123, "deterministic_hashing": True},
    }
    lg = Logger(run_id=None, instance_id="demo_inst", config=demo_config)
    lg.log("experiment_start", {"config_snapshot_path": "results_demo/demo_config.json", "seed": 123})
    lg.log("cg_iteration", {"iteration": 1, "rlmp_obj": 12.345, "num_columns": 10})
    lg.log("bn_branch_node", {"node_id": "n1", "status": "created"})
    lg.log("incumbent_update", {"new_objective": 10.5, "time_s": 12.3})
    lg.save_run({"status": "completed", "final_objective": 10.5})
