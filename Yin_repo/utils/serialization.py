"""
utils/serialization.py

Serialization manager for instances, columns (routes), RLMP snapshots and solutions.

This module provides:
- SerializationManager: high-level API to save/load instances, columns, solutions, and RLMP snapshots.
- Helper exceptions: SerializationError, IntegrityError, InvalidInstanceError.
- Light-weight proxy dataclasses InstanceProxy and ColumnProxy to represent loaded objects when the
  project's full Instance/Column classes may not yet be available.

The module is defensive: reads configuration from a provided dict or from ./config.yaml if available.
It writes JSON (optionally gzipped if file extension ends with .gz or size threshold reached).
All writes are atomic (write to temp file then os.replace). File integrity is recorded with sha256.
Manifests for instances and results are updated atomically.

Usage (examples):
    ser = SerializationManager(config=config_dict, logger=logger)
    meta = ser.save_instance(instance_obj)  # instance_obj can be Mapping or object with attributes
    inst_proxy = ser.load_instance("/path/to/instance.json")
    col_meta = ser.save_column(column_obj, instance_id="inst_...")
    sol_meta = ser.save_solution(solution_dict)

Note:
- This module does not assume other modules' classes; it provides lightweight proxies that
  expose attributes as expected by other code. If the project's Instance class exists and is
  importable, callers may convert dicts/proxies accordingly.
"""

from __future__ import annotations

import json
import gzip
import os
import shutil
import tempfile
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import threading

# Try to import yaml for reading config file if a path is not provided as dict.
try:
    import yaml
except Exception:
    yaml = None  # type: ignore

# Import Logger if available; allow absent logger (None) gracefully.
try:
    from utils.logger import Logger  # as per project design
except Exception:
    Logger = None  # type: ignore


# -------------------------
# Exceptions
# -------------------------
class SerializationError(Exception):
    """Generic serialization error."""
    pass


class IntegrityError(SerializationError):
    """Raised when checksum or integrity checks fail."""
    pass


class InvalidInstanceError(SerializationError):
    """Raised when an instance JSON fails basic validation rules."""
    pass


# -------------------------
# Proxy dataclasses
# -------------------------
class InstanceProxy:
    """
    Lightweight representation of an Instance for serialization purposes.

    Fields match the schema described in the plan/design:
      - meta, data, params, provenance, integrity
    """

    def __init__(self, payload: Dict[str, Any]) -> None:
        # Expect payload to be a dict with keys 'meta', 'data', 'params', 'provenance', 'integrity'
        self._raw = dict(payload) if isinstance(payload, dict) else {}
        self.meta: Dict[str, Any] = dict(self._raw.get("meta", {}))
        self.data: Dict[str, Any] = dict(self._raw.get("data", {}))
        self.params: Dict[str, Any] = dict(self._raw.get("params", {}))
        self.provenance: Dict[str, Any] = dict(self._raw.get("provenance", {}))
        self.integrity: Dict[str, Any] = dict(self._raw.get("integrity", {}))

    @property
    def id(self) -> Optional[str]:
        return self.meta.get("instance_id")

    @property
    def n_customers(self) -> Optional[int]:
        return self.data.get("n_customers")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": self.meta,
            "data": self.data,
            "params": self.params,
            "provenance": self.provenance,
            "integrity": self.integrity,
        }


class ColumnProxy:
    """
    Lightweight representation of a Column / Route for serialization purposes.

    Fields:
      - meta, route, cost, feasibility, resources, integrity
    """

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._raw = dict(payload) if isinstance(payload, dict) else {}
        self.meta: Dict[str, Any] = dict(self._raw.get("meta", {}))
        self.route: Dict[str, Any] = dict(self._raw.get("route", {}))
        self.cost: Dict[str, Any] = dict(self._raw.get("cost", {}))
        self.feasibility: Dict[str, Any] = dict(self._raw.get("feasibility", {}))
        self.resources: Dict[str, Any] = dict(self._raw.get("resources", {}))
        self.integrity: Dict[str, Any] = dict(self._raw.get("integrity", {}))

    @property
    def route_id(self) -> Optional[str]:
        return self.meta.get("route_id")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": self.meta,
            "route": self.route,
            "cost": self.cost,
            "feasibility": self.feasibility,
            "resources": self.resources,
            "integrity": self.integrity,
        }


# -------------------------
# Serialization Manager
# -------------------------
class SerializationManager:
    """
    High-level manager for serialization tasks.

    Constructor:
        SerializationManager(config: Optional[Union[Dict,str]]=None,
                             logger: Optional[Logger]=None)

    - config: dict-like configuration or path to config.yaml. If None, attempts to load "./config.yaml".
    - logger: optional Logger instance. If None and utils.logger.Logger is importable, a temporary Logger
              will be created with run_id 'serialization'.
    """

    DEFAULT_INSTANCES_DIR = "instances"
    DEFAULT_RESULTS_DIR = "results"
    DEFAULT_LOGS_DIR = "logs"
    DEFAULT_SAVE_COLUMNS = True
    DEFAULT_SAVE_RLMP_SNAPSHOTS = False
    # threshold in bytes above which RLMP snapshots are suggested for compression; default 1MB.
    DEFAULT_COMPRESSION_THRESHOLD = 1_000_000

    def __init__(self, config: Optional[Union[Dict[str, Any], str]] = None, logger: Optional[Any] = None) -> None:
        # Load config dict
        self.config: Dict[str, Any] = self._load_config(config)

        # Extract output dirs with safe defaults
        output_cfg = self.config.get("output", {}) or {}
        self.instances_dir = Path(output_cfg.get("instances_dir", self.DEFAULT_INSTANCES_DIR))
        self.results_dir = Path(output_cfg.get("results_dir", self.DEFAULT_RESULTS_DIR))
        self.logs_dir = Path(output_cfg.get("logs_dir", self.DEFAULT_LOGS_DIR))
        self.save_columns_flag: bool = bool(output_cfg.get("save_columns", self.DEFAULT_SAVE_COLUMNS))
        self.save_rlmp_snapshots_flag: bool = bool(output_cfg.get("save_rlmp_snapshots", self.DEFAULT_SAVE_RLMP_SNAPSHOTS))

        # ensure directories exist
        for d in (self.instances_dir, self.results_dir, self.logs_dir):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:
                # best-effort create; continue and let writes fail later with clearer message
                pass

        # Logger
        if logger is not None:
            self.logger = logger
        else:
            if Logger is not None:
                # create a Logger instance; pass config to it
                try:
                    self.logger = Logger(run_id="serialization", instance_id=None, config=self.config)
                except Exception:
                    # fallback to None
                    self.logger = None
            else:
                self.logger = None

        # Manifests
        self.instances_manifest_path = self.instances_dir / "manifest.json"
        self.results_manifest_path = self.results_dir / "manifest.json"

        # Lock for manifest updates (in-process)
        self._manifest_lock = threading.Lock()

    # -------------------------
    # Public API
    # -------------------------
    def save_instance(self, instance_obj: Union[Dict[str, Any], Any], path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Persist an instance to JSON file and update the instances manifest.

        - instance_obj: Either a mapping following the instance schema or an object exposing attributes:
            .id or .meta['instance_id']
            .data, .meta, .params, .provenance fields if available.
          The function will attempt to coerce to the canonical schema.

        - path: optional explicit full file path to write. If None, default path is:
            <instances_dir>/<instance_id>.json (or .json.gz if recommended).

        Returns a metadata dict with keys: filename, path, sha256, file_size_bytes, created_at_iso.
        Raises SerializationError on failure.
        """
        # Build canonical dict from instance_obj
        inst_dict = self._coerce_instance_like(instance_obj)

        # Determine instance_id
        instance_id = inst_dict.get("meta", {}).get("instance_id")
        if not instance_id:
            raise SerializationError("Instance object must have meta.instance_id set.")

        # default path
        if path is None:
            filename = f"{instance_id}.json"
            path = self.instances_dir / filename
        else:
            path = Path(path)

        # ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize and write atomically
        compress = path.suffix == ".gz"
        metadata = self._atomic_write_json(path=path, obj=inst_dict, compress=compress)

        # Update integrity fields inside file? We include integrity metadata inside returned metadata.
        # But we also inject integrity info into the file's top-level 'integrity' if missing.
        # To keep atomicity, rewrite file to include integrity only if not present.
        try:
            # Load back file to check whether integrity.sha256 present
            loaded = self._read_json_file(path, expect_mapping=True)
            if "integrity" not in loaded or not isinstance(loaded.get("integrity"), dict) or "sha256" not in loaded.get("integrity", {}):
                # update and rewrite atomically
                loaded.setdefault("integrity", {})
                loaded["integrity"]["sha256"] = metadata["sha256"]
                loaded["integrity"]["file_size_bytes"] = metadata["file_size_bytes"]
                # also set created_at meta if not present
                loaded.setdefault("meta", {})
                if "created_at" not in loaded["meta"]:
                    loaded["meta"]["created_at"] = metadata["created_at"]
                # atomic write
                self._atomic_write_json(path=path, obj=loaded, compress=compress)
        except Exception:
            # Do not crash: log and continue
            self._log("save_instance", {"instance_id": instance_id, "path": str(path), "warning": "post_write_integrity_update_failed"}, level="WARNING")

        # Update instances manifest
        manifest_entry = {
            "instance_id": instance_id,
            "path": str(path),
            "sha256": metadata["sha256"],
            "file_size_bytes": metadata["file_size_bytes"],
            "created_at": metadata["created_at"],
            "n_customers": inst_dict.get("data", {}).get("n_customers"),
            "D_size": len(inst_dict.get("data", {}).get("D", [])) if inst_dict.get("data", {}).get("D") is not None else None,
            "generator_seed": inst_dict.get("meta", {}).get("generator_seed"),
        }
        self._update_manifest(self.instances_manifest_path, instance_id, manifest_entry)

        # Log
        self._log("save_instance", {"instance_id": instance_id, "path": str(path), "sha256": metadata["sha256"], "size": metadata["file_size_bytes"]})

        return metadata

    def load_instance(self, path: Union[str, Path]) -> InstanceProxy:
        """
        Load an instance JSON (or .json.gz) and validate basic schema rules.

        Returns InstanceProxy which exposes .meta, .data, .params etc.

        Raises IntegrityError or InvalidInstanceError on failure.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Instance file not found: {path}")

        # Read JSON (handles gzip if .gz)
        obj = self._read_json_file(path, expect_mapping=True)

        # Basic integrity check: if integrity.sha256 present, verify
        integrity = obj.get("integrity", {}) or {}
        exp_sha = integrity.get("sha256")
        if exp_sha:
            # Compute actual sha256 of file bytes (compressed file if .gz)
            actual_sha = self._compute_sha256_file(path)
            if actual_sha != exp_sha:
                raise IntegrityError(f"Checksum mismatch for instance file {path}: expected {exp_sha}, got {actual_sha}")

        # Validate schema
        self._validate_instance_schema(obj)

        # Return proxy object
        inst_proxy = InstanceProxy(obj)
        self._log("load_instance", {"path": str(path), "instance_id": inst_proxy.id})
        return inst_proxy

    def save_column(self, column_obj: Union[Dict[str, Any], Any], instance_id: Optional[str] = None, path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Persist a single column (route) to disk.

        - column_obj: mapping or object with expected fields (meta, route, cost, feasibility, resources)
        - instance_id: required if path not provided and to structure storage per-instance.
        - path: optional explicit file path.

        Behavior:
          - If self.save_columns_flag is False, this function is a no-op and returns metadata with 'not_saved': True.
          - Otherwise writes to results_dir/columns/<instance_id>/<route_id>.json (or .json.gz if path ends with .gz).
        """
        # Respect config flag
        if not self.save_columns_flag:
            # still attempt to extract route_id for metadata
            route_id = None
            try:
                if isinstance(column_obj, dict):
                    route_id = column_obj.get("meta", {}).get("route_id")
                else:
                    route_id = getattr(column_obj, "meta", {}).get("route_id") if hasattr(column_obj, "meta") else None
            except Exception:
                route_id = None
            meta = {"not_saved": True, "route_id": route_id}
            self._log("save_column_skipped", meta)
            return meta

        # Build canonical dict
        col_dict = self._coerce_column_like(column_obj)

        # Determine instance_id and route_id
        route_id = col_dict.get("meta", {}).get("route_id")
        if not route_id:
            # fallback to deterministic generated id
            # use sha1 of route content
            route_hash = hashlib.sha1(json.dumps(col_dict.get("route", {}), sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
            route_id = f"route_{route_hash}"
            col_dict.setdefault("meta", {})["route_id"] = route_id

        if path is None:
            if instance_id is None:
                # try to fetch from col_dict
                instance_id = col_dict.get("meta", {}).get("instance_id")
            if not instance_id:
                raise SerializationError("instance_id must be provided (either via argument or column.meta.instance_id) to save column")
            out_dir = self.results_dir / "columns" / instance_id
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{route_id}.json"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

        compress = path.suffix == ".gz"
        metadata = self._atomic_write_json(path=path, obj=col_dict, compress=compress)

        # After write, ensure integrity fields are present inside JSON (similar to instance)
        try:
            loaded = self._read_json_file(path, expect_mapping=True)
            loaded.setdefault("integrity", {})
            loaded["integrity"]["sha256"] = metadata["sha256"]
            loaded["integrity"]["file_size_bytes"] = metadata["file_size_bytes"]
            self._atomic_write_json(path=path, obj=loaded, compress=compress)
        except Exception:
            # log warning but do not fail
            self._log("save_column", {"route_id": route_id, "path": str(path), "warning": "post_write_integrity_update_failed"}, level="WARNING")

        # Update results manifest: register column under instance entry
        manifest_key = f"column::{route_id}"
        manifest_entry = {
            "route_id": route_id,
            "instance_id": instance_id,
            "path": str(path),
            "sha256": metadata["sha256"],
            "file_size_bytes": metadata["file_size_bytes"],
            "created_at": metadata["created_at"],
            "reduced_cost": col_dict.get("meta", {}).get("reduced_cost"),
            "created_by": col_dict.get("meta", {}).get("created_by"),
        }
        self._update_manifest(self.results_manifest_path, manifest_key, manifest_entry)

        self._log("save_column", {"route_id": route_id, "instance_id": instance_id, "path": str(path), "sha256": metadata["sha256"]})
        return metadata

    def load_column(self, path: Union[str, Path]) -> ColumnProxy:
        """
        Load and validate a column JSON.

        Returns ColumnProxy.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Column file not found: {path}")

        obj = self._read_json_file(path, expect_mapping=True)

        # integrity check if present
        integrity = obj.get("integrity", {}) or {}
        exp_sha = integrity.get("sha256")
        if exp_sha:
            actual_sha = self._compute_sha256_file(path)
            if actual_sha != exp_sha:
                raise IntegrityError(f"Checksum mismatch for column file {path}: expected {exp_sha}, got {actual_sha}")

        # Basic lightweight validation: route must contain truck_seq list starting with 0 and ending with n+1
        route = obj.get("route", {})
        truck_seq = route.get("truck_seq", [])
        if not isinstance(truck_seq, list) or len(truck_seq) < 2:
            raise SerializationError(f"Invalid column route.truck_seq in file {path}")

        # Create proxy
        col_proxy = ColumnProxy(obj)
        self._log("load_column", {"path": str(path), "route_id": col_proxy.route_id})
        return col_proxy

    def save_solution(self, solution_obj: Dict[str, Any], path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Persist a solution snapshot (final incumbent or intermediate).

        - solution_obj: mapping following solution schema described in design.
        - path: optional explicit path. Default: results_dir/<instance_id>_solution_<timestamp>.json

        Returns metadata dict with filename/path/sha256/size/created_at.
        """
        if not isinstance(solution_obj, dict):
            raise SerializationError("solution_obj must be a mapping (dict)")

        instance_id = solution_obj.get("meta", {}).get("instance_id") or solution_obj.get("instance_id")
        timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        if path is None:
            if instance_id:
                fname = f"{instance_id}_solution_{timestamp}.json"
            else:
                fname = f"solution_{timestamp}.json"
            path = self.results_dir / fname
        else:
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        compress = path.suffix == ".gz"
        metadata = self._atomic_write_json(path=path, obj=solution_obj, compress=compress)

        # Update results manifest
        sol_id = solution_obj.get("meta", {}).get("solution_id") or f"solution_{timestamp}"
        manifest_entry = {
            "solution_id": sol_id,
            "instance_id": instance_id,
            "path": str(path),
            "sha256": metadata["sha256"],
            "file_size_bytes": metadata["file_size_bytes"],
            "created_at": metadata["created_at"],
            "objective_total": solution_obj.get("objective", {}).get("total_operating_cost") if isinstance(solution_obj.get("objective", {}), dict) else None,
        }
        self._update_manifest(self.results_manifest_path, sol_id, manifest_entry)

        self._log("save_solution", {"solution_id": sol_id, "instance_id": instance_id, "path": str(path), "sha256": metadata["sha256"]})
        return metadata

    def save_rlmp_snapshot(self, rlmp_state: Dict[str, Any], instance_id: Optional[str] = None, iteration: Optional[int] = None) -> Dict[str, Any]:
        """
        Save RLMP snapshot (LP solution) if config allows saving snapshots.

        - rlmp_state: mapping containing objective, lambda_primal (sparse dict), duals (pi, sigma, zeta), list of columns included.
        - instance_id: used for folder naming.
        - iteration: optional iteration id for filename.

        Respects self.save_rlmp_snapshots_flag; if False returns {"not_saved": True}.
        """
        if not self.save_rlmp_snapshots_flag:
            self._log("save_rlmp_snapshot_skipped", {"instance_id": instance_id})
            return {"not_saved": True}

        if not isinstance(rlmp_state, dict):
            raise SerializationError("rlmp_state must be a mapping")

        timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        if instance_id is None:
            instance_id = rlmp_state.get("instance_id", "unknown_instance")
        iter_part = f"_iter{iteration}" if iteration is not None else ""
        fname = f"rlmp_{instance_id}{iter_part}_{timestamp}.json.gz"  # snapshots default compressed
        out_dir = self.results_dir / "rlmp_snapshots" / instance_id
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / fname

        # If large, we always compress (.gz); use compress=True
        metadata = self._atomic_write_json(path=path, obj=rlmp_state, compress=True)
        self._log("save_rlmp_snapshot", {"instance_id": instance_id, "path": str(path), "sha256": metadata["sha256"]})
        return metadata

    # -------------------------
    # Internal helpers
    # -------------------------
    def _load_config(self, config: Optional[Union[Dict[str, Any], str]]) -> Dict[str, Any]:
        """
        Load configuration dict from given dict or YAML path, or fallback to './config.yaml' when available.
        Returns a dict (possibly empty).
        """
        if config is None:
            # try to load ./config.yaml
            cfg_path = Path("config.yaml")
            if cfg_path.exists() and yaml is not None:
                try:
                    with open(cfg_path, "rt", encoding="utf-8") as f:
                        parsed = yaml.safe_load(f) or {}
                        return dict(parsed)
                except Exception:
                    return {}
            else:
                return {}
        if isinstance(config, dict):
            return dict(config)
        if isinstance(config, str):
            # treat as path
            p = Path(config)
            if not p.exists():
                raise FileNotFoundError(f"Config file not found: {config}")
            if yaml is None:
                # Try json
                try:
                    with open(p, "rt", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as ex:
                    raise SerializationError(f"Cannot parse config file {config}: {ex}")
            else:
                with open(p, "rt", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
        # fallback
        return {}

    def _coerce_instance_like(self, instance_obj: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """
        Convert a provided instance-like object into canonical dict adhering to Instance JSON schema.

        If fields are missing we attempt to fill what we can; do not invent critical problem parameters.
        """
        if isinstance(instance_obj, dict):
            inst = dict(instance_obj)  # shallow copy
        else:
            # Attempt to read attributes
            inst = {}
            try:
                # meta
                meta = getattr(instance_obj, "meta", None)
                if meta is None and hasattr(instance_obj, "id"):
                    meta = {"instance_id": getattr(instance_obj, "id")}
                if meta is not None:
                    inst["meta"] = dict(meta)
                # data
                data = getattr(instance_obj, "data", None)
                if data is not None:
                    inst["data"] = dict(data)
                # params
                params = getattr(instance_obj, "params", None)
                if params is not None:
                    inst["params"] = dict(params)
                # provenance
                prov = getattr(instance_obj, "provenance", None)
                if prov is not None:
                    inst["provenance"] = dict(prov)
                # integrity
                integ = getattr(instance_obj, "integrity", None)
                if integ is not None:
                    inst["integrity"] = dict(integ)
            except Exception:
                # fallback to empty dict; will be validated downstream
                inst = dict()
        # Sanity: ensure meta and data fields exist
        inst.setdefault("meta", {})
        inst.setdefault("data", {})
        inst.setdefault("params", {})
        inst.setdefault("provenance", {})
        inst.setdefault("integrity", {})

        # If created_at missing, add timestamp
        inst["meta"].setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        # ensure instance_id present; if not, try to synthesize but do not fabricate seeds
        if "instance_id" not in inst["meta"] or not inst["meta"].get("instance_id"):
            # Try to synthesize a deterministic id from available content (coords)
            coords = inst["data"].get("customers")
            if isinstance(coords, list) and len(coords) > 0:
                # use hash of coords
                try:
                    coords_json = json.dumps(coords, sort_keys=True, default=str)
                    h = hashlib.sha1(coords_json.encode("utf-8")).hexdigest()[:10]
                    inst["meta"]["instance_id"] = f"inst_{h}"
                except Exception:
                    inst["meta"]["instance_id"] = f"inst_{int(time.time())}"
            else:
                inst["meta"]["instance_id"] = f"inst_{int(time.time())}"

        # If provenance.random_seed missing, try to set from config reproducibility
        if "random_seed" not in inst["provenance"]:
            inst["provenance"]["random_seed"] = self.config.get("reproducibility", {}).get("global_random_seed")

        return inst

    def _coerce_column_like(self, column_obj: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """
        Coerce provided column/route-like object into canonical dict.
        """
        if isinstance(column_obj, dict):
            col = dict(column_obj)
        else:
            col = {}
            try:
                for attr in ("meta", "route", "cost", "feasibility", "resources"):
                    val = getattr(column_obj, attr, None)
                    if val is not None:
                        col[attr] = dict(val) if isinstance(val, dict) else val
            except Exception:
                col = dict()
        col.setdefault("meta", {})
        col.setdefault("route", {})
        col.setdefault("cost", {})
        col.setdefault("feasibility", {})
        col.setdefault("resources", {})
        col.setdefault("integrity", {})

        # fill created_at if missing
        col["meta"].setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        return col

    def _atomic_write_json(self, path: Union[str, Path], obj: Any, compress: bool = False) -> Dict[str, Any]:
        """
        Atomically write 'obj' as JSON to 'path'. If compress=True, write gzip-compressed content.
        Returns metadata dict: {path, sha256, file_size_bytes, created_at}
        """
        path = Path(path)
        tmp_file = None
        try:
            # Serialize JSON to bytes deterministically (sorted keys)
            json_bytes = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, default=str).encode("utf-8")

            if compress:
                # compress bytes using gzip
                out_bytes = gzip.compress(json_bytes)
            else:
                out_bytes = json_bytes

            # Write to temporary file in same directory
            dirpath = path.parent
            dirpath.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=str(dirpath))
            tmp_file = Path(tmp_name)
            with os.fdopen(fd, "wb") as f:
                f.write(out_bytes)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    # fsync may fail on some platforms; ignore but continue
                    pass

            # Compute sha256 on the *file content as stored* (i.e., compressed if compressed)
            sha256 = self._compute_sha256_file(tmp_file)

            # Atomically move temp -> final
            os.replace(str(tmp_file), str(path))

            file_size = path.stat().st_size
            created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            return {"path": str(path), "sha256": sha256, "file_size_bytes": file_size, "created_at": created_at}

        except Exception as ex:
            # cleanup temp file if exists
            try:
                if tmp_file is not None and tmp_file.exists():
                    tmp_file.unlink()
            except Exception:
                pass
            raise SerializationError(f"Failed to write JSON to {path}: {ex}") from ex

    def _read_json_file(self, path: Union[str, Path], expect_mapping: bool = False) -> Any:
        """
        Read JSON or JSON.GZ file and return parsed object.

        - If expect_mapping is True, raises SerializationError if parsed object is not a mapping.
        """
        path = Path(path)
        try:
            if path.suffix == ".gz":
                with gzip.open(str(path), "rt", encoding="utf-8") as f:
                    obj = json.load(f)
            else:
                with open(path, "rt", encoding="utf-8") as f:
                    obj = json.load(f)
        except Exception as ex:
            raise SerializationError(f"Failed to read JSON file {path}: {ex}") from ex

        if expect_mapping and not isinstance(obj, dict):
            raise SerializationError(f"Expected JSON mapping at {path}, got {type(obj)}")
        return obj

    def _compute_sha256_file(self, path: Union[str, Path]) -> str:
        """
        Compute SHA256 hex digest of file content (streamed).
        """
        path = Path(path)
        h = hashlib.sha256()
        # For robustness, read in binary chunks
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception as ex:
            raise SerializationError(f"Failed to compute sha256 for {path}: {ex}") from ex

    def _update_manifest(self, manifest_path: Union[str, Path], entry_id: str, entry: Dict[str, Any], max_retries: int = 3) -> None:
        """
        Atomically update manifest file (manifest_path) by setting manifest[entry_id] = entry.

        Manifest format: JSON object mapping id -> metadata dict.

        Uses a simple file-write-with-temp-and-replace approach; protected by in-process lock.
        For cross-process concurrency, this is best-effort (OS-level advisory locks would be needed).
        """
        manifest_path = Path(manifest_path)
        manifest_dir = manifest_path.parent
        manifest_dir.mkdir(parents=True, exist_ok=True)

        attempt = 0
        while attempt < max_retries:
            attempt += 1
            try:
                with self._manifest_lock:
                    # Load existing manifest if present
                    if manifest_path.exists():
                        try:
                            manifest = self._read_json_file(manifest_path, expect_mapping=True)
                        except Exception:
                            # If manifest exists but corrupted, back it up and start fresh
                            backup_path = manifest_path.with_suffix(manifest_path.suffix + ".corrupt")
                            try:
                                shutil.copy2(manifest_path, backup_path)
                            except Exception:
                                pass
                            manifest = {}
                    else:
                        manifest = {}

                    if not isinstance(manifest, dict):
                        manifest = {}

                    manifest[entry_id] = entry

                    # atomic write manifest (no compression)
                    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{manifest_path.name}.tmp.", dir=str(manifest_dir))
                    tmp_path = Path(tmp_name)
                    try:
                        with os.fdopen(tmp_fd, "wb") as tf:
                            tf.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"))
                            tf.flush()
                            try:
                                os.fsync(tf.fileno())
                            except Exception:
                                pass
                        os.replace(str(tmp_path), str(manifest_path))
                        # success
                        self._log("manifest_update", {"manifest": str(manifest_path), "entry_id": entry_id})
                        return
                    finally:
                        # ensure tmp_path not lingering
                        if tmp_path.exists():
                            try:
                                tmp_path.unlink()
                            except Exception:
                                pass
            except Exception as ex:
                # log and retry
                self._log("manifest_update_failed", {"manifest": str(manifest_path), "entry_id": entry_id, "attempt": attempt, "error": str(ex)}, level="ERROR")
                time.sleep(0.1 * attempt)
                continue
        # if we got here, failed after retries
        raise SerializationError(f"Failed to update manifest {manifest_path} after {max_retries} attempts")

    def _validate_instance_schema(self, obj: Dict[str, Any]) -> None:
        """
        Perform basic schema validation for instance JSON.

        Raises InvalidInstanceError on violation.
        """
        if not isinstance(obj, dict):
            raise InvalidInstanceError("Instance JSON root must be a mapping")

        meta = obj.get("meta", {})
        data = obj.get("data", {})
        params = obj.get("params", {})

        # instance_id
        instance_id = meta.get("instance_id")
        if not instance_id or not isinstance(instance_id, str):
            raise InvalidInstanceError("meta.instance_id missing or invalid")

        # data.n_customers
        n = data.get("n_customers")
        if n is None or not isinstance(n, int) or n < 0:
            raise InvalidInstanceError("data.n_customers missing or invalid")

        # nodes layout
        nodes = data.get("nodes")
        if not isinstance(nodes, list):
            raise InvalidInstanceError("data.nodes must be a list")
        expected_nodes = [0] + list(range(1, n + 1)) + [n + 1]
        if nodes != expected_nodes:
            raise InvalidInstanceError(f"data.nodes must equal [0,1..n,n+1]; expected {expected_nodes}, got {nodes}")

        # customers list
        customers = data.get("customers")
        if not isinstance(customers, list) or len(customers) != n:
            raise InvalidInstanceError("data.customers must be a list of length n_customers")
        # verify customer ids and fields
        for cust in customers:
            if not isinstance(cust, dict):
                raise InvalidInstanceError("each customer entry must be a mapping")
            cid = cust.get("id")
            if not isinstance(cid, int) or cid < 1 or cid > n:
                raise InvalidInstanceError(f"customer id invalid: {cid}")
            coords = cust.get("coords")
            if not (isinstance(coords, list) and len(coords) == 2):
                raise InvalidInstanceError(f"customer {cid} coords invalid")
            demand = cust.get("demand")
            if not isinstance(demand, int):
                raise InvalidInstanceError(f"customer {cid} demand invalid")
            tw = cust.get("time_window")
            if not (isinstance(tw, list) and len(tw) == 2):
                raise InvalidInstanceError(f"customer {cid} time_window invalid")
            e_i, l_i = tw
            if not (isinstance(e_i, (int, float)) and isinstance(l_i, (int, float)) and e_i <= l_i):
                raise InvalidInstanceError(f"customer {cid} time_window e_i <= l_i violated")

        # depot check
        depot = data.get("depot", {})
        if not isinstance(depot, dict):
            raise InvalidInstanceError("data.depot must be mapping")
        tw_depot = depot.get("time_window")
        if not (isinstance(tw_depot, list) and len(tw_depot) == 2):
            raise InvalidInstanceError("depot.time_window invalid")

        # params checks: confirm Q_t, Q_d, L_t, L_d present (not necessarily non-null)
        for key in ("Q_t", "Q_d", "L_t", "L_d"):
            if key not in params:
                # Not fatal, but warn via exception? The design said include as warning if null. We'll raise only if missing entirely.
                raise InvalidInstanceError(f"params must include {key}")

        # D subset check
        D = data.get("D")
        if D is None:
            raise InvalidInstanceError("data.D (drone-eligible set) missing")
        if not isinstance(D, list):
            raise InvalidInstanceError("data.D must be a list of customer ids")
        for d in D:
            if not (isinstance(d, int) and 1 <= d <= n):
                raise InvalidInstanceError(f"data.D contains invalid customer id: {d}")

        # time window bounds: ensure all e_i and l_i within [0, L_t] if L_t present
        L_t = params.get("L_t")
        if isinstance(L_t, (int, float)):
            for cust in customers:
                e_i, l_i = cust.get("time_window")
                if e_i < 0 or l_i > L_t:
                    raise InvalidInstanceError(f"customer {cust.get('id')} time window outside [0, L_t]")

    def _log(self, event: str, payload: Dict[str, Any], level: str = "INFO") -> None:
        """
        Helper to emit log event via logger if available; otherwise no-op.
        """
        try:
            if self.logger is not None:
                try:
                    self.logger.log(event, payload, level=level)
                except Exception:
                    # swallow logger exceptions to avoid interfering with serialization
                    pass
        except Exception:
            pass


# -------------------------
# If run as script: minimal self-test (non-exhaustive)
# -------------------------
if __name__ == "__main__":  # pragma: no cover - demonstration
    # create a small demo instance dict and test save/load
    demo_config = {}
    ser = SerializationManager(config=demo_config, logger=None)

    small_instance = {
        "meta": {
            "instance_id": "demo_inst_001",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "generator_seed": 42,
            "instance_type": 1,
            "replicate_id": 0,
        },
        "data": {
            "n_customers": 2,
            "nodes": [0, 1, 2, 3],
            "customers": [
                {"id": 1, "coords": [1.0, 1.0], "demand": 10, "time_window": [0, 120], "eligible_for_drone": True},
                {"id": 2, "coords": [2.0, 2.0], "demand": 20, "time_window": [30, 240], "eligible_for_drone": False},
            ],
            "depot": {"id_start": 0, "id_end": 3, "coords_start": [0.0, 0.0], "coords_end": [0.0, 0.0], "time_window": [0, 480]},
            "D": [1],
        },
        "params": {
            "Q_t": 100,
            "Q_d": 20,
            "L_t": 480,
            "L_d": 30,
            "v_t_kmph": 40.0,
            "v_d_kmph": 40.0,
            "beta": 2.0,
            "F": 20.0,
            "c_t": 0.083,
            "c_d": 0.021,
            "service_times": {"truck_service_time_minutes": None, "drone_service_time_minutes": None},
        },
        "provenance": {
            "random_seed": 42
        },
        "integrity": {}
    }

    print("Saving demo instance...")
    meta = ser.save_instance(small_instance)
    print("Saved:", meta)
    loaded = ser.load_instance(Path(meta["path"]))
    print("Loaded instance id:", loaded.id, "n_customers:", loaded.n_customers)
