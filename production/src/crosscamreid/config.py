from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CameraConfig:
    camera_id: str
    role: str
    source: str


@dataclass
class ModelConfig:
    pose_path: str
    reid_onnx_path: str
    reid_tensorrt_engine_path: str | None
    fastreid_root: str | None = None
    fastreid_config: str | None = None
    fastreid_weights: str | None = None
    fastreid_device: str = "cuda"


@dataclass
class CaptureConfig:
    buffer_size: int
    reconnect_initial_delay_sec: float
    reconnect_max_delay_sec: float
    max_read_failures: int


@dataclass
class GatingConfig:
    person_conf_thresh: float
    keypoint_conf_thresh: float
    match_thresh: float
    min_region_side: int
    region_pad_frac: float
    max_embeddings_per_sid: int


@dataclass
class EnrollmentConfig:
    qualify_frames: int
    enroll_frames: int
    early_lock_on_match: bool = False


@dataclass
class QdrantConfig:
    mode: str           # "local" | "cloud"
    local_path: str
    cloud_url: str
    cloud_api_key: str
    collection: str
    keep_db: bool


@dataclass
class PostgresConfig:
    enabled: bool


@dataclass
class DatabaseConfig:
    qdrant: QdrantConfig
    postgres: PostgresConfig


@dataclass
class RuntimeConfig:
    tracker: str
    reid_backend: str
    no_display: bool
    display_width: int
    log_json: bool
    embed_grayscale: bool = False
    roi_based_master: bool = False
    sid_persist_on_kp_loss: bool = False
    tid_recover_max_missing_frames: int = 2
    tid_recover_iou_threshold: float = 0.5


@dataclass
class AppConfig:
    cameras: list[CameraConfig]
    models: ModelConfig
    capture: CaptureConfig
    gating: GatingConfig
    enrollment: EnrollmentConfig
    database: DatabaseConfig
    runtime: RuntimeConfig


def _require(raw: dict[str, Any], key: str) -> Any:
    if key not in raw:
        raise ValueError(f"Missing required config key: {key}")
    return raw[key]


def _resolve_path(base_dir: Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _resolve_tracker_path(base_dir: Path, value: str) -> str:
    """Tracker may be either a bare ultralytics built-in name like
    ``"bytetrack.yaml"`` (passed through verbatim so ultralytics resolves it
    against its packaged trackers) or a path relative to the YAML file like
    ``"./bytetrack_consistent.yaml"`` (resolved to an absolute path here so
    the runner doesn't depend on cwd)."""
    if not value:
        return value
    if "/" in value or "\\" in value or value.startswith("."):
        resolved = _resolve_path(base_dir, value)
        return str(resolved) if resolved is not None else value
    return value


def load_config(config_path: str) -> AppConfig:
    cfg_path = Path(config_path).resolve()
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping.")

    base_dir = cfg_path.parent

    cameras = _parse_cameras(raw)
    models = _require(raw, "models")
    capture = _require(raw, "capture")
    gating = _require(raw, "gating")
    enrollment = _require(raw, "enrollment")
    database = _require(raw, "database")
    runtime = _require(raw, "runtime")

    app_cfg = AppConfig(
        cameras=cameras,
        models=ModelConfig(
            pose_path=str(_resolve_path(base_dir, str(_require(models, "pose_path")))),
            reid_onnx_path=str(_resolve_path(base_dir, str(_require(models, "reid_onnx_path")))),
            reid_tensorrt_engine_path=_resolve_path(
                base_dir, models.get("reid_tensorrt_engine_path")
            ),
            fastreid_root=_resolve_path(base_dir, models.get("fastreid_root")),
            fastreid_config=_resolve_path(base_dir, models.get("fastreid_config")),
            fastreid_weights=_resolve_path(base_dir, models.get("fastreid_weights")),
            fastreid_device=str(models.get("fastreid_device", "cuda")).strip(),
        ),
        capture=CaptureConfig(
            buffer_size=int(_require(capture, "buffer_size")),
            reconnect_initial_delay_sec=float(_require(capture, "reconnect_initial_delay_sec")),
            reconnect_max_delay_sec=float(_require(capture, "reconnect_max_delay_sec")),
            max_read_failures=int(_require(capture, "max_read_failures")),
        ),
        gating=GatingConfig(
            person_conf_thresh=float(_require(gating, "person_conf_thresh")),
            keypoint_conf_thresh=float(_require(gating, "keypoint_conf_thresh")),
            match_thresh=float(_require(gating, "match_thresh")),
            min_region_side=int(_require(gating, "min_region_side")),
            region_pad_frac=float(_require(gating, "region_pad_frac")),
            max_embeddings_per_sid=int(_require(gating, "max_embeddings_per_sid")),
        ),
        enrollment=EnrollmentConfig(
            qualify_frames=int(_require(enrollment, "qualify_frames")),
            enroll_frames=int(_require(enrollment, "enroll_frames")),
            early_lock_on_match=bool(enrollment.get("early_lock_on_match", False)),
        ),
        database=_parse_database(base_dir, database),
        runtime=RuntimeConfig(
            tracker=_resolve_tracker_path(base_dir, str(_require(runtime, "tracker"))),
            reid_backend=str(_require(runtime, "reid_backend")).lower().strip(),
            no_display=bool(_require(runtime, "no_display")),
            display_width=int(_require(runtime, "display_width")),
            log_json=bool(_require(runtime, "log_json")),
            embed_grayscale=bool(runtime.get("embed_grayscale", False)),
            roi_based_master=bool(runtime.get("roi_based_master", False)),
            sid_persist_on_kp_loss=bool(runtime.get("sid_persist_on_kp_loss", False)),
            tid_recover_max_missing_frames=int(runtime.get("tid_recover_max_missing_frames", 2)),
            tid_recover_iou_threshold=float(runtime.get("tid_recover_iou_threshold", 0.5)),
        ),
    )

    if app_cfg.runtime.reid_backend not in {"onnxruntime", "tensorrt", "fastreid"}:
        raise ValueError(
            "runtime.reid_backend must be one of: onnxruntime, tensorrt, fastreid"
        )

    if app_cfg.runtime.reid_backend == "fastreid":
        m = app_cfg.models
        if not (m.fastreid_root and m.fastreid_config and m.fastreid_weights):
            raise ValueError(
                "runtime.reid_backend=fastreid requires models.fastreid_root, "
                "models.fastreid_config and models.fastreid_weights"
            )

    if app_cfg.enrollment.enroll_frames < 1:
        raise ValueError("enrollment.enroll_frames must be >= 1")

    if app_cfg.enrollment.qualify_frames < 1:
        raise ValueError("enrollment.qualify_frames must be >= 1")

    if app_cfg.runtime.tid_recover_max_missing_frames < 0:
        raise ValueError("runtime.tid_recover_max_missing_frames must be >= 0")

    if not (0.0 <= app_cfg.runtime.tid_recover_iou_threshold <= 1.0):
        raise ValueError("runtime.tid_recover_iou_threshold must be between 0 and 1")

    return app_cfg


def _parse_database(base_dir: Path, raw: dict[str, Any]) -> DatabaseConfig:
    qdrant_raw = _require(raw, "qdrant")
    postgres_raw = raw.get("postgres", {})

    mode = str(qdrant_raw.get("mode", "local")).lower().strip()
    if mode not in {"local", "cloud"}:
        raise ValueError("database.qdrant.mode must be 'local' or 'cloud'")

    local_path = str(
        _resolve_path(base_dir, str(_require(qdrant_raw, "local_path")))
    )
    cloud_url = str(qdrant_raw.get("cloud_url", "") or "")
    cloud_api_key = str(
        qdrant_raw.get("cloud_api_key", "")
        or os.environ.get("QDRANT_API_KEY", "")
    )

    if mode == "cloud" and not cloud_url:
        raise ValueError(
            "database.qdrant.cloud_url is required when mode=cloud"
        )

    return DatabaseConfig(
        qdrant=QdrantConfig(
            mode=mode,
            local_path=local_path,
            cloud_url=cloud_url,
            cloud_api_key=cloud_api_key,
            collection=str(_require(qdrant_raw, "collection")),
            keep_db=bool(qdrant_raw.get("keep_db", False)),
        ),
        postgres=PostgresConfig(
            enabled=bool(postgres_raw.get("enabled", True)),
        ),
    )


def _parse_cameras(raw: dict[str, Any]) -> list[CameraConfig]:
    """
    Preferred schema:
      cameras:
        - id: "cam1"
          role: "master"
          source: "rtsp://..."

    Backward-compatible schema:
      sources:
        master: "..."
        slave: "..."
    """
    camera_list = raw.get("cameras")
    if camera_list is not None:
        if not isinstance(camera_list, list) or not camera_list:
            raise ValueError("cameras must be a non-empty list.")

        parsed: list[CameraConfig] = []
        master_count = 0
        for idx, camera in enumerate(camera_list):
            if not isinstance(camera, dict):
                raise ValueError(f"cameras[{idx}] must be a mapping.")
            camera_id = str(_require(camera, "id")).strip()
            role = str(_require(camera, "role")).strip().upper()
            source = str(_require(camera, "source")).strip()
            if not camera_id:
                raise ValueError(f"cameras[{idx}].id cannot be empty.")
            if role not in {"MASTER", "SLAVE"}:
                raise ValueError(f"cameras[{idx}].role must be MASTER or SLAVE.")
            if role == "MASTER":
                master_count += 1
            parsed.append(CameraConfig(camera_id=camera_id, role=role, source=source))

        if master_count < 1:
            raise ValueError("At least one MASTER camera is required.")
        return parsed

    if "sources" in raw:
        legacy_sources = _require(raw, "sources")
        master_src = str(_require(legacy_sources, "master"))
        slave_src = str(_require(legacy_sources, "slave"))
        return [
            CameraConfig(camera_id="cam1", role="MASTER", source=master_src),
            CameraConfig(camera_id="cam2", role="SLAVE", source=slave_src),
        ]

    raise ValueError("Config must include 'cameras' or legacy 'sources'.")
