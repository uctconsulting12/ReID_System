"""
reid_query.py
=============
ORM-level upsert for CrossCamReid detection results.
SQLAlchemy + psycopg2; thread-safe via per-call engine.begin() connections.
"""

import logging
from datetime import datetime

from sqlalchemy.dialects.postgresql import insert as pg_insert

from .database import get_engine, init_db
from .models import ReidDetection, ReidStreamFrame

logger = logging.getLogger("reid_query")
logger.setLevel(logging.INFO)


def ensure_table() -> None:
    """Create tables if missing and verify DB connection. Called once per worker."""
    init_db()


def insert_reid_detection(data: dict) -> None:
    """
    Upsert one frame's detection results into reid_detections.

    Expected keys in data:
        cam_id        — str   camera identifier
        frame_id      — str   unique per-frame ID (FR_cam1_<ms>)
        timestamp     — str (ISO) or datetime
        frame_count   — int   frame counter for this camera
        fps           — float
        total_detected — int  number of persons detected
        detections    — list  [{track_id, sid, phase, bbox,
                                similarity_score, keypoint_valid}, ...]
        org_id        — int | None
        userid        — int | None
    """
    ts = data.get("timestamp")
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)

    detections = data.get("detections") or []

    values = {
        "camid":             data.get("cam_id"),
        "org_id":            data.get("org_id"),
        "userid":            data.get("userid"),
        "frame_id":          data["frame_id"],
        "time_stamp":        ts,
        "frame_count":       data.get("frame_count"),
        "fps":               data.get("fps"),
        "total_detected":    data.get("total_detected", len(detections)),
        "track_ids":         [d["track_id"]          for d in detections],
        "sids":              [d["sid"]                for d in detections],
        "phases":            [d["phase"]              for d in detections],
        "bounding_boxes":    [d["bbox"]               for d in detections],
        "similarity_scores": [d.get("similarity_score") for d in detections],
        "keypoint_valid":    [d.get("keypoint_valid") for d in detections],
        "locked_count":      sum(1 for d in detections if d.get("phase") == "LOCK"),
        "qualifying_count":  sum(1 for d in detections if d.get("phase") in ("QUALIFY", "ENROLL")),
    }

    engine = get_engine()
    with engine.begin() as conn:
        stmt = (
            pg_insert(ReidDetection)
            .values(**values)
            .on_conflict_do_update(
                index_elements=["frame_id"],
                set_={k: v for k, v in values.items() if k != "frame_id"},
            )
        )
        conn.execute(stmt)

    logger.debug("Saved frame %s  cam=%s  detected=%d",
                 data["frame_id"], data.get("cam_id"), len(detections))


def insert_reid_stream_frame(data: dict) -> None:
    """
    Upsert one multi-camera stream cycle into reid_stream_frames.

    Expected keys:
      stream_frame_id: str (unique row key per cycle)
      client_id: str
      timestamp: str|datetime
      fps: float
      requested_cameras: list[str]
      cameras: dict[str, dict]   per-camera payload
      total_detected: int
      org_id: int|None
      userid: int|None
      status: str|None
    """
    ts = data.get("timestamp")
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)

    camera_payloads = data.get("cameras") or {}
    requested_cameras = data.get("requested_cameras") or list(camera_payloads.keys())
    total_detected = data.get("total_detected")
    if total_detected is None:
        total_detected = 0
        for payload in camera_payloads.values():
            if isinstance(payload, dict):
                total_detected += int(payload.get("total_detected", 0) or 0)

    values = {
        "stream_frame_id": data["stream_frame_id"],
        "client_id": data.get("client_id"),
        "org_id": data.get("org_id"),
        "userid": data.get("userid"),
        "requested_cameras": requested_cameras,
        "camera_payloads": camera_payloads,
        "time_stamp": ts,
        "fps": data.get("fps"),
        "camera_count": len(requested_cameras),
        "total_detected": int(total_detected),
        "status": data.get("status"),
    }

    engine = get_engine()
    with engine.begin() as conn:
        stmt = (
            pg_insert(ReidStreamFrame)
            .values(**values)
            .on_conflict_do_update(
                index_elements=["stream_frame_id"],
                set_={k: v for k, v in values.items() if k != "stream_frame_id"},
            )
        )
        conn.execute(stmt)

    logger.debug(
        "Saved stream frame %s client=%s cams=%d detected=%d",
        data["stream_frame_id"],
        data.get("client_id"),
        len(requested_cameras),
        int(total_detected),
    )
