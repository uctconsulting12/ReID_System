"""
reid_runner.py
==============
Detection runner for WebSocket-driven CrossCamReid sessions.

Runs the full production pipeline (YOLO + keypoint gating + ReID),
streams per-frame JSON to the WebSocket client, and saves every frame
to PostgreSQL independently — DB save is never skipped due to WS errors.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import cv2
import numpy as np
from ultralytics import YOLO

from ..capture import RTSPCapture
from ..config import AppConfig, CameraConfig
from ..overlay import draw_overlay
from ..processor import process_master, process_slave
from ..reid.factory import create_reid_backend
from ..state import TIDStateManager
from ..store import SIDStore

logger = logging.getLogger("reid_runner")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_ch)


def _json_default(obj):
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def _phase(rec: dict) -> str:
    if rec["sid"] != "UNKNOWN" and rec["enroll_left"] == 0:
        return "LOCK"
    if rec["enroll_left"] > 0:
        return "ENROLL"
    if rec["qualified"] > 0:
        return "QUALIFY"
    return "NEW"


def _encode_jpeg_base64(frame: np.ndarray, quality: int) -> str | None:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _run_frame(pose, frame, config, processor, reid_backend, store, states):
    """Run YOLO tracking + ReID on a single frame. Returns list of records."""
    result = pose.track(
        frame,
        persist=True,
        classes=[0],
        conf=config.gating.person_conf_thresh,
        tracker=config.runtime.tracker,
        verbose=False,
    )[0]

    records: list[dict] = []
    alive: set[int] = set()
    alive_boxes: dict[int, tuple[float, float, float, float]] = {}

    if (
        result.boxes is not None
        and len(result.boxes) > 0
        and result.keypoints is not None
        and result.boxes.id is not None
    ):
        boxes   = result.boxes.xyxy.cpu().numpy()
        tids    = result.boxes.id.cpu().numpy().astype(int)
        stable_tids = states.remap_tids(tids, boxes)
        kp_xy   = result.keypoints.xy.cpu().numpy()
        kp_conf = result.keypoints.conf.cpu().numpy()

        for i in range(len(boxes)):
            bbox = [float(c) for c in boxes[i]]
            rec  = processor(
                frame, bbox, kp_xy[i], kp_conf[i], int(stable_tids[i]),
                reid_backend, store, states, config,
            )
            records.append(rec)
            stable_tid = int(stable_tids[i])
            alive.add(stable_tid)
            alive_boxes[stable_tid] = (bbox[0], bbox[1], bbox[2], bbox[3])

    states.forget(alive, alive_boxes)
    return records


# ── Storage worker ────────────────────────────────────────────────────────────

def _storage_worker(store_queue: queue.Queue, postgres_enabled: bool) -> None:
    """Dedicated thread: drains the queue and writes every frame to PostgreSQL."""
    if not postgres_enabled:
        logger.info("Storage worker: postgres disabled, draining queue silently")
        while True:
            if store_queue.get() is None:
                break
        return

    try:
        from ..database.reid_query import ensure_table, insert_reid_stream_frame
        ensure_table()
        logger.info("Storage worker: DB ready, processing queue")
    except Exception as exc:
        logger.error("Storage worker: DB init failed — %s. Frames will NOT be saved.", exc)
        while True:
            if store_queue.get() is None:
                break
        return

    while True:
        item = store_queue.get()
        if item is None:
            break
        try:
            insert_reid_stream_frame(item)
        except Exception as exc:
            logger.error("Storage error stream frame %s: %s", item.get("stream_frame_id"), exc)

    logger.info("Storage worker: exiting")


# ── Detection runner ──────────────────────────────────────────────────────────

def run_reid_detection(
    client_id: str,
    cameras: list[dict],
    config: AppConfig,
    sessions: dict,
    loop: asyncio.AbstractEventLoop,
    storage_executor: ThreadPoolExecutor,
    postgres_enabled: bool = True,
    org_id: int | None = None,
    userid: int | None = None,
    include_annotated_frame: bool = False,
    frame_jpeg_quality: int = 80,
) -> None:
    """
    Runs the full CrossCamReid pipeline for a WebSocket session.
    DB save and WS send are fully independent — a WS error never skips the DB.

    cameras: [{"cam_id": "cam1", "role": "MASTER", "rtsp_url": "rtsp://..."}, ...]
    """
    def _is_streaming() -> bool:
        try:
            return sessions[client_id]["streaming"]
        except KeyError:
            return False

    # ── Build CameraConfig from WebSocket message ─────────────────────────────
    cam_configs: list[CameraConfig] = [
        CameraConfig(
            camera_id=c["cam_id"],
            role=c["role"].upper(),
            source=c["rtsp_url"],
        )
        for c in cameras
    ]

    # ── Load models ───────────────────────────────────────────────────────────
    logger.info("[%s] Loading YOLO pose model...", client_id)
    pose_models: dict[str, YOLO] = {
        cam.camera_id: YOLO(config.models.pose_path) for cam in cam_configs
    }

    reid_backend = create_reid_backend(
        backend=config.runtime.reid_backend,
        onnx_path=config.models.reid_onnx_path,
        tensorrt_engine_path=config.models.reid_tensorrt_engine_path,
        fastreid_root=config.models.fastreid_root,
        fastreid_config=config.models.fastreid_config,
        fastreid_weights=config.models.fastreid_weights,
        fastreid_device=config.models.fastreid_device,
        use_grayscale=config.runtime.embed_grayscale,
    )

    q = config.database.qdrant
    store = SIDStore(
        collection=q.collection,
        dim=reid_backend.dim,
        fresh=not q.keep_db,
        max_embeddings_per_sid=config.gating.max_embeddings_per_sid,
        db_path=q.local_path if q.mode == "local" else None,
        cloud_url=q.cloud_url if q.mode == "cloud" else None,
        cloud_api_key=q.cloud_api_key if q.mode == "cloud" else None,
    )

    state_managers: dict[str, TIDStateManager] = {
        cam.camera_id: TIDStateManager(
            max_missing_frames=config.runtime.tid_recover_max_missing_frames,
            recover_iou_threshold=config.runtime.tid_recover_iou_threshold,
        )
        for cam in cam_configs
    }
    captures: dict[str, RTSPCapture] = {
        cam.camera_id: RTSPCapture(
            cam.source, f"{cam.camera_id}-{cam.role[0]}", config.capture
        ).start()
        for cam in cam_configs
    }

    # ── Storage worker thread ─────────────────────────────────────────────────
    store_queue: queue.Queue = queue.Queue(maxsize=1000)
    storage_executor.submit(_storage_worker, store_queue, postgres_enabled)

    fps_count, fps_start, fps = 0, time.time(), 0.0
    frame_counters: dict[str, int] = {cam.camera_id: 0 for cam in cam_configs}
    requested_camera_ids = [cam.camera_id for cam in cam_configs]

    logger.info("[%s] Pipeline running. Cameras: %s  org=%s user=%s",
                client_id, [c.camera_id for c in cam_configs], org_id, userid)

    try:
        while _is_streaming():
            frames = {
                cam.camera_id: captures[cam.camera_id].get_frame()
                for cam in cam_configs
            }
            if all(f is None for f in frames.values()):
                time.sleep(0.05)
                continue

            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                fps_count, fps_start = 0, time.time()

            per_cycle_camera_payloads: dict[str, dict] = {}

            for cam in cam_configs:
                frame = frames.get(cam.camera_id)
                if frame is None:
                    per_cycle_camera_payloads[cam.camera_id] = {
                        "frame_available": False,
                        "frame_id": None,
                        "timestamp": None,
                        "frame_count": frame_counters[cam.camera_id],
                        "fps": round(fps, 2),
                        "total_detected": 0,
                        "detections": [],
                    }
                    continue

                frame_counters[cam.camera_id] += 1
                processor = process_master if cam.role == "MASTER" else process_slave

                records = _run_frame(
                    pose_models[cam.camera_id], frame, config,
                    processor, reid_backend, store,
                    state_managers[cam.camera_id],
                )

                frame_id  = f"FR_{cam.camera_id}_{int(time.time() * 1000)}"
                timestamp = datetime.now(timezone.utc).isoformat()

                detections_out = [
                    {
                        "track_id":         rec["tid"],
                        "sid":              rec["sid"],
                        "phase":            _phase(rec),
                        "bbox":             rec["bbox"],
                        "similarity_score": rec["similarity_score"],
                        "keypoint_valid":   rec["keypoint_valid"],
                    }
                    for rec in records
                ]

                payload = {
                    "cam_id":         cam.camera_id,
                    "frame_id":       frame_id,
                    "timestamp":      timestamp,
                    "frame_count":    frame_counters[cam.camera_id],
                    "fps":            round(fps, 2),
                    "total_detected": len(detections_out),
                    "detections":     detections_out,
                    "org_id":         org_id,
                    "userid":         userid,
                }

                per_cycle_camera_payloads[cam.camera_id] = {
                    "frame_available": True,
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "frame_count": frame_counters[cam.camera_id],
                    "fps": round(fps, 2),
                    "total_detected": len(detections_out),
                    "detections": detections_out,
                }

                # ── 2. Stream to WebSocket ────────────────────────────────────
                ws = sessions.get(client_id, {}).get("ws")
                if ws:
                    try:
                        ws_payload = payload
                        if include_annotated_frame:
                            annotated = draw_overlay(
                                frame.copy(),
                                records,
                                fps,
                                store.total_sids(),
                                cam_label=cam.camera_id,
                                mode_label=cam.role,
                                config=config,
                            )
                            frame_b64 = _encode_jpeg_base64(annotated, frame_jpeg_quality)
                            ws_payload = dict(payload)
                            ws_payload["annotated_frame_b64"] = frame_b64
                            ws_payload["annotated_frame_format"] = "jpg"
                        asyncio.run_coroutine_threadsafe(
                            ws.send_text(json.dumps(ws_payload, default=_json_default)),
                            loop,
                        )
                    except Exception as exc:
                        logger.error("[%s][%s] WS send error: %s",
                                     client_id, cam.camera_id, exc)
                        # DB save already done — do NOT break, keep processing

            # ── 1. Save one aggregated row for all requested cameras ──────────
            stream_frame_id = f"SFR_{client_id}_{int(time.time() * 1000)}"
            cycle_timestamp = datetime.now(timezone.utc).isoformat()
            total_detected_cycle = 0
            for cam_id in requested_camera_ids:
                cam_payload = per_cycle_camera_payloads.get(cam_id)
                if cam_payload is None:
                    cam_payload = {
                        "frame_available": False,
                        "frame_id": None,
                        "timestamp": None,
                        "frame_count": frame_counters.get(cam_id, 0),
                        "fps": round(fps, 2),
                        "total_detected": 0,
                        "detections": [],
                    }
                    per_cycle_camera_payloads[cam_id] = cam_payload
                total_detected_cycle += int(cam_payload.get("total_detected", 0) or 0)

            db_row_payload = {
                "stream_frame_id": stream_frame_id,
                "client_id": client_id,
                "timestamp": cycle_timestamp,
                "fps": round(fps, 2),
                "requested_cameras": requested_camera_ids,
                "cameras": per_cycle_camera_payloads,
                "total_detected": total_detected_cycle,
                "org_id": org_id,
                "userid": userid,
                "status": "ok",
            }
            try:
                store_queue.put(db_row_payload, timeout=0.05)
            except queue.Full:
                logger.warning("[%s] Storage queue full — aggregated stream frame dropped: %s",
                               client_id, stream_frame_id)

    finally:
        for cap in captures.values():
            cap.stop()
        store_queue.put(None)   # sentinel: tell storage worker to exit
        logger.info("[%s] Pipeline stopped. Total SIDs enrolled: %d",
                    client_id, store.total_sids())
