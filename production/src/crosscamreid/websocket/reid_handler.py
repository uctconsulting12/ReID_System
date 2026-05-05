"""
reid_handler.py
===============
WebSocket lifecycle handler for CrossCamReid sessions.

Client sends JSON messages:

  Start stream:
    {
      "action": "start_stream",
      "cameras": [
        {"cam_id": "cam1", "role": "MASTER", "rtsp_url": "rtsp://..."},
        {"cam_id": "cam2", "role": "SLAVE",  "rtsp_url": "rtsp://..."}
      ]
    }

  Stop stream:
    {"action": "stop_stream"}

Server streams per-frame detections back to the client as JSON.
"""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("reid_handler")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_ch)

_REQUIRED_CAM_KEYS = {"cam_id", "role", "rtsp_url"}
_VALID_ROLES = {"MASTER", "SLAVE"}


async def reid_websocket_handler(
    detection_executor: ThreadPoolExecutor,
    storage_executor: ThreadPoolExecutor,
    ws: WebSocket,
    client_id: str,
    sessions: dict,
    run_detection_fn,
    config,
    postgres_enabled: bool = True,
) -> None:
    await ws.accept()
    loop = asyncio.get_running_loop()

    sessions[client_id] = {
        "ws": ws,
        "streaming": False,
        "future": None,
    }
    logger.info("[%s] WebSocket connected", client_id)

    try:
        while True:
            try:
                raw = await ws.receive_text()
                if not raw.strip():
                    continue
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("[%s] Invalid JSON received", client_id)
                continue
            except WebSocketDisconnect:
                logger.info("[%s] Client disconnected", client_id)
                break
            except Exception:
                logger.exception("[%s] receive error", client_id)
                continue

            action = data.get("action")

            # ── start_stream ──────────────────────────────────────────────────
            if action == "start_stream":
                cameras = data.get("cameras")
                if not cameras or not isinstance(cameras, list):
                    await ws.send_json({
                        "status": "error",
                        "message": "cameras list required: [{cam_id, role, rtsp_url}, ...]",
                    })
                    continue

                # Validate each camera entry
                error = None
                for cam in cameras:
                    missing = _REQUIRED_CAM_KEYS - set(cam.keys())
                    if missing:
                        error = f"Camera entry missing fields: {missing}"
                        break
                    if cam["role"].upper() not in _VALID_ROLES:
                        error = f"role must be MASTER or SLAVE, got: {cam['role']}"
                        break
                if error:
                    await ws.send_json({"status": "error", "message": error})
                    continue

                # Optional metadata for DB storage
                org_id = data.get("org_id")
                userid = data.get("userid")
                include_annotated_frame = bool(data.get("include_annotated_frame", False))
                frame_jpeg_quality = data.get("frame_jpeg_quality", 80)
                if org_id is not None:
                    try:
                        org_id = int(org_id)
                    except (TypeError, ValueError):
                        org_id = None
                if userid is not None:
                    try:
                        userid = int(userid)
                    except (TypeError, ValueError):
                        userid = None
                try:
                    frame_jpeg_quality = int(frame_jpeg_quality)
                except (TypeError, ValueError):
                    frame_jpeg_quality = 80
                frame_jpeg_quality = max(30, min(95, frame_jpeg_quality))

                # Stop any running session before starting a new one
                if sessions[client_id].get("streaming"):
                    sessions[client_id]["streaming"] = False
                    fut = sessions[client_id].get("future")
                    if fut:
                        fut.cancel()

                sessions[client_id]["streaming"] = True

                future = loop.run_in_executor(
                    detection_executor,
                    run_detection_fn,
                    client_id,
                    cameras,
                    config,
                    sessions,
                    loop,
                    storage_executor,
                    postgres_enabled,
                    org_id,
                    userid,
                    include_annotated_frame,
                    frame_jpeg_quality,
                )
                sessions[client_id]["future"] = future

                cam_ids = [c["cam_id"] for c in cameras]
                logger.info("[%s] Started %d camera(s): %s  org=%s user=%s",
                            client_id, len(cameras), cam_ids, org_id, userid)
                await ws.send_json({
                    "status": "ok",
                    "message": f"Stream started for {len(cameras)} camera(s)",
                    "cameras": cam_ids,
                    "include_annotated_frame": include_annotated_frame,
                    "frame_jpeg_quality": frame_jpeg_quality,
                })

            # ── stop_stream ───────────────────────────────────────────────────
            elif action == "stop_stream":
                sessions[client_id]["streaming"] = False
                fut = sessions[client_id].get("future")
                if fut:
                    fut.cancel()
                logger.info("[%s] Stream stopped by client", client_id)
                await ws.send_json({"status": "ok", "message": "Stream stopped"})

            else:
                await ws.send_json({
                    "status": "error",
                    "message": f"Unknown action '{action}'. Use start_stream or stop_stream.",
                })

    except Exception:
        logger.exception("[%s] Unexpected handler error", client_id)

    finally:
        sessions[client_id]["streaming"] = False
        fut = sessions[client_id].get("future")
        if fut:
            fut.cancel()
        sessions.pop(client_id, None)
        logger.info("[%s] Session cleaned up", client_id)
