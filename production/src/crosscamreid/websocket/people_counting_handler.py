"""
people_counting_handler.py
==========================
WebSocket lifecycle handler for the people-counting API.

Endpoint:
    ws://<host>:8002/ws/people_counting/{client_id}?token=<JWT>

Inbound JSON:
    { "action": "start_stream", ... }
    { "action": "stop_stream",  "camera_id": <int>, "org_id": ..., "user_id": ... }
    { "action": "stop_all",     "org_id": ..., "user_id": ... }
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect, status

from ..counting.auth import AuthError, validate_token
from ..counting.org_registry import OrgRegistry
from ..config import AppConfig
from .people_counting_runner import (
    CameraSpec,
    PeopleCountingSession,
)

logger = logging.getLogger("people_counting.handler")


def _coerce_int(v: Any, default: int | None = None) -> int | None:
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _coerce_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _parse_cameras(raw: Any) -> tuple[list[CameraSpec], list[str]]:
    """Validate and convert the inbound camera list. Returns (specs, errors)."""
    errors: list[str] = []
    if not isinstance(raw, list) or not raw:
        return [], ["'cameras' must be a non-empty list"]
    if len(raw) > PeopleCountingSession.MAX_CAMERAS:
        errors.append(
            f"max {PeopleCountingSession.MAX_CAMERAS} cameras per request, "
            f"received {len(raw)} (extras will be rejected)"
        )

    specs: list[CameraSpec] = []
    for i, c in enumerate(raw):
        if not isinstance(c, dict):
            errors.append(f"cameras[{i}] must be an object")
            continue
        cam_id = _coerce_int(c.get("camera_id"))
        if cam_id is None:
            errors.append(f"cameras[{i}].camera_id must be an integer")
            continue
        url = c.get("stream_url")
        if not isinstance(url, str) or not url.strip():
            errors.append(f"cameras[{i}].stream_url is required (string)")
            continue
        region = c.get("region")
        if region is not None and not isinstance(region, str):
            errors.append(f"cameras[{i}].region must be a string if provided")
            region = None
        # Optional role: defaults to MASTER (legacy behaviour) when absent
        # so old clients without master/slave UI keep enrolling new SIDs.
        role_raw = c.get("role", "MASTER")
        role = str(role_raw).strip().upper() if role_raw is not None else "MASTER"
        if role not in {"MASTER", "SLAVE"}:
            errors.append(
                f"cameras[{i}].role must be 'MASTER' or 'SLAVE' if provided"
            )
            role = "MASTER"
        specs.append(CameraSpec(
            camera_id=cam_id,
            stream_url=url.strip(),
            region=(region.strip() if isinstance(region, str) and region.strip() else None),
            role=role,
        ))

    return specs, errors


async def people_counting_websocket_handler(
    ws: WebSocket,
    *,
    client_id: str,
    token: str | None,
    config: AppConfig,
    org_registry: OrgRegistry,
    sessions: dict,
) -> None:
    # ── auth ────────────────────────────────────────────────────────────
    try:
        claims = validate_token(token)
    except AuthError as exc:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason=str(exc))
        logger.info("[%s] auth rejected: %s", client_id, exc)
        return

    await ws.accept()
    loop = asyncio.get_running_loop()
    logger.info("[%s] connected (claims=%s)", client_id, {k: claims.get(k) for k in ("sub", "org_id", "user_id") if k in claims})

    if client_id in sessions:
        # Reject duplicate client_ids; ws clients should reconnect with a fresh id.
        await ws.send_json({
            "event": "error",
            "status": "failed",
            "error_message": f"client_id '{client_id}' already has an active session",
        })
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    session: PeopleCountingSession | None = None
    sessions[client_id] = None

    try:
        while True:
            try:
                raw = await ws.receive_text()
            except WebSocketDisconnect:
                logger.info("[%s] client disconnected", client_id)
                break
            except Exception:
                logger.exception("[%s] receive error", client_id)
                break

            if not raw or not raw.strip():
                continue

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({
                    "event": "error",
                    "status": "failed",
                    "error_message": "invalid JSON",
                })
                continue

            action = data.get("action")

            # ── start_stream ─────────────────────────────────────────────
            if action == "start_stream":
                org_id = _coerce_int(data.get("org_id"))
                user_id = _coerce_int(data.get("user_id"))
                if org_id is None or user_id is None:
                    await ws.send_json({
                        "event": "error",
                        "status": "failed",
                        "error_message": "org_id and user_id are required (int)",
                    })
                    continue

                # Cross-org isolation: if a session is already attached to a
                # different org_id, refuse the request.
                if session is not None and session.org_id != org_id:
                    await ws.send_json({
                        "event": "error",
                        "status": "failed",
                        "error_message": (
                            "this connection is bound to org_id="
                            f"{session.org_id}; reconnect to switch org"
                        ),
                    })
                    continue

                threshold = _coerce_int(data.get("threshold"), 0) or 0
                alert_rate = _coerce_float(data.get("alert_rate"), 80.0)
                fps = _coerce_int(data.get("fps"), 0) or 0
                frame_skip = _coerce_int(data.get("frame_skip"), 0) or 0
                exit_timeout = _coerce_float(data.get("exit_timeout_sec"), 2.0)
                include_frame = bool(data.get("include_annotated_frame", True))
                jpeg_quality = _coerce_int(data.get("frame_jpeg_quality"), 70) or 70
                send_interval = _coerce_int(data.get("frame_send_interval"), 20) or 20

                cam_specs, validation_errors = _parse_cameras(data.get("cameras"))
                for err in validation_errors:
                    await ws.send_json({
                        "event": "error",
                        "status": "failed",
                        "error_message": err,
                    })
                if not cam_specs:
                    continue

                if session is None:
                    session = PeopleCountingSession(
                        client_id=client_id,
                        org_id=org_id,
                        user_id=user_id,
                        config=config,
                        org_registry=org_registry,
                        ws=ws,
                        loop=loop,
                        threshold=threshold,
                        alert_rate=alert_rate,
                        target_fps=fps,
                        frame_skip=frame_skip,
                        exit_timeout_sec=exit_timeout,
                        include_annotated_frame=include_frame,
                        frame_jpeg_quality=jpeg_quality,
                        frame_send_interval=send_interval,
                    )
                    sessions[client_id] = session
                else:
                    # Re-tune parameters on subsequent start_stream calls.
                    session.threshold = threshold
                    session.alert_rate = alert_rate
                    session.target_fps = fps
                    session.frame_skip = frame_skip
                    session.exit_timeout_sec = exit_timeout
                    session.include_annotated_frame = include_frame
                    session.frame_jpeg_quality = max(30, min(95, jpeg_quality))
                    session.frame_send_interval = max(1, send_interval)

                started, errors = session.start_cameras(cam_specs)
                for cam_id, err in errors:
                    await ws.send_json({
                        "event": "error",
                        "camera_id": cam_id,
                        "status": "failed",
                        "error_message": err,
                    })
                logger.info(
                    "[%s] start_stream: started=%s errors=%s",
                    client_id, started, [c for c, _ in errors],
                )

            # ── stop_stream ──────────────────────────────────────────────
            elif action == "stop_stream":
                cam_id = _coerce_int(data.get("camera_id"))
                if cam_id is None:
                    await ws.send_json({
                        "event": "error",
                        "status": "failed",
                        "error_message": "camera_id is required (int)",
                    })
                    continue
                if session is None:
                    await ws.send_json({
                        "event": "error",
                        "camera_id": cam_id,
                        "status": "failed",
                        "error_message": "no active session",
                    })
                    continue

                ok = session.stop_camera(cam_id)
                await ws.send_json({
                    "event": "stream_stopped",
                    "camera_id": cam_id,
                    "status": "ok" if ok else "failed",
                    "message": (
                        "Camera stopped" if ok else f"camera_id {cam_id} not active"
                    ),
                })

            # ── stop_all ─────────────────────────────────────────────────
            elif action == "stop_all":
                if session is None:
                    await ws.send_json({
                        "event": "all_stopped",
                        "status": "ok",
                        "message": "no active session",
                        "stopped": 0,
                    })
                    continue
                stopped = session.stop_all()
                await ws.send_json({
                    "event": "all_stopped",
                    "status": "ok",
                    "stopped": stopped,
                })

            # ── unknown ──────────────────────────────────────────────────
            else:
                await ws.send_json({
                    "event": "error",
                    "status": "failed",
                    "error_message": (
                        f"unknown action '{action}'. "
                        "Use start_stream | stop_stream | stop_all."
                    ),
                })

    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                logger.exception("[%s] session close failed", client_id)
        sessions.pop(client_id, None)
        logger.info("[%s] cleaned up", client_id)
