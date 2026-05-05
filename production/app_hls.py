"""
app_hls.py
==========
FastAPI HTTP server that runs the CrossCamReid pipeline and exposes per-camera
HLS playlists. Each session is one ``stream_id`` that owns N cameras (master +
slaves) sharing one ReID store.

Start:
    uvicorn app_hls:app --host 0.0.0.0 --port 9000

Endpoints (mirroring the ffmpeg_test app shape):
    POST /streams/start         -> spawn a new session
    POST /streams/stop/{id}     -> stop and tear down a session
    GET  /streams               -> list active sessions
    GET  /streams/{id}          -> session details
    GET  /health                -> liveness + counters
    GET  /hls/{id}/{cam}/...    -> static HLS playlists / segments
"""

from __future__ import annotations

import logging
import mimetypes
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crosscamreid.config import AppConfig, CameraConfig, load_config
from crosscamreid.server import HLSStreamSession

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app_hls")

# m3u8 mime type isn't always registered on Windows
mimetypes.add_type("application/vnd.apple.mpegurl", ".m3u8")
mimetypes.add_type("video/mp2t", ".ts")

# ── paths ─────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG_PATH = THIS_DIR / "config" / "config.yaml"
HLS_ROOT = THIS_DIR / "hls_out"
HLS_ROOT.mkdir(parents=True, exist_ok=True)

# ── session registry ──────────────────────────────────────────────────────────

_sessions: dict[str, HLSStreamSession] = {}
_sessions_lock = threading.Lock()

# ── pydantic models ───────────────────────────────────────────────────────────

class CameraOverride(BaseModel):
    id: str
    role: str = Field(..., description="MASTER or SLAVE")
    source: str

class StartRequest(BaseModel):
    config_path: str | None = Field(
        default=None,
        description="Path to a YAML config. Defaults to production/config/config.yaml",
    )
    cameras: list[CameraOverride] | None = Field(
        default=None,
        description="Optional override: replaces the cameras list from the YAML.",
    )
    use_nvenc: bool = Field(
        default=True,
        description="Reserved knob (matches ffmpeg_test API). Encoding falls "
                    "back to libx264 unless explicitly enabled in the writer.",
    )
    output_fps: int = Field(default=25, ge=1, le=60)


# ── helpers ───────────────────────────────────────────────────────────────────

def _apply_camera_overrides(cfg: AppConfig, overrides: list[CameraOverride]) -> AppConfig:
    if not overrides:
        return cfg
    new_cams: list[CameraConfig] = []
    master_count = 0
    for o in overrides:
        role = o.role.strip().upper()
        if role not in {"MASTER", "SLAVE"}:
            raise HTTPException(400, f"camera {o.id}: role must be MASTER or SLAVE")
        if role == "MASTER":
            master_count += 1
        new_cams.append(CameraConfig(camera_id=o.id.strip(), role=role, source=o.source.strip()))
    if master_count < 1:
        raise HTTPException(400, "at least one MASTER camera is required")
    cfg.cameras = new_cams
    return cfg


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("CrossCamReid HLS server starting. HLS root: %s", HLS_ROOT)
    logger.info("Default config: %s", DEFAULT_CONFIG_PATH)
    yield
    # Stop everything on shutdown
    with _sessions_lock:
        ids = list(_sessions.keys())
    for sid in ids:
        try:
            _sessions[sid].stop(remove_files=True)
        except Exception as exc:
            logger.warning("shutdown: failed to stop %s: %s", sid, exc)
    logger.info("CrossCamReid HLS server stopped.")


app = FastAPI(title="CrossCamReid HLS API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves /hls/<stream_id>/<cam_id>/stream.m3u8 + segments
app.mount("/hls", StaticFiles(directory=str(HLS_ROOT)), name="hls")


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    with _sessions_lock:
        return {
            "status": "ok",
            "version": "1.0",
            "active_sessions": len(_sessions),
            "session_ids": list(_sessions.keys()),
            "hls_root": str(HLS_ROOT),
        }


@app.post("/streams/start")
def start_stream(req: StartRequest):
    cfg_path = Path(req.config_path) if req.config_path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise HTTPException(400, f"config file not found: {cfg_path}")

    try:
        cfg = load_config(str(cfg_path))
    except Exception as exc:
        raise HTTPException(400, f"failed to load config: {exc}") from exc

    if req.cameras:
        cfg = _apply_camera_overrides(cfg, req.cameras)

    stream_id = str(uuid.uuid4())
    session = HLSStreamSession(
        stream_id=stream_id,
        config=cfg,
        hls_root=HLS_ROOT,
        use_nvenc=req.use_nvenc,
        output_fps=req.output_fps,
    )

    with _sessions_lock:
        _sessions[stream_id] = session

    try:
        session.start()
    except Exception as exc:
        with _sessions_lock:
            _sessions.pop(stream_id, None)
        raise HTTPException(500, f"failed to start session: {exc}") from exc

    logger.info("started session %s on thread %s with %d cameras",
                stream_id, session.thread_name, len(cfg.cameras))

    return {
        "stream_id": stream_id,
        "thread_name": session.thread_name,
        "hls_urls": session.hls_urls(),
        "cameras": [
            {"id": c.camera_id, "role": c.role, "source": c.source}
            for c in cfg.cameras
        ],
    }


@app.post("/streams/stop/{stream_id}")
def stop_stream(stream_id: str):
    with _sessions_lock:
        session = _sessions.pop(stream_id, None)
    if session is None:
        raise HTTPException(404, f"unknown stream_id: {stream_id}")
    try:
        session.stop(remove_files=True)
    except Exception as exc:
        raise HTTPException(500, f"stop failed: {exc}") from exc
    logger.info("stopped session %s", stream_id)
    return {"status": "stopped", "stream_id": stream_id}


@app.get("/streams")
def list_streams():
    with _sessions_lock:
        return {"sessions": [s.status() for s in _sessions.values()]}


@app.get("/streams/{stream_id}")
def get_stream(stream_id: str):
    with _sessions_lock:
        session = _sessions.get(stream_id)
    if session is None:
        raise HTTPException(404, f"unknown stream_id: {stream_id}")
    return session.status()
