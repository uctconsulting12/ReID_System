"""
app_server.py
=============
FastAPI WebSocket server for CrossCamReid.

Start:
    uvicorn app_server:app --host 0.0.0.0 --port 8000

Postman WebSocket — connect to:
    ws://localhost:8000/ws/reid/<any-client-id>

Send to start:
    {
      "action": "start_stream",
      "cameras": [
        {"cam_id": "cam1", "role": "MASTER", "rtsp_url": "rtsp://..."},
        {"cam_id": "cam2", "role": "SLAVE",  "rtsp_url": "rtsp://..."}
      ]
    }

Send to stop:
    {"action": "stop_stream"}
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR  = THIS_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crosscamreid.config import load_config
from crosscamreid.websocket.reid_handler import reid_websocket_handler
from crosscamreid.websocket.reid_runner import run_reid_detection

# ── Config ────────────────────────────────────────────────────────────────────

config           = load_config(str(THIS_DIR / "config" / "config.yaml"))
POSTGRES_ENABLED = config.database.postgres.enabled

MAX_SESSIONS       = 5
detection_executor = ThreadPoolExecutor(max_workers=MAX_SESSIONS)
storage_executor   = ThreadPoolExecutor(max_workers=MAX_SESSIONS)
sessions: dict     = {}

# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ──
    print("[Server] CrossCamReid API starting up...")
    print(f"[Config] ReID backend  : {config.runtime.reid_backend}")
    print(f"[Config] Qdrant mode   : {config.database.qdrant.mode}")
    print(f"[Config] Postgres save : {POSTGRES_ENABLED}")

    if POSTGRES_ENABLED:
        try:
            from crosscamreid.database.database import init_db
            init_db()                        # prints "[DB] PostgreSQL connected successfully ✓"
        except Exception as exc:
            print(f"[DB] PostgreSQL connection FAILED: {exc}")
            print("[DB] Continuing without DB — set postgres.enabled=false in config.yaml to suppress")

    print("[Server] Ready. Listening for WebSocket connections.")
    yield
    # ── shutdown ──
    detection_executor.shutdown(wait=False)
    storage_executor.shutdown(wait=False)
    print("[Server] Shutdown complete.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="CrossCamReid API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.websocket("/ws/reid/{client_id}")
async def websocket_reid(ws: WebSocket, client_id: str):
    await reid_websocket_handler(
        detection_executor=detection_executor,
        storage_executor=storage_executor,
        ws=ws,
        client_id=client_id,
        sessions=sessions,
        run_detection_fn=run_reid_detection,
        config=config,
        postgres_enabled=POSTGRES_ENABLED,
    )


@app.get("/health")
def health():
    return {
        "status":           "ok",
        "version":          "1.0",
        "max_sessions":     MAX_SESSIONS,
        "active_sessions":  len(sessions),
        "postgres_enabled": POSTGRES_ENABLED,
        "qdrant_mode":      config.database.qdrant.mode,
    }
