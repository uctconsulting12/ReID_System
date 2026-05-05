# CrossCamReid — production runners

This folder ships **three** ways to run the same multi-camera ReID pipeline
(YOLOv8 pose + keypoint gating + ReID embeddings + Qdrant SID store):

| Runner | File | Purpose |
|---|---|---|
| **CLI** | `app.py` | Local desktop run with cv2 windows. Good for development. |
| **WebSocket** | `app_server.py` | One client per WebSocket; per-frame JSON detections streamed back. |
| **HLS HTTP** | `app_hls.py` | Multi-session HTTP API that exposes annotated **HLS playlists** per camera (browser/VLC playback). Mirrors the `ffmpeg_test` API shape. |

The three runners share the same code under `src/crosscamreid/` — switching
between them does not change pipeline behavior.

---

## Prerequisites

- **Python 3.10+**
- **ffmpeg** on `PATH` (only for `app_hls.py`)
- Optional: **NVIDIA GPU + CUDA** (faster YOLO; required for the
  `tensorrt`/`fastreid` backends configured in `config/config.yaml`)

Install Python deps from the repo root:

```bash
pip install -r ../requirements.txt
```

The repo `requirements.txt` already includes `fastapi` and `uvicorn[standard]`,
which both `app_server.py` and `app_hls.py` need.

---

## Configuration

All three runners load the same YAML schema (`production/config/config.yaml`).
Key sections:

- `cameras` — list of `{id, role: MASTER|SLAVE, source}`. Source can be RTSP,
  HTTP, a local file path, or a webcam index.
- `models.pose_path` / `models.reid_*` — model weights (paths are resolved
  relative to the YAML file).
- `gating` / `enrollment` — ReID thresholds and enrollment voting parameters.
- `database.qdrant` — local or cloud vector store (`keep_db: true` preserves
  the SID gallery between runs).
- `runtime.reid_backend` — `onnxruntime`, `tensorrt`, or `fastreid`.
- `runtime.roi_based_master` — interactive ROI selection on the master frame.
  **Disabled automatically when running under `app_hls.py`** because there is
  no GUI in a server context.
- `runtime.sid_persist_on_kp_loss` — when a tracker ID is already locked to an
  SID, keep returning that SID across transient keypoint dropouts (occlusion).
  Default `false`; enabled only in `localtest/config/local_video.yaml`.

---

## 1. CLI runner — `app.py`

```bash
cd production
python app.py --config config/config.yaml
```

Opens an OpenCV window grid showing every camera with detection overlays.
Press `q` to quit.

---

## 2. WebSocket runner — `app_server.py`

```bash
cd production
uvicorn app_server:app --host 0.0.0.0 --port 8000
```

Connect from a WebSocket client (Postman, `wscat`, etc.):

```
ws://localhost:8000/ws/reid/<any-client-id>
```

Send to start:

```json
{
  "action": "start_stream",
  "cameras": [
    {"cam_id": "cam1", "role": "MASTER", "rtsp_url": "rtsp://..."},
    {"cam_id": "cam2", "role": "SLAVE",  "rtsp_url": "rtsp://..."}
  ]
}
```

Send to stop: `{"action": "stop_stream"}`.

You'll receive one JSON message per processed frame containing per-track SID,
phase (`NEW` / `QUALIFY` / `ENROLL` / `LOCK`), bbox, and similarity score.

`GET /health` returns server status.

---

## 3. HLS HTTP runner — `app_hls.py` (recommended for browser playback)

```bash
cd production
uvicorn app_hls:app --host 0.0.0.0 --port 9000
```

The pipeline runs in a **dedicated worker thread per session**; each camera
in a session has its own `RTSPCapture` reader thread (existing) and its own
`ffmpeg` subprocess that turns annotated frames into an HLS playlist.

### Architecture

```
POST /streams/start
        │
        ▼
HLSStreamSession  (1 worker thread)
        │
        ├── RTSPCapture(cam1)    ── frames ──┐
        ├── RTSPCapture(cam2)    ── frames ──┤
        │                                    │
        │   YOLO + ReID + draw_overlay ◄─────┘
        │                                    │
        ├── FFmpegHLSWriter(cam1) ── stdin ──► ffmpeg ─► hls_out/<id>/cam1/stream.m3u8
        └── FFmpegHLSWriter(cam2) ── stdin ──► ffmpeg ─► hls_out/<id>/cam2/stream.m3u8
```

Multiple sessions can run in parallel; each one gets a fresh `stream_id`
(UUID), its own folder under `hls_out/`, and its own ffmpeg processes. All
cameras inside a single session share one `SIDStore`, so cross-camera ReID
still works.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/streams/start` | Start a new session. |
| `POST` | `/streams/stop/{stream_id}` | Stop a session and remove its HLS folder. |
| `GET` | `/streams` | List all active sessions. |
| `GET` | `/streams/{stream_id}` | Status of one session (fps, frames written, hls urls). |
| `GET` | `/health` | Liveness + active session count. |
| `GET` | `/hls/{stream_id}/{cam_id}/stream.m3u8` | HLS playlist (auto-served from `hls_out/`). |
| `GET` | `/hls/{stream_id}/{cam_id}/seg_*.ts` | HLS segments. |

### Start a session — using the YAML config as-is

```bash
curl -X POST http://localhost:9000/streams/start \
  -H "Content-Type: application/json" \
  -d '{
    "config_path": "config/config.yaml",
    "use_nvenc": false,
    "output_fps": 25
  }'
```

If `config_path` is omitted, the server falls back to
`production/config/config.yaml`.

### Start a session — overriding the camera list

```bash
curl -X POST http://localhost:9000/streams/start \
  -H "Content-Type: application/json" \
  -d '{
    "cameras": [
      {"id": "cam1", "role": "MASTER", "source": "rtsp://user:pass@192.168.1.10:554/Streaming/Channels/101"},
      {"id": "cam2", "role": "SLAVE",  "source": "rtsp://user:pass@192.168.1.11:554/Streaming/Channels/101"}
    ],
    "use_nvenc": false
  }'
```

Response:

```json
{
  "stream_id": "550e8400-e29b-41d4-a716-446655440000",
  "thread_name": "hls-550e8400",
  "hls_urls": {
    "cam1": "/hls/550e8400-e29b-41d4-a716-446655440000/cam1/stream.m3u8",
    "cam2": "/hls/550e8400-e29b-41d4-a716-446655440000/cam2/stream.m3u8"
  },
  "cameras": [
    {"id": "cam1", "role": "MASTER", "source": "rtsp://..."},
    {"id": "cam2", "role": "SLAVE",  "source": "rtsp://..."}
  ]
}
```

### Watch the stream

Open the playlist URL in any HLS-capable player:

- **VLC**: *Media → Open Network Stream* →
  `http://localhost:9000/hls/<stream_id>/cam1/stream.m3u8`
- **ffplay**: `ffplay http://localhost:9000/hls/<stream_id>/cam1/stream.m3u8`
- **Safari**: paste the URL into the address bar.
- **Chrome / Firefox**: use a tiny [hls.js](https://github.com/video-dev/hls.js)
  page — Chrome and Firefox don't play HLS natively.

It usually takes ~3–5 seconds before the first segment lands on disk and the
playlist becomes valid.

### Stop a session

```bash
curl -X POST http://localhost:9000/streams/stop/<stream_id>
```

This signals the worker thread, drains and closes ffmpeg, stops the capture
threads, and deletes `hls_out/<stream_id>/`.

### Inspect sessions

```bash
curl http://localhost:9000/streams                         # list all
curl http://localhost:9000/streams/<stream_id>             # one session
curl http://localhost:9000/health
```

### `use_nvenc` flag

Mirrors the `ffmpeg_test` API. The writer accepts it and, if the host has
NVIDIA hardware + ffmpeg compiled with `--enable-nvenc`, swaps libx264 for
`h264_nvenc`. Set `false` to force software encoding, which is portable.

---

## Local testing without RTSP cameras

For quick smoke tests, point `source` at a video file or HTTP MP4:

```bash
curl -X POST http://localhost:9000/streams/start \
  -H "Content-Type: application/json" \
  -d '{
    "cameras": [
      {"id": "cam1", "role": "MASTER", "source": "/path/to/sample.mp4"}
    ],
    "use_nvenc": false
  }'
```

The same source field accepts a webcam index as a string (e.g. `"0"`).

---

## Troubleshooting

- **`ffmpeg not found on PATH`** — install ffmpeg and reopen the shell.
  On Windows, `winget install Gyan.FFmpeg`. On macOS, `brew install ffmpeg`.
- **HLS playlist 404 for a few seconds after start** — normal: ffmpeg needs
  to write at least one segment (`hls_time: 2s`) before the playlist is
  valid. Reload after ~5 seconds.
- **Worker exits immediately, session shows `error`** — usually one of:
  RTSP URL unreachable, codec OpenCV can't open, or model file missing.
  `GET /streams/{id}` shows the error string.
- **Cross-camera SIDs don't match between sessions** — expected. Each
  session is independent unless `database.qdrant.keep_db: true` is set in
  the YAML; with `keep_db: true`, sessions share the persisted gallery.
- **`runtime.roi_based_master` warning in logs** — ROI selection requires a
  GUI window. Server mode disables it automatically; the warning is just FYI.
- **High GPU memory** — every session loads its own YOLO instance per camera.
  Cap concurrent sessions or reduce camera count per session.

---

## File layout (post-change)

```
production/
├── app.py                                  # CLI runner (cv2 window)
├── app_server.py                           # WebSocket server
├── app_hls.py                              # HLS HTTP server  ◄── new
├── README.md                               # this file        ◄── new
├── config/
│   └── config.yaml
├── hls_out/                                # created at runtime by app_hls.py
└── src/crosscamreid/
    ├── capture.py                          # RTSPCapture (thread per camera)
    ├── config.py
    ├── keypoints.py
    ├── overlay.py
    ├── pipeline.py                         # used by app.py
    ├── processor.py                        # process_master / _slave / _master_roi
    ├── reid/                               # onnx / tensorrt / fastreid backends
    ├── state.py
    ├── store.py
    ├── database/                           # PostgreSQL persistence
    ├── websocket/                          # used by app_server.py
    └── server/                             # used by app_hls.py  ◄── new
        ├── __init__.py
        ├── ffmpeg_hls.py                   # FFmpegHLSWriter
        └── hls_session.py                  # HLSStreamSession (worker thread)
```
