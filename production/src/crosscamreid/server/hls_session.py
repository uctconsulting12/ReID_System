from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path

from ultralytics import YOLO

from ..capture import RTSPCapture
from ..config import AppConfig
from ..overlay import draw_overlay
from ..pose_loader import load_pose_model
from ..processor import process_master, process_slave
from ..reid.factory import create_reid_backend
from ..state import TIDStateManager
from ..store import SIDStore
from .ffmpeg_hls import FFmpegHLSWriter

logger = logging.getLogger("hls.session")


def _run_frame(pose, frame, config, processor, reid_backend, store, states):
    """One YOLO+ReID pass over a single frame. Returns the per-detection records
    that ``draw_overlay`` consumes."""
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
        boxes = result.boxes.xyxy.cpu().numpy()
        tids = result.boxes.id.cpu().numpy().astype(int)
        stable_tids = states.remap_tids(tids, boxes)
        kp_xy = result.keypoints.xy.cpu().numpy()
        kp_conf = result.keypoints.conf.cpu().numpy()

        for i in range(len(boxes)):
            bbox = [float(c) for c in boxes[i]]
            rec = processor(
                frame, bbox, kp_xy[i], kp_conf[i], int(stable_tids[i]),
                reid_backend, store, states, config,
            )
            records.append(rec)
            stable_tid = int(stable_tids[i])
            alive.add(stable_tid)
            alive_boxes[stable_tid] = (bbox[0], bbox[1], bbox[2], bbox[3])

    states.forget(alive, alive_boxes)
    return records


class HLSStreamSession:
    """One pipeline session = one ``stream_id``. Owns:

    - a ``RTSPCapture`` thread per camera (existing reader threads),
    - a single worker thread that runs YOLO+ReID over all cameras,
    - one ``FFmpegHLSWriter`` (and one ffmpeg subprocess) per camera that
      consumes annotated frames over stdin and emits an HLS playlist.

    All cameras in the session share one ``SIDStore`` and one ``reid_backend``
    so cross-camera ReID still works.
    """

    def __init__(
        self,
        stream_id: str,
        config: AppConfig,
        hls_root: Path,
        use_nvenc: bool = False,
        output_fps: int = 25,
    ) -> None:
        self.stream_id = stream_id
        self.config = config
        self.hls_root = Path(hls_root)
        self.session_dir = self.hls_root / stream_id
        self.use_nvenc = use_nvenc
        self.output_fps = max(1, int(output_fps))

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.thread_name: str | None = None

        self.captures: dict[str, RTSPCapture] = {}
        self.writers: dict[str, FFmpegHLSWriter] = {}

        self.started_at = time.time()
        self.fps = 0.0
        self.error: str | None = None
        self.total_sids = 0

    # ── public API ────────────────────────────────────────────────────────────

    def hls_urls(self) -> dict[str, str]:
        return {
            cam.camera_id: f"/hls/{self.stream_id}/{cam.camera_id}/stream.m3u8"
            for cam in self.config.cameras
        }

    def status(self) -> dict:
        return {
            "stream_id": self.stream_id,
            "thread_name": self.thread_name,
            "started_at": self.started_at,
            "uptime_sec": round(time.time() - self.started_at, 1),
            "fps": round(self.fps, 2),
            "total_sids": self.total_sids,
            "cameras": [
                {
                    "id": cam.camera_id,
                    "role": cam.role,
                    "source": cam.source,
                    "frames_written": self.writers.get(cam.camera_id).frames_written
                                       if cam.camera_id in self.writers else 0,
                    "hls_url": f"/hls/{self.stream_id}/{cam.camera_id}/stream.m3u8",
                }
                for cam in self.config.cameras
            ],
            "error": self.error,
            "alive": self.is_alive(),
        }

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError(f"session {self.stream_id} already started")
        self._thread = threading.Thread(
            target=self._run_safe,
            name=f"hls-{self.stream_id[:8]}",
            daemon=True,
        )
        self._thread.start()
        self.thread_name = self._thread.name

    def stop(self, remove_files: bool = True, join_timeout: float = 10.0) -> None:
        self._stop_event.set()
        for cap in list(self.captures.values()):
            try: cap.stop()
            except Exception: pass
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
        for w in list(self.writers.values()):
            try: w.close()
            except Exception: pass
        if remove_files:
            shutil.rmtree(self.session_dir, ignore_errors=True)

    # ── worker ────────────────────────────────────────────────────────────────

    def _run_safe(self) -> None:
        try:
            self._run()
        except Exception as exc:
            logger.exception("[%s] session crashed: %s", self.stream_id, exc)
            self.error = f"{type(exc).__name__}: {exc}"

    def _run(self) -> None:
        cfg = self.config

        if cfg.runtime.roi_based_master:
            logger.warning(
                "[%s] runtime.roi_based_master=true is not supported in the HLS "
                "server (no GUI for cv2.selectROI); ROI mode is disabled for "
                "this session.", self.stream_id,
            )

        logger.info("[%s] loading YOLO pose model per camera (%d cameras)",
                    self.stream_id, len(cfg.cameras))
        pose_models: dict[str, YOLO] = {
            cam.camera_id: load_pose_model(cfg.models.pose_path) for cam in cfg.cameras
        }

        reid_backend = create_reid_backend(
            backend=cfg.runtime.reid_backend,
            onnx_path=cfg.models.reid_onnx_path,
            tensorrt_engine_path=cfg.models.reid_tensorrt_engine_path,
            fastreid_root=cfg.models.fastreid_root,
            fastreid_config=cfg.models.fastreid_config,
            fastreid_weights=cfg.models.fastreid_weights,
            fastreid_device=cfg.models.fastreid_device,
            use_grayscale=cfg.runtime.embed_grayscale,
        )

        q = cfg.database.qdrant
        store = SIDStore(
            collection=q.collection,
            dim=reid_backend.dim,
            fresh=not q.keep_db,
            max_embeddings_per_sid=cfg.gating.max_embeddings_per_sid,
            db_path=q.local_path if q.mode == "local" else None,
            cloud_url=q.cloud_url if q.mode == "cloud" else None,
            cloud_api_key=q.cloud_api_key if q.mode == "cloud" else None,
        )

        state_managers: dict[str, TIDStateManager] = {
            cam.camera_id: TIDStateManager(
                max_missing_frames=cfg.runtime.tid_recover_max_missing_frames,
                recover_iou_threshold=cfg.runtime.tid_recover_iou_threshold,
            )
            for cam in cfg.cameras
        }

        for cam in cfg.cameras:
            self.captures[cam.camera_id] = RTSPCapture(
                cam.source, f"{cam.camera_id}-{cam.role[0]}", cfg.capture,
            ).start()
            self.writers[cam.camera_id] = FFmpegHLSWriter(
                self.session_dir / cam.camera_id,
                fps=self.output_fps,
                use_nvenc=self.use_nvenc,
            )

        fps_count, fps_start, fps_local = 0, time.time(), 0.0
        logger.info("[%s] pipeline running. cameras=%s",
                    self.stream_id, [c.camera_id for c in cfg.cameras])

        try:
            while not self._stop_event.is_set():
                frames = {
                    cam.camera_id: self.captures[cam.camera_id].get_frame()
                    for cam in cfg.cameras
                }
                if all(f is None for f in frames.values()):
                    time.sleep(0.05)
                    continue

                fps_count += 1
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    fps_local = fps_count / elapsed
                    fps_count, fps_start = 0, time.time()
                self.fps = fps_local

                for cam in cfg.cameras:
                    frame = frames.get(cam.camera_id)
                    if frame is None:
                        continue

                    processor = process_master if cam.role == "MASTER" else process_slave
                    records = _run_frame(
                        pose_models[cam.camera_id], frame, cfg,
                        processor, reid_backend, store,
                        state_managers[cam.camera_id],
                    )

                    annotated = draw_overlay(
                        frame, records, fps_local, store.total_sids(),
                        cam.camera_id, cam.role, cfg,
                    )
                    self.writers[cam.camera_id].write(annotated)

                self.total_sids = store.total_sids()
        finally:
            for cap in self.captures.values():
                try: cap.stop()
                except Exception: pass
            for w in self.writers.values():
                try: w.close()
                except Exception: pass
            logger.info("[%s] pipeline stopped. total_sids=%d",
                        self.stream_id, self.total_sids)
