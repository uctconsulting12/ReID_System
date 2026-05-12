"""
people_counting_runner.py
=========================
PeopleCountingSession — multi-camera people-counting / occupancy runner that
streams JSON payloads back to the connected WebSocket.

Design:
  * One ``PeopleCountingSession`` per WebSocket connection.
  * Each session owns up to 5 cameras; each camera runs in its own thread so
    that one slow / disconnected stream does not stall the others.
  * All cameras of the same ``org_id`` share one ReID gallery (SIDStore +
    backend) via :class:`OrgRegistry`, so global IDs are consistent across
    cameras within an org.
  * ReID logic is reused unchanged: ``process_master`` from ``processor.py``.
  * The emitted payload follows the public spec exactly (see app docs).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import cv2
import numpy as np
from ultralytics import YOLO

from ..capture import RTSPCapture
from ..config import AppConfig
from ..counting.dwell import DwellTracker
from ..counting.entry_exit import EntryExitTracker
from ..counting.gpu_stats import gpu_stats
from ..counting.kvs_resolver import resolve_stream_url
from ..counting.occupancy import OccupancyTracker
from ..counting.org_registry import OrgRegistry, OrgResources
from ..overlay import draw_overlay
from ..processor import UNKNOWN_LABEL, process_master, process_slave
from ..state import TIDStateManager

logger = logging.getLogger("people_counting.runner")


# ── helpers ──────────────────────────────────────────────────────────────────


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def _now_iso_us() -> str:
    """ISO-8601 timestamp in UTC with microsecond precision (e.g.
    ``2024-12-17T15:39:49.123456Z``)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _iso_seconds(epoch: float) -> str:
    """ISO-8601 timestamp in UTC truncated to seconds (used for
    Entry_time / Exit_time)."""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _global_id(sid: int | str) -> str:
    if isinstance(sid, str):
        return sid
    return f"G{int(sid)}"


def _encode_jpeg_b64(frame: np.ndarray, quality: int) -> str | None:
    """Encode a BGR frame as JPEG and return its base64-ASCII string. Returns
    ``None`` if encoding fails."""
    q = max(30, min(95, int(quality)))
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


@dataclass
class CameraSpec:
    camera_id: int
    stream_url: str
    region: str | None = None
    # "MASTER" → enrolls new SIDs and matches against the gallery.
    # "SLAVE"  → matches only; never creates new identities.
    # Default is "MASTER" so omitting the field preserves legacy behaviour.
    role: str = "MASTER"


# ── per-camera worker ────────────────────────────────────────────────────────


class _CameraWorker:
    """One thread per camera. Owns its own YOLO instance + capture + per-cam
    state, but shares ``OrgResources`` (ReID backend + SIDStore) with the
    rest of its org."""

    def __init__(
        self,
        session: "PeopleCountingSession",
        spec: CameraSpec,
        resources: OrgResources,
    ) -> None:
        self.session = session
        self.spec = spec
        self.cam_id = int(spec.camera_id)
        self.resources = resources
        self.role = (spec.role or "MASTER").upper()
        # Pick the per-detection processor once at worker construction time.
        # MASTER  → search + enroll (creates new SIDs).
        # SLAVE   → search-only (only recognizes SIDs that masters created).
        self._process_fn = process_slave if self.role == "SLAVE" else process_master

        cfg = session.config
        self.config = cfg

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # Each camera has its own YOLO instance to keep per-thread state
        # isolated; the heavy weights are mmap-shared by torch under the hood
        # so RAM cost is small.
        self._pose: YOLO | None = None
        self._capture: RTSPCapture | None = None

        self._states = TIDStateManager(
            max_missing_frames=cfg.runtime.tid_recover_max_missing_frames,
            recover_iou_threshold=cfg.runtime.tid_recover_iou_threshold,
        )

        self._entry_exit = EntryExitTracker(
            exit_timeout_sec=session.exit_timeout_sec,
            shared_dwell=resources.dwell_lifetime_sec,
            shared_dwell_lock=resources.dwell_lock,
            peer_active=lambda sid: session._peer_active_for(sid, self),
        )
        self._occupancy = OccupancyTracker(
            threshold=session.threshold,
            alert_rate=session.alert_rate,
        )

        self._frame_idx = 0
        self._processed_idx = 0
        self._fps_count = 0
        self._fps_window_start = time.time()
        self._stream_fps = 0.0

        self._min_frame_interval = (
            1.0 / float(session.target_fps) if session.target_fps > 0 else 0.0
        )
        self._last_processed_at = 0.0

        # GPU stats are pulled via pynvml — cheap but not free. Cache for ~1s
        # so we don't pay it on every payload.
        self._gpu_stats_cache: tuple[float, float] = (0.0, 0.0)
        self._gpu_stats_at: float = 0.0

        # Used to dedupe "no fresh frame" payloads -- we don't want to spam.
        self._last_frame_signature: int = -1

        self.error_message: str | None = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run_safe,
            name=f"pc-cam{self.cam_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, join_timeout: float = 5.0) -> None:
        self._stop.set()
        if self._capture is not None:
            try:
                self._capture.stop()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)

    # ── internals ────────────────────────────────────────────────────────

    def _send(self, payload: dict) -> None:
        self.session.send_json(payload)

    def _send_error(self, message: str) -> None:
        self._send({
            "event": "error",
            "camera_id": self.cam_id,
            "status": "failed",
            "error_message": message,
        })

    def _run_safe(self) -> None:
        try:
            self._run()
        except Exception as exc:
            logger.exception("[cam=%s] worker crashed: %s", self.cam_id, exc)
            self.error_message = f"{type(exc).__name__}: {exc}"
            self._send_error(self.error_message)

    def _run(self) -> None:
        # ── resolve stream URL (handle kvs://) ───────────────────────────
        try:
            resolved = resolve_stream_url(self.spec.stream_url, region=self.spec.region)
        except Exception as exc:
            self._send_error(f"failed to resolve stream URL: {exc}")
            return

        # ── load model + open capture ────────────────────────────────────
        try:
            self._pose = YOLO(self.config.models.pose_path)
        except Exception as exc:
            self._send_error(f"failed to load pose model: {exc}")
            return

        try:
            self._capture = RTSPCapture(
                resolved, f"pc-cam{self.cam_id}", self.config.capture
            ).start()
        except Exception as exc:
            self._send_error(f"failed to open stream: {exc}")
            return

        self._send({
            "event": "stream_started",
            "camera_id": self.cam_id,
            "role": self.role,
            "status": "ok",
            "message": "Stream initialized successfully",
        })
        logger.info("[cam=%s] role=%s started", self.cam_id, self.role)

        first_frame_deadline = time.time() + 30.0
        got_first_frame = False

        # ── main loop ────────────────────────────────────────────────────
        try:
            while not self._stop.is_set():
                now = time.time()

                # Pace the loop to roughly hit target_fps.
                if self._min_frame_interval > 0:
                    sleep_for = (self._last_processed_at + self._min_frame_interval) - now
                    if sleep_for > 0:
                        time.sleep(min(sleep_for, 0.05))
                        continue

                frame = self._capture.get_frame()
                if frame is None:
                    # haven't got a frame yet; if we've been waiting too long
                    # at startup, surface it once.
                    if not got_first_frame and now > first_frame_deadline:
                        self._send_error("Unable to connect to stream")
                        first_frame_deadline = now + 30.0
                    time.sleep(0.05)
                    continue
                got_first_frame = True

                # Use frame identity to detect 'same frame as last tick' and
                # skip when the capture hasn't refreshed.
                sig = id(frame)
                self._frame_idx += 1
                if self._frame_idx % max(1, self.session.frame_skip + 1) != 0:
                    continue
                if sig == self._last_frame_signature:
                    # No new frame from capture; sweep + emit a heartbeat-ish
                    # tick so the client still gets exits / dwell updates,
                    # but throttle these to ~1 Hz.
                    if now - self._last_processed_at < 1.0:
                        time.sleep(0.02)
                        continue

                self._last_frame_signature = sig
                self._last_processed_at = now

                tic = time.perf_counter()

                # ── detection + ReID ─────────────────────────────────────
                people_visible_records, _ = self._detect_and_reid(frame)

                # ── entry / exit -> parallel arrays (per spec) ───────────
                people_ids: list[str] = []
                entry_times: list[str] = []
                dwell_times: list[str] = []
                confidence_scores: list[float] = []
                accuracy_scores: list[float] = []

                for rec in people_visible_records:
                    sid_raw = rec["sid"]
                    if sid_raw == "UNKNOWN":
                        # Tracks that haven't locked to a global SID yet are
                        # not yet "identified"; they'll appear once enrolled.
                        continue
                    sid = _global_id(sid_raw)
                    bbox = [float(c) for c in rec["bbox"]]
                    conf = float(rec.get("similarity_score") or 0.0)

                    is_new = self._entry_exit.observe(
                        sid, bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                        confidence=conf, now=now,
                    )
                    info = self._entry_exit._active.get(sid)
                    entry_epoch = info.entry_time if info is not None else now

                    people_ids.append(sid)
                    entry_times.append(_iso_seconds(entry_epoch))
                    # Accumulated lifetime dwell for this SID on this camera —
                    # re-entries add to the running total instead of resetting.
                    dwell_times.append(
                        DwellTracker.lifetime(self._entry_exit, sid, now=now)
                    )
                    confidence_scores.append(round(conf, 2))
                    accuracy_scores.append(round(conf, 3))
                    if is_new:
                        logger.debug("[cam=%s] entry: %s", self.cam_id, sid)

                # `sweep` walks the active set; we don't need its return value
                # because we read the rolling last-10 exits straight from the
                # tracker below.
                self._entry_exit.sweep(now=now)
                recent = self._entry_exit.recent_exits(10)
                exit_ids = [sid for sid, _, _ in recent]
                exit_times = [_iso_seconds(t_exit) for _, _, t_exit in recent]

                occ_snap = self._occupancy.update(self._entry_exit.current_occupancy)

                # ── stream FPS ───────────────────────────────────────────
                self._fps_count += 1
                fps_elapsed = now - self._fps_window_start
                if fps_elapsed >= 1.0:
                    self._stream_fps = self._fps_count / fps_elapsed
                    self._fps_count = 0
                    self._fps_window_start = now

                self._processed_idx += 1

                # ── annotated frame (base64 JPEG, every payload) ─────────
                annotated_b64: str | None = None
                if self.session.include_annotated_frame:
                    try:
                        dwell_by_sid = dict(zip(people_ids, dwell_times))
                        avg_dwell_str = DwellTracker.average(self._entry_exit, now=now)
                        annotated = draw_overlay(
                            frame.copy(),
                            people_visible_records,
                            self._stream_fps,
                            int(self.resources.store.total_sids()),
                            cam_label=str(self.cam_id),
                            mode_label="MASTER",
                            config=self.config,
                            dwell_by_sid=dwell_by_sid,
                            avg_dwell=avg_dwell_str,
                            occupancy=self._entry_exit.current_occupancy,
                        )
                        annotated_b64 = _encode_jpeg_b64(
                            annotated, self.session.frame_jpeg_quality
                        )
                    except Exception as exc:
                        logger.warning(
                            "[cam=%s] overlay/encode failed: %s", self.cam_id, exc
                        )

                processing_ms = round((time.perf_counter() - tic) * 1000.0, 2)
                if now - self._gpu_stats_at >= 1.0:
                    self._gpu_stats_cache = gpu_stats()
                    self._gpu_stats_at = now
                gpu_mem, gpu_util = self._gpu_stats_cache

                payload = self._build_payload(
                    people_ids=people_ids,
                    entry_times=entry_times,
                    dwell_times=dwell_times,
                    confidence_scores=confidence_scores,
                    accuracy_scores=accuracy_scores,
                    exit_ids=exit_ids,
                    exit_times=exit_times,
                    occupancy=occ_snap,
                    processing_ms=processing_ms,
                    gpu_mem=gpu_mem,
                    gpu_util=gpu_util,
                    now=now,
                    annotated_b64=annotated_b64,
                )
                self._send(payload)

        finally:
            try:
                if self._capture is not None:
                    self._capture.stop()
            except Exception:
                pass

    # ── detection + ReID over a single frame ─────────────────────────────

    def _detect_and_reid(self, frame) -> tuple[list[dict], dict]:
        """Run the existing master pipeline on one frame; return the same
        ``records`` shape produced by ``processor.process_master``."""
        cfg = self.config
        tracker_path = (
            cfg.runtime.botsort_tracker
            if cfg.runtime.tracker_mode == "botsort"
            else cfg.runtime.tracker
        )
        result = self._pose.track(
            frame,
            persist=True,
            classes=[0],
            conf=cfg.gating.person_conf_thresh,
            tracker=tracker_path,
            imgsz=480,
            half=True,
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
            stable_tids = self._states.remap_tids(tids, boxes)
            kp_xy = result.keypoints.xy.cpu().numpy()
            kp_conf = result.keypoints.conf.cpu().numpy()

            occluded_mask = (
                self._compute_occlusion_mask(boxes, cfg.runtime.occlusion_iou_thresh)
                if cfg.runtime.occlusion_freeze_embeddings
                else [False] * len(boxes)
            )

            for i in range(len(boxes)):
                bbox = [float(c) for c in boxes[i]]
                rec = self._process_fn(
                    frame, bbox, kp_xy[i], kp_conf[i], int(stable_tids[i]),
                    self.resources.reid_backend,
                    self.resources.store,
                    self._states,
                    cfg,
                    occluded=bool(occluded_mask[i]),
                )
                records.append(rec)
                stable_tid = int(stable_tids[i])
                alive.add(stable_tid)
                alive_boxes[stable_tid] = (bbox[0], bbox[1], bbox[2], bbox[3])

        self._states.forget(alive, alive_boxes)
        self._resolve_sid_conflicts(records)
        return records, {}

    @staticmethod
    def _compute_occlusion_mask(boxes: np.ndarray, iou_thresh: float) -> list[bool]:
        """Per-detection occlusion flag — True if this bbox overlaps any
        other bbox in the same frame with IoU >= iou_thresh. Used to
        suppress gallery embedding updates during heavy occlusion."""
        n = len(boxes)
        if n < 2:
            return [False] * n
        x1 = boxes[:, 0]; y1 = boxes[:, 1]
        x2 = boxes[:, 2]; y2 = boxes[:, 3]
        area = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        flags = [False] * n
        for i in range(n):
            for j in range(i + 1, n):
                ix1 = max(x1[i], x1[j]); iy1 = max(y1[i], y1[j])
                ix2 = min(x2[i], x2[j]); iy2 = min(y2[i], y2[j])
                iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
                inter = iw * ih
                if inter <= 0.0:
                    continue
                denom = area[i] + area[j] - inter
                if denom <= 0.0:
                    continue
                if (inter / denom) >= iou_thresh:
                    flags[i] = True
                    flags[j] = True
        return flags

    def _resolve_sid_conflicts(self, records: list[dict]) -> None:
        """Enforce one-SID-per-frame.

        If the same global SID appears on more than one detection in this
        frame, keep the holder with the highest matching score and revert
        the others to UNKNOWN. The score used for an incumbent (no current
        match was run because it short-circuited on its existing lock) is
        the similarity score recorded at lock time on its TIDState.

        Losers also have their lock cleared so the next frame can reassign
        them (either to a different SID via ReID, or back to this SID once
        the winner moves out of view).
        """
        by_sid: dict[int, list[int]] = {}
        for idx, rec in enumerate(records):
            sid = rec["sid"]
            if sid == UNKNOWN_LABEL:
                continue
            by_sid.setdefault(int(sid), []).append(idx)

        def score_of(idx: int) -> float:
            rec = records[idx]
            cur = rec.get("similarity_score")
            if cur is not None:
                return float(cur)
            state = self._states.get(int(rec["tid"]))
            return float(getattr(state, "lock_score", 0.0))

        for sid, idxs in by_sid.items():
            if len(idxs) <= 1:
                continue
            winner = max(idxs, key=score_of)
            for idx in idxs:
                if idx == winner:
                    continue
                losing = records[idx]
                state = self._states.get(int(losing["tid"]))
                if state is not None:
                    state.locked_sid = None
                    state.lock_score = 0.0
                    state.decided = False
                logger.debug(
                    "[cam=%s] SID conflict on %s: tid=%s lost (score=%.3f) "
                    "to tid=%s (score=%.3f)",
                    self.cam_id, sid, losing["tid"], score_of(idx),
                    records[winner]["tid"], score_of(winner),
                )
                losing["sid"] = UNKNOWN_LABEL
                losing["similarity_score"] = None

    # ── payload formatter ────────────────────────────────────────────────

    def _build_payload(
        self,
        *,
        people_ids: list[str],
        entry_times: list[str],
        dwell_times: list[str],
        confidence_scores: list[float],
        accuracy_scores: list[float],
        exit_ids: list[str],
        exit_times: list[str],
        occupancy,
        processing_ms: float,
        gpu_mem: float,
        gpu_util: float,
        now: float,
        annotated_b64: str | None = None,
    ) -> dict:
        ts_ms = int(now * 1000)
        frame_id = f"FR_{self.cam_id}_{ts_ms}"

        # Status: empty when normal/approaching, "High Occupancy" once we're
        # at or above the configured threshold.
        if occupancy.status in ("at_capacity", "over_capacity"):
            status_str = "High Occupancy"
        else:
            status_str = ""

        return {
            "detections": {
                "camid": self.cam_id,
                "org_id": self.session.org_id,
                "userid": self.session.user_id,

                "Frame_Id": frame_id,
                "Time_stamp": _now_iso_us(),
                "Frame_Count": self._processed_idx,

                "Total_people_detected": len(people_ids),
                "Current_occupancy": self._entry_exit.current_occupancy,

                "People_ids": people_ids,
                "Entry_time": entry_times,
                "People_dwell_time": dwell_times,
                "Confidence_scores": confidence_scores,
                "accuracy": accuracy_scores,

                "Exit_time": exit_times,
                "exitid": exit_ids,

                "Total_entries": self._entry_exit.total_entries,
                "Total_exits": self._entry_exit.total_exits,
                "Net_count": self._entry_exit.net_count,

                "Occupancy_percentage": occupancy.percentage,
                "Over_capacity_count": occupancy.over_capacity,
                "Max_occupancy": occupancy.threshold,
                "Average_dwell_time": DwellTracker.average(self._entry_exit, now=now),

                "Status": status_str,
                "is_alert_triggered": bool(occupancy.alert_triggered),

                "Processing_Status": 1,
                "processing_time_ms": processing_ms,
                "gpu_memory_percent": round(gpu_mem, 1),
                "gpu_utilization_percent": round(gpu_util, 1),
                "reid_gallery_size": int(self.resources.store.total_sids()),

                "annotated_frame": annotated_b64,
            }
        }


# ── session ──────────────────────────────────────────────────────────────────


class PeopleCountingSession:
    """Owns a set of ``_CameraWorker`` instances on behalf of one WebSocket."""

    MAX_CAMERAS = 5

    def __init__(
        self,
        *,
        client_id: str,
        org_id: int,
        user_id: int,
        config: AppConfig,
        org_registry: OrgRegistry,
        ws,
        loop: asyncio.AbstractEventLoop,
        threshold: int,
        alert_rate: float,
        target_fps: int,
        frame_skip: int,
        exit_timeout_sec: float = 2.0,
        include_annotated_frame: bool = True,
        frame_jpeg_quality: int = 70,
        frame_send_interval: int = 20,
    ) -> None:
        self.client_id = client_id
        self.org_id = int(org_id)
        self.user_id = int(user_id)
        self.config = config
        self.registry = org_registry
        self.ws = ws
        self.loop = loop

        self.threshold = int(threshold)
        self.alert_rate = float(alert_rate)
        self.target_fps = max(0, int(target_fps))
        self.frame_skip = max(0, int(frame_skip))
        self.exit_timeout_sec = float(exit_timeout_sec)
        self.include_annotated_frame = bool(include_annotated_frame)
        self.frame_jpeg_quality = max(30, min(95, int(frame_jpeg_quality)))
        self.frame_send_interval = max(1, int(frame_send_interval))

        self._lock = threading.Lock()
        self._workers: dict[int, _CameraWorker] = {}
        self._send_lock = threading.Lock()
        self._closed = False

        self._resources: OrgResources = self.registry.acquire(
            self.org_id,
            reset_collection=bool(
                getattr(self.config.database.qdrant, "reset_on_connect", False)
            ),
        )

    def _peer_active_for(self, sid: str, caller):
        """Yield in-progress entry_time floats for `sid` on every camera
        worker in this session *other than* ``caller``. EntryExitTracker
        already adds the caller's own active entry_time, so we must not
        double-count it here."""
        for worker in list(self._workers.values()):
            if worker is caller:
                continue
            tracker = getattr(worker, "_entry_exit", None)
            if tracker is None:
                continue
            info = tracker._active.get(sid)
            if info is not None:
                yield info.entry_time

    # ── thread-safe websocket send ───────────────────────────────────────

    def send_json(self, payload: dict) -> None:
        if self._closed:
            return
        ws = self.ws
        if ws is None:
            return
        try:
            text = json.dumps(payload, default=_json_default)
        except Exception as exc:
            logger.error("[%s] payload serialisation failed: %s", self.client_id, exc)
            return
        try:
            asyncio.run_coroutine_threadsafe(ws.send_text(text), self.loop)
        except Exception as exc:
            logger.warning("[%s] ws send failed: %s", self.client_id, exc)

    # ── api ──────────────────────────────────────────────────────────────

    def start_cameras(self, cameras: list[CameraSpec]) -> tuple[list[int], list[tuple[int, str]]]:
        """Start workers for each camera. Returns (started_ids, errors)."""
        started: list[int] = []
        errors: list[tuple[int, str]] = []
        with self._lock:
            available = self.MAX_CAMERAS - len(self._workers)
            if available <= 0:
                for c in cameras:
                    errors.append((c.camera_id, "max 5 cameras per connection"))
                return started, errors

            for spec in cameras[:available]:
                if spec.camera_id in self._workers:
                    errors.append((spec.camera_id, "camera already streaming"))
                    continue
                worker = _CameraWorker(self, spec, self._resources)
                self._workers[spec.camera_id] = worker
                worker.start()
                started.append(spec.camera_id)

            for spec in cameras[available:]:
                errors.append((spec.camera_id, "max 5 cameras per connection"))

        return started, errors

    def stop_camera(self, camera_id: int) -> bool:
        with self._lock:
            worker = self._workers.pop(int(camera_id), None)
        if worker is None:
            return False
        worker.stop()
        return True

    def stop_all(self) -> int:
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for w in workers:
            try:
                w.stop()
            except Exception:
                pass
        return len(workers)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop_all()
        try:
            self.registry.release(self.org_id)
        except Exception:
            pass

    # ── status ───────────────────────────────────────────────────────────

    def status(self) -> dict:
        with self._lock:
            return {
                "client_id": self.client_id,
                "org_id": self.org_id,
                "user_id": self.user_id,
                "active_cameras": [
                    {"camera_id": cid, "error": w.error_message}
                    for cid, w in self._workers.items()
                ],
            }
