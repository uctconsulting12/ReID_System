from __future__ import annotations

import json
import time

import cv2
from ultralytics import YOLO

from .capture import RTSPCapture
from .config import AppConfig
from .overlay import combine_frames_grid, draw_overlay
from .pose_loader import load_pose_model
from .processor import process_master, process_master_roi, process_slave
from .reid.factory import create_reid_backend
from .state import TIDStateManager
from .store import SIDStore


def _bbox_inside_roi(bbox, roi) -> bool:
    """True when all four corners of the person bbox lie inside the ROI rect."""
    if roi is None:
        return True
    x1, y1, x2, y2 = bbox
    rx, ry, rw, rh = roi
    return x1 >= rx and y1 >= ry and x2 <= rx + rw and y2 <= ry + rh


def _select_master_roi(captures, master_cam_id: str, timeout_sec: float = 30.0):
    print(f"[ROI] Waiting for first frame on master '{master_cam_id}'...")
    deadline = time.time() + timeout_sec
    frame = None
    while time.time() < deadline:
        frame = captures[master_cam_id].get_frame()
        if frame is not None:
            break
        time.sleep(0.1)
    if frame is None:
        print(f"[ROI] No frame from '{master_cam_id}' within {timeout_sec:.0f}s; ROI mode disabled.")
        return None
    title = "Select MASTER ROI - drag rectangle, ENTER to confirm, c to cancel"
    print(f"[ROI] {title}")
    rect = cv2.selectROI(title, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(title)
    cv2.waitKey(1)
    x, y, w, h = (int(v) for v in rect)
    if w == 0 or h == 0:
        print("[ROI] No ROI selected; ROI mode disabled.")
        return None
    print(f"[ROI] Master ROI set: x={x} y={y} w={w} h={h}")
    return (x, y, w, h)


def _run_stream(
    pose: YOLO,
    frame,
    config: AppConfig,
    pick_processor,
    reid_backend,
    store,
    states,
):
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
            processor = pick_processor(bbox)
            rec = processor(
                frame,
                bbox,
                kp_xy[i],
                kp_conf[i],
                int(stable_tids[i]),
                reid_backend,
                store,
                states,
                config,
            )
            records.append(rec)
            stable_tid = int(stable_tids[i])
            alive.add(stable_tid)
            alive_boxes[stable_tid] = (bbox[0], bbox[1], bbox[2], bbox[3])

    states.forget(alive, alive_boxes)
    return records


def _make_picker(role: str, master_roi):
    if role == "MASTER":
        if master_roi is not None:
            def pick(bbox, _roi=master_roi):
                return process_master_roi if _bbox_inside_roi(bbox, _roi) else process_slave
            return pick
        return lambda bbox: process_master
    return lambda bbox: process_slave


def run_app(config: AppConfig) -> int:
    print("\n" + "=" * 70)
    print("  CrossCamReid (Multi-camera, torso-region ReID)")
    for cam in config.cameras:
        print(f"  {cam.camera_id:<14} [{cam.role}] : {cam.source}")
    print(f"  Pose model       : {config.models.pose_path}")
    print(f"  ReID backend     : {config.runtime.reid_backend}")
    print(f"  ReID ONNX model  : {config.models.reid_onnx_path}")
    print(f"  ReID TRT engine  : {config.models.reid_tensorrt_engine_path}")
    if config.runtime.reid_backend == "fastreid":
        print(f"  fast-reid root   : {config.models.fastreid_root}")
        print(f"  fast-reid config : {config.models.fastreid_config}")
        print(f"  fast-reid weights: {config.models.fastreid_weights}")
        print(f"  fast-reid device : {config.models.fastreid_device}")
    q = config.database.qdrant
    print(f"  Qdrant mode      : {q.mode}")
    if q.mode == "local":
        print(f"  DB path          : {q.local_path}")
    else:
        print(f"  DB cloud URL     : {q.cloud_url}")
    print(f"  Collection       : {q.collection}")
    print(f"  Match threshold  : {config.gating.match_thresh}")
    print(f"  Region pad frac  : {config.gating.region_pad_frac}")
    print(f"  Fresh DB         : {not q.keep_db}")
    print(f"  Postgres enabled : {config.database.postgres.enabled}")
    print("=" * 70 + "\n")

    print("[Pose] Loading per-camera pose model instances...")
    pose_models: dict[str, YOLO] = {}
    for cam in config.cameras:
        pose_models[cam.camera_id] = load_pose_model(config.models.pose_path)

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

    state_managers: dict[str, TIDStateManager] = {}
    captures: dict[str, RTSPCapture] = {}
    for cam in config.cameras:
        state_managers[cam.camera_id] = TIDStateManager(
            max_missing_frames=config.runtime.tid_recover_max_missing_frames,
            recover_iou_threshold=config.runtime.tid_recover_iou_threshold,
        )
        cap_name = f"{cam.camera_id}-{cam.role[0]}"
        captures[cam.camera_id] = RTSPCapture(cam.source, cap_name, config.capture).start()

    master_roi = None
    if config.runtime.roi_based_master:
        if config.runtime.no_display:
            print("[ROI] roi_based_master requires a GUI; cannot select ROI when no_display=true. Disabling.")
        else:
            master_cam = next((c for c in config.cameras if c.role == "MASTER"), None)
            if master_cam is None:
                print("[ROI] No MASTER camera found; ROI mode disabled.")
            else:
                master_roi = _select_master_roi(captures, master_cam.camera_id)

    pickers = {
        cam.camera_id: _make_picker(
            cam.role, master_roi if cam.role == "MASTER" else None
        )
        for cam in config.cameras
    }

    fps_count, fps_start, fps = 0, time.time(), 0.0
    print("[Main] Running. Press Q to quit.")

    try:
        while True:
            frames = {cam.camera_id: captures[cam.camera_id].get_frame() for cam in config.cameras}
            if all(frame is None for frame in frames.values()):
                time.sleep(0.05)
                continue

            camera_records: dict[str, list[dict]] = {cam.camera_id: [] for cam in config.cameras}

            for cam in config.cameras:
                frame = frames[cam.camera_id]
                if frame is None:
                    continue
                camera_records[cam.camera_id] = _run_stream(
                    pose_models[cam.camera_id],
                    frame,
                    config,
                    pickers[cam.camera_id],
                    reid_backend,
                    store,
                    state_managers[cam.camera_id],
                )

            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                fps_count, fps_start = 0, time.time()

            if config.runtime.log_json and any(camera_records.values()):
                payload = {
                    "fps": round(fps, 2),
                    "cameras": {
                        cam.camera_id: [
                            {k: v for k, v in rec.items() if k not in ("kp_xy", "kp_conf")}
                            for rec in camera_records[cam.camera_id]
                        ]
                        for cam in config.cameras
                    },
                }
                print(json.dumps(payload))

            if config.runtime.no_display:
                if any(camera_records.values()):
                    def _fmt(tag, recs):
                        parts = []
                        for rec in recs:
                            score = rec["similarity_score"]
                            suffix = f"({score:.2f})" if score is not None else ""
                            parts.append(f"{tag}T{rec['tid']}->{rec['sid']}{suffix}")
                        return ", ".join(parts)

                    chunks = []
                    for cam in config.cameras:
                        recs = camera_records[cam.camera_id]
                        if not recs:
                            continue
                        tag = f"{cam.camera_id}[{cam.role[0]}]:"
                        chunks.append(_fmt(tag, recs))
                    line = " | ".join(chunks)
                    print(f"[{fps:5.1f} fps] {line}")
            else:
                display_frames = []
                labels = []
                for cam in config.cameras:
                    frame = frames[cam.camera_id]
                    if frame is not None:
                        frame = draw_overlay(
                            frame,
                            camera_records[cam.camera_id],
                            fps,
                            store.total_sids(),
                            cam.camera_id,
                            cam.role,
                            config,
                        )
                        if cam.role == "MASTER" and master_roi is not None:
                            rx, ry, rw, rh = master_roi
                            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
                            cv2.putText(
                                frame, "MASTER ROI",
                                (rx, max(15, ry - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                            )
                    display_frames.append(frame)
                    labels.append(cam.camera_id)
                combined = combine_frames_grid(display_frames, labels, config.runtime.display_width)
                cv2.imshow("CrossCamReid | Q to quit", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n[Main] Interrupted.")
    finally:
        for capture in captures.values():
            capture.stop()
        if not config.runtime.no_display:
            cv2.destroyAllWindows()

    print(f"[Main] Done. Total SIDs: {store.total_sids()}")
    return 0
