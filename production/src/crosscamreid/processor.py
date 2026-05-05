from __future__ import annotations

from .config import AppConfig
from .keypoints import keypoint_gate, torso_region_bbox
from .reid.base import BaseReIDBackend
from .state import TIDState, TIDStateManager
from .store import SIDStore

UNKNOWN_LABEL = "UNKNOWN"


def _blank_record(tid: int, bbox, kp_xy, kp_conf, kp_ok: bool) -> dict:
    return {
        "tid": int(tid),
        "sid": UNKNOWN_LABEL,
        "keypoint_valid": bool(kp_ok),
        "similarity_score": None,
        "bbox": bbox,
        "kp_xy": kp_xy,
        "kp_conf": kp_conf,
        "qualified": 0,
        "enroll_left": 0,
        "region_bbox": None,
    }


def _pick_best_voted_sid(
    state: TIDState, qualify_frames: int
) -> tuple[int | None, float]:
    # Require at least 1/3 of the qualify window to consistently match a single SID.
    min_votes = max(1, qualify_frames // 3)
    best_sid: int | None = None
    best_votes = 0
    best_score = 0.0
    for sid, scores in state.match_votes.items():
        if len(scores) >= min_votes and len(scores) > best_votes:
            best_sid = sid
            best_votes = len(scores)
            best_score = max(scores)
    return best_sid, best_score


def process_master(
    frame,
    bbox,
    kp_xy,
    kp_conf,
    tid,
    reid: BaseReIDBackend,
    store: SIDStore,
    states: TIDStateManager,
    config: AppConfig,
) -> dict:
    tid = int(tid)
    kp_ok = keypoint_gate(kp_conf, config.gating)
    record = _blank_record(tid, bbox, kp_xy, kp_conf, kp_ok)

    state = states.get(tid)

    if not kp_ok:
        if config.runtime.sid_persist_on_kp_loss and state.locked_sid is not None:
            record["sid"] = state.locked_sid
        return record

    record["qualified"] = state.qualified

    region = torso_region_bbox(kp_xy, kp_conf, frame.shape, config.gating)
    record["region_bbox"] = region

    if state.locked_sid is not None:
        record["sid"] = state.locked_sid
        return record

    crop_bbox = bbox if reid.use_full_body else region
    if crop_bbox is None:
        return record

    embedding = reid.embed(frame, crop_bbox)
    if embedding is None:
        return record

    qualify_frames = config.enrollment.qualify_frames
    enroll_frames = config.enrollment.enroll_frames

    sample_interval = max(1, qualify_frames // max(1, enroll_frames))
    if (
        state.qualified % sample_interval == 0
        and len(state.embedding_buffer) < enroll_frames
    ):
        state.embedding_buffer.append(embedding)

    matched_sid, score = store.search_top1(embedding)
    if matched_sid is not None and score >= config.gating.match_thresh:
        if config.enrollment.early_lock_on_match:
            state.locked_sid = matched_sid
            state.decided = True
            record["sid"] = matched_sid
            record["similarity_score"] = float(score)
            return record
        state.match_votes.setdefault(matched_sid, []).append(float(score))

    state.qualified += 1
    record["qualified"] = state.qualified
    if score > 0:
        record["similarity_score"] = float(score)

    if state.qualified < qualify_frames:
        return record

    best_sid, best_score = _pick_best_voted_sid(state, qualify_frames)
    if best_sid is not None:
        state.locked_sid = best_sid
        state.decided = True
        record["sid"] = best_sid
        record["similarity_score"] = best_score
        return record

    if not state.embedding_buffer:
        state.embedding_buffer.append(embedding)

    new_sid = store.new_sid(state.embedding_buffer[0])
    for emb in state.embedding_buffer[1:]:
        store.append(new_sid, emb)

    store.prune_outliers(new_sid)

    state.locked_sid = new_sid
    state.decided = True
    record["sid"] = new_sid
    record["similarity_score"] = None
    return record


def process_master_roi(
    frame,
    bbox,
    kp_xy,
    kp_conf,
    tid,
    reid: BaseReIDBackend,
    store: SIDStore,
    states: TIDStateManager,
    config: AppConfig,
) -> dict:
    """Master flow used inside an ROI: search first, enroll only on miss.

    Each in-ROI frame searches the gallery. On a hit the detection is locked
    to that SID immediately. On a miss the embedding is buffered; once the
    buffer reaches `enrollment.enroll_frames` consecutive misses, a new SID
    is created and outliers are pruned.
    """
    tid = int(tid)
    kp_ok = keypoint_gate(kp_conf, config.gating)
    record = _blank_record(tid, bbox, kp_xy, kp_conf, kp_ok)

    state = states.get(tid)

    if not kp_ok:
        if config.runtime.sid_persist_on_kp_loss and state.locked_sid is not None:
            record["sid"] = state.locked_sid
        return record

    record["qualified"] = state.qualified

    region = torso_region_bbox(kp_xy, kp_conf, frame.shape, config.gating)
    record["region_bbox"] = region

    if state.locked_sid is not None:
        record["sid"] = state.locked_sid
        return record

    crop_bbox = bbox if reid.use_full_body else region
    if crop_bbox is None:
        return record

    embedding = reid.embed(frame, crop_bbox)
    if embedding is None:
        return record

    matched_sid, score = store.search_top1(embedding)
    if matched_sid is not None and score >= config.gating.match_thresh:
        state.locked_sid = matched_sid
        state.decided = True
        state.embedding_buffer.clear()
        record["sid"] = matched_sid
        record["similarity_score"] = float(score)
        return record

    state.embedding_buffer.append(embedding)
    if score > 0:
        record["similarity_score"] = float(score)

    enroll_frames = config.enrollment.enroll_frames
    if len(state.embedding_buffer) < enroll_frames:
        return record

    new_sid = store.new_sid(state.embedding_buffer[0])
    for emb in state.embedding_buffer[1:]:
        store.append(new_sid, emb)
    state.embedding_buffer.clear()
    store.prune_outliers(new_sid)

    state.locked_sid = new_sid
    state.decided = True
    record["sid"] = new_sid
    record["similarity_score"] = None
    return record


def process_slave(
    frame,
    bbox,
    kp_xy,
    kp_conf,
    tid,
    reid: BaseReIDBackend,
    store: SIDStore,
    states: TIDStateManager,
    config: AppConfig,
) -> dict:
    tid = int(tid)
    kp_ok = keypoint_gate(kp_conf, config.gating)
    record = _blank_record(tid, bbox, kp_xy, kp_conf, kp_ok)

    state = states.get(tid)

    if not kp_ok:
        if config.runtime.sid_persist_on_kp_loss and state.locked_sid is not None:
            record["sid"] = state.locked_sid
        return record

    region = torso_region_bbox(kp_xy, kp_conf, frame.shape, config.gating)
    record["region_bbox"] = region

    record["qualified"] = state.qualified

    if state.locked_sid is not None:
        record["sid"] = state.locked_sid
        return record

    crop_bbox = bbox if reid.use_full_body else region
    if crop_bbox is None:
        return record

    embedding = reid.embed(frame, crop_bbox)
    if embedding is None:
        return record

    matched_sid, score = store.search_top1(embedding)
    if score > 0:
        record["similarity_score"] = float(score)

    if matched_sid is not None and score >= config.gating.match_thresh:
        if config.enrollment.early_lock_on_match:
            state.locked_sid = matched_sid
            state.decided = True
            record["sid"] = matched_sid
            return record
        state.match_votes.setdefault(matched_sid, []).append(float(score))

    state.qualified += 1
    record["qualified"] = state.qualified

    qualify_frames = config.enrollment.qualify_frames
    if state.qualified < qualify_frames:
        return record

    best_sid, best_score = _pick_best_voted_sid(state, qualify_frames)
    if best_sid is not None:
        state.locked_sid = best_sid
        state.decided = True
        record["sid"] = best_sid
        record["similarity_score"] = best_score

    return record
