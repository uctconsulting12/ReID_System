from __future__ import annotations

import numpy as np


class TIDState:
    __slots__ = (
        "qualified",
        "locked_sid",
        "match_votes",
        "embedding_buffer",
        "decided",
    )

    def __init__(self):
        self.qualified: int = 0
        self.locked_sid: int | None = None
        self.match_votes: dict[int, list[float]] = {}
        self.embedding_buffer: list[np.ndarray] = []
        self.decided: bool = False


class TIDStateManager:
    def __init__(
        self,
        max_missing_frames: int = 2,
        recover_iou_threshold: float = 0.5,
    ):
        self._states: dict[int, TIDState] = {}
        self._raw_to_stable: dict[int, int] = {}
        self._last_bbox: dict[int, tuple[float, float, float, float]] = {}
        self._missing: dict[int, int] = {}
        self._next_stable_tid: int = 1
        self.max_missing_frames = max(0, int(max_missing_frames))
        self.recover_iou_threshold = float(recover_iou_threshold)

    def get(self, tid: int) -> TIDState:
        state = self._states.get(tid)
        if state is None:
            state = TIDState()
            self._states[tid] = state
            if tid >= self._next_stable_tid:
                self._next_stable_tid = tid + 1
        return state

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        if denom <= 0.0:
            return 0.0
        return inter / denom

    def _new_stable_tid(self, used: set[int]) -> int:
        while self._next_stable_tid in used or self._next_stable_tid in self._states:
            self._next_stable_tid += 1
        tid = self._next_stable_tid
        self._next_stable_tid += 1
        return tid

    def remap_tids(self, raw_tids: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Build stable TIDs for the current frame.

        If the tracker emits a new raw ID after a short miss, re-link it to the
        recently lost stable ID with highest IoU above threshold.
        """
        n = len(raw_tids)
        if n == 0:
            self._raw_to_stable = {}
            return raw_tids.astype(int)

        stable = np.full(n, -1, dtype=int)
        taken: set[int] = set()
        next_raw_to_stable: dict[int, int] = {}

        # 1) Keep existing raw->stable mappings when available.
        unresolved: list[int] = []
        for i in range(n):
            raw_tid = int(raw_tids[i])
            mapped = self._raw_to_stable.get(raw_tid)
            if mapped is not None and mapped not in taken:
                stable[i] = mapped
                taken.add(mapped)
                next_raw_to_stable[raw_tid] = mapped
            else:
                unresolved.append(i)

        # 2) Recover from short misses using IoU with recently lost tracks.
        candidates = [
            sid for sid, miss in self._missing.items()
            if 0 < miss <= self.max_missing_frames and sid in self._last_bbox and sid not in taken
        ]
        if unresolved and candidates and self.recover_iou_threshold > 0.0:
            pairs: list[tuple[float, int, int]] = []
            for i in unresolved:
                b = boxes[i]
                bbox = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                for sid in candidates:
                    iou = self._iou(bbox, self._last_bbox[sid])
                    if iou >= self.recover_iou_threshold:
                        pairs.append((iou, i, sid))
            pairs.sort(reverse=True, key=lambda x: x[0])
            used_idx: set[int] = set()
            used_sid: set[int] = set()
            for _, i, sid in pairs:
                if i in used_idx or sid in used_sid or stable[i] != -1 or sid in taken:
                    continue
                raw_tid = int(raw_tids[i])
                stable[i] = sid
                taken.add(sid)
                next_raw_to_stable[raw_tid] = sid
                used_idx.add(i)
                used_sid.add(sid)

        # 3) Unresolved detections keep raw ID when possible; else allocate fresh stable ID.
        for i in unresolved:
            if stable[i] != -1:
                continue
            raw_tid = int(raw_tids[i])
            sid = raw_tid
            if sid in taken:
                sid = self._new_stable_tid(taken)
            stable[i] = sid
            taken.add(sid)
            next_raw_to_stable[raw_tid] = sid

        self._raw_to_stable = next_raw_to_stable
        return stable

    def forget(self, alive: set[int], alive_boxes: dict[int, tuple[float, float, float, float]] | None = None) -> None:
        alive_boxes = alive_boxes or {}
        for tid in list(self._states.keys()):
            if tid in alive:
                self._missing[tid] = 0
                if tid in alive_boxes:
                    self._last_bbox[tid] = alive_boxes[tid]
                continue
            misses = self._missing.get(tid, 0) + 1
            self._missing[tid] = misses
            if misses > self.max_missing_frames:
                self._states.pop(tid, None)
                self._missing.pop(tid, None)
                self._last_bbox.pop(tid, None)
