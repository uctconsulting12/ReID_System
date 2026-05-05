from __future__ import annotations

import cv2
import math
import numpy as np

from .config import AppConfig
from .processor import UNKNOWN_LABEL

_COLORS = [
    (255, 80, 80),
    (80, 255, 80),
    (80, 80, 255),
    (255, 200, 50),
    (200, 50, 255),
    (50, 200, 255),
    (255, 120, 200),
    (120, 255, 150),
    (150, 120, 255),
    (255, 180, 100),
    (100, 255, 200),
    (180, 100, 255),
]
_UNKNOWN_COLOR = (160, 160, 160)


def _color_for(sid):
    if sid == UNKNOWN_LABEL:
        return _UNKNOWN_COLOR
    return _COLORS[(int(sid) - 1) % len(_COLORS)]


def draw_overlay(frame, records, fps, sid_count, cam_label: str, mode_label: str, config: AppConfig):
    """Minimal overlay: bbox per detection (color is unique per assigned ID)
    plus a single label — ``ID <n>`` for a locked SID, ``Identifying...``
    while the pipeline is still deciding.
    """
    for rec in records:
        x1, y1, x2, y2 = [int(c) for c in rec["bbox"]]
        sid = rec["sid"]
        color = _color_for(sid)
        label = "Identifying..." if sid == UNKNOWN_LABEL else f"ID {sid}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def combine_frames_grid(frames: list[np.ndarray | None], labels: list[str], total_width: int):
    if not frames:
        return np.zeros((540, total_width, 3), dtype=np.uint8)

    count = len(frames)
    cols = max(1, int(math.ceil(math.sqrt(count))))
    rows = int(math.ceil(count / cols))
    cell_w = max(320, total_width // cols)
    cell_h = max(240, int(cell_w * 9 / 16))

    def _fit(frame, label):
        if frame is None:
            blank = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            cv2.putText(blank, f"{label}: No Signal", (20, cell_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return blank
        h, w = frame.shape[:2]
        scale = min(cell_w / w, cell_h / h)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        resized = cv2.resize(frame, (nw, nh))
        canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        ox = (cell_w - nw) // 2
        oy = (cell_h - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = resized
        return canvas

    cells = [_fit(frame, labels[i]) for i, frame in enumerate(frames)]
    while len(cells) < rows * cols:
        cells.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

    row_images = []
    for r in range(rows):
        row_cells = cells[r * cols:(r + 1) * cols]
        row_images.append(np.hstack(row_cells))
    return np.vstack(row_images)
