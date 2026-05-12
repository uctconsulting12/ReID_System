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


def _sid_dwell_key(sid) -> str | None:
    """Map a record's ``sid`` to the key used in ``dwell_by_sid`` (``G<n>``).
    Returns ``None`` for unidentified tracks."""
    if sid == UNKNOWN_LABEL:
        return None
    if isinstance(sid, str):
        return sid
    return f"G{int(sid)}"


def _put_centered_text(frame, text, center_xy, *, color, font_scale=0.7, thickness=2, pad=4):
    """Draw ``text`` filled-rect-backed and centered on ``center_xy``."""
    cx, cy = center_xy
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = int(cx - tw / 2)
    y = int(cy + th / 2)
    cv2.rectangle(
        frame,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        color, -1,
    )
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)


def draw_overlay(
    frame, records, fps, sid_count, cam_label: str, mode_label: str, config: AppConfig,
    *,
    dwell_by_sid: dict[str, str] | None = None,
    avg_dwell: str = "00:00:00",
    occupancy: int = 0,
):
    """Annotate the frame.

    Per detection:
      * coloured bbox
      * ``ID <n>`` (or ``Identifying...``) centered inside the bbox
      * dwell timer ``HH:MM:SS`` above the bbox top edge

    Per frame (top bar): ``Cam <id> | Avg Dwell: HH:MM:SS | Occupancy: N``.
    """
    dwell_by_sid = dwell_by_sid or {}

    for rec in records:
        x1, y1, x2, y2 = [int(c) for c in rec["bbox"]]
        sid = rec["sid"]
        color = _color_for(sid)
        id_label = "Identifying..." if sid == UNKNOWN_LABEL else f"ID {sid}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        _put_centered_text(frame, id_label, (cx, cy),
                           color=color, font_scale=0.7, thickness=2)

        key = _sid_dwell_key(sid)
        timer = dwell_by_sid.get(key) if key is not None else None
        if timer:
            (tw, th), _ = cv2.getTextSize(timer, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            timer_cy = max(y1 - th, th + 4)
            _put_centered_text(frame, timer, (cx, timer_cy),
                               color=color, font_scale=0.6, thickness=2)

    _draw_frame_header(frame, cam_label, avg_dwell, occupancy)
    return frame


def _draw_frame_header(frame, cam_label: str, avg_dwell: str, occupancy: int):
    header = f"Cam {cam_label}  |  Avg Dwell: {avg_dwell}  |  Occupancy: {occupancy}"
    font_scale = 0.7
    thickness = 2
    pad = 8
    (tw, th), baseline = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    fh, fw = frame.shape[:2]
    bar_h = th + baseline + 2 * pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    x = max(pad, (fw - tw) // 2)
    y = pad + th
    cv2.putText(frame, header, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)


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
