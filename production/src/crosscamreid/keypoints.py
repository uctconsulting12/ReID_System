from __future__ import annotations

import numpy as np

from .config import GatingConfig


KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14

REQUIRED_KPS = (
    KP_LEFT_EYE,
    KP_RIGHT_EYE,
    KP_LEFT_SHOULDER,
    KP_RIGHT_SHOULDER,
    KP_LEFT_KNEE,
    KP_RIGHT_KNEE,
)
REGION_KPS = (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, KP_LEFT_KNEE, KP_RIGHT_KNEE)


def keypoint_gate(kp_conf: np.ndarray, gating_cfg: GatingConfig) -> bool:
    return all(kp_conf[i] >= gating_cfg.keypoint_conf_thresh for i in REQUIRED_KPS)


def torso_region_bbox(
    kp_xy: np.ndarray,
    kp_conf: np.ndarray,
    frame_shape,
    gating_cfg: GatingConfig,
) -> tuple[int, int, int, int] | None:
    if not all(kp_conf[i] >= gating_cfg.keypoint_conf_thresh for i in REGION_KPS):
        return None

    pts = kp_xy[list(REGION_KPS)]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)

    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    px = w * gating_cfg.region_pad_frac
    py = h * gating_cfg.region_pad_frac

    x1 -= px
    y1 -= py
    x2 += px
    y2 += py

    fh, fw = frame_shape[:2]
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(fw, int(round(x2)))
    y2 = min(fh, int(round(y2)))

    if (x2 - x1) < gating_cfg.min_region_side or (y2 - y1) < gating_cfg.min_region_side:
        return None

    return x1, y1, x2, y2
