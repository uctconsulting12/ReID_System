from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from .base import BaseReIDBackend


class FastReIDBackend(BaseReIDBackend):
    """ReID backend that wraps the fast-reid DefaultPredictor.

    Mean/std normalization is handled inside the fast-reid model
    (cfg.MODEL.PIXEL_MEAN / PIXEL_STD), so we feed raw float32 RGB
    pixels in [0, 255] — matching `demo/predictor.py:run_on_image`.
    """

    use_full_body = False

    def __init__(
        self,
        fastreid_root: str,
        config_file: str,
        weights_path: str,
        device: str = "cuda",
        use_grayscale: bool = False,
    ):
        self.use_grayscale = use_grayscale
        root = Path(fastreid_root).resolve()
        if not root.exists():
            raise FileNotFoundError(f"fast-reid root not found: {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        import torch
        from fastreid.config import get_cfg
        from fastreid.engine import DefaultPredictor

        cfg = get_cfg()
        cfg.merge_from_file(str(Path(config_file).resolve()))
        cfg.MODEL.WEIGHTS = str(Path(weights_path).resolve())
        if device == "cuda" and not torch.cuda.is_available():
            print("[ReID] CUDA not available, falling back to CPU.")
            device = "cpu"
        cfg.MODEL.DEVICE = device
        cfg.freeze()

        print(f"[ReID] Loading fast-reid: {weights_path}")
        print(f"[ReID] fast-reid config: {config_file}")
        self._cfg = cfg
        self._torch = torch
        self._predictor = DefaultPredictor(cfg)

        size_h, size_w = cfg.INPUT.SIZE_TEST
        self.input_hw = (int(size_h), int(size_w))

        h, w = self.input_hw
        dummy = torch.zeros((1, 3, h, w), dtype=torch.float32)
        with torch.no_grad():
            out = self._predictor(dummy)
        self._dim = int(out.flatten().shape[0])
        print(f"[ReID] Active device: {device}  Input HxW: {h}x{w}  Embedding dim: {self._dim}")

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        x1, y1, x2, y2 = [int(c) for c in bbox]
        fh, fw = frame.shape[:2]
        x1 = max(0, min(x1, fw - 1))
        y1 = max(0, min(y1, fh - 1))
        x2 = max(x1 + 1, min(x2, fw))
        y2 = max(y1 + 1, min(y2, fh))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 5:
            return None
# this is a try
        crop = self._apply_color_mode_bgr(crop)
        h, w = self.input_hw
        crop = crop[:, :, ::-1]  # BGR -> RGB
        crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)
        tensor = self._torch.as_tensor(
            crop.astype(np.float32).transpose(2, 0, 1)
        )[None]

        with self._torch.no_grad():
            feat = self._predictor(tensor)
        feat = feat.cpu().numpy()
        return self._postprocess(feat)
