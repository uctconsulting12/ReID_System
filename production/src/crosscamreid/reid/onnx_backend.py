from __future__ import annotations

import numpy as np
import onnxruntime as ort

from .base import BaseReIDBackend


class ONNXReIDBackend(BaseReIDBackend):
    def __init__(self, model_path: str, use_grayscale: bool = False):
        self.use_grayscale = use_grayscale
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print(f"[ReID] Loading ONNX: {model_path} (grayscale={use_grayscale})")
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"[ReID] Active provider: {self.session.get_providers()[0]}")

        h, w = self.input_hw
        dummy = np.zeros((1, 3, h, w), dtype=np.float32)
        out = self.session.run([self.output_name], {self.input_name: dummy})[0]
        self._dim = int(out.flatten().shape[0])
        print(f"[ReID] Input HxW: {h}x{w}  Embedding dim: {self._dim}")

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        inp = self._preprocess(frame, bbox)
        if inp is None:
            return None
        feat = self.session.run([self.output_name], {self.input_name: inp})[0]
        return self._postprocess(feat)

