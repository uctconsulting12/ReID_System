from __future__ import annotations

from .base import BaseReIDBackend
from .onnx_backend import ONNXReIDBackend
from .tensorrt_backend import TensorRTReIDBackend


def create_reid_backend(
    backend: str,
    onnx_path: str,
    tensorrt_engine_path: str | None,
    fastreid_root: str | None = None,
    fastreid_config: str | None = None,
    fastreid_weights: str | None = None,
    fastreid_device: str = "cuda",
    use_grayscale: bool = False,
) -> BaseReIDBackend:
    backend = backend.lower().strip()
    if backend == "onnxruntime":
        return ONNXReIDBackend(onnx_path, use_grayscale=use_grayscale)

    if backend == "tensorrt":
        if not tensorrt_engine_path:
            raise ValueError(
                "runtime.reid_backend=tensorrt requires models.reid_tensorrt_engine_path"
            )
        return TensorRTReIDBackend(tensorrt_engine_path, use_grayscale=use_grayscale)

    if backend == "fastreid":
        if not (fastreid_root and fastreid_config and fastreid_weights):
            raise ValueError(
                "runtime.reid_backend=fastreid requires models.fastreid_root, "
                "models.fastreid_config and models.fastreid_weights"
            )
        from .fastreid_backend import FastReIDBackend
        return FastReIDBackend(
            fastreid_root=fastreid_root,
            config_file=fastreid_config,
            weights_path=fastreid_weights,
            device=fastreid_device,
            use_grayscale=use_grayscale,
        )

    raise ValueError(f"Unsupported ReID backend: {backend}")
