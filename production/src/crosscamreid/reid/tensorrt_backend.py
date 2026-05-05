from __future__ import annotations

import numpy as np

from .base import BaseReIDBackend


class TensorRTReIDBackend(BaseReIDBackend):
    """
    TensorRT runtime backend that consumes a serialized engine file.

    Notes:
    - Expects the engine to have exactly one input and one output tensor.
    - Input tensor shape must be compatible with NCHW (1, 3, 224, 224).
    """

    def __init__(self, engine_path: str, use_grayscale: bool = False):
        self.use_grayscale = use_grayscale
        try:
            import tensorrt as trt
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT backend requested, but tensorrt/pycuda is not installed."
            ) from exc

        self._trt = trt
        self._cuda = cuda

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        print(f"[ReID] Loading TensorRT engine: {engine_path}")
        with open(engine_path, "rb") as handle:
            engine = runtime.deserialize_cuda_engine(handle.read())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.engine = engine
        self.context = engine.create_execution_context()
        self.stream = cuda.Stream()

        self._setup_bindings()
        print(f"[ReID] TensorRT backend ready. Embedding dim: {self._dim}")

    def _setup_bindings(self) -> None:
        trt = self._trt
        engine = self.engine
        context = self.context

        if hasattr(engine, "num_io_tensors"):
            names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
            input_names = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            output_names = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

            if len(input_names) != 1 or len(output_names) != 1:
                raise RuntimeError("TensorRT engine must expose 1 input and 1 output tensor.")

            self.input_name = input_names[0]
            self.output_name = output_names[0]

            input_shape = tuple(context.get_tensor_shape(self.input_name))
            target_shape = (1, 3, self.input_hw[0], self.input_hw[1])
            if -1 in input_shape:
                context.set_input_shape(self.input_name, target_shape)

            in_shape = tuple(context.get_tensor_shape(self.input_name))
            out_shape = tuple(context.get_tensor_shape(self.output_name))
            if -1 in in_shape or -1 in out_shape:
                raise RuntimeError("TensorRT input/output shapes are not fully resolved.")

            in_dtype = trt.nptype(engine.get_tensor_dtype(self.input_name))
            out_dtype = trt.nptype(engine.get_tensor_dtype(self.output_name))

            self.host_input = np.empty(in_shape, dtype=in_dtype)
            self.host_output = np.empty(out_shape, dtype=out_dtype)
            self.dev_input = self._cuda.mem_alloc(self.host_input.nbytes)
            self.dev_output = self._cuda.mem_alloc(self.host_output.nbytes)

            context.set_tensor_address(self.input_name, int(self.dev_input))
            context.set_tensor_address(self.output_name, int(self.dev_output))
        else:
            raise RuntimeError(
                "Unsupported TensorRT Python API version. Please use TensorRT 8.6+."
            )

        self._dim = int(np.prod(self.host_output.shape))

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        inp = self._preprocess(frame, bbox)
        if inp is None:
            return None

        np.copyto(self.host_input, inp)
        self._cuda.memcpy_htod_async(self.dev_input, self.host_input, self.stream)
        if not self.context.execute_async_v3(stream_handle=self.stream.handle):
            return None
        self._cuda.memcpy_dtoh_async(self.host_output, self.dev_output, self.stream)
        self.stream.synchronize()
        return self._postprocess(self.host_output)

