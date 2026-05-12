from __future__ import annotations

import threading

import numpy as np

from .base import BaseReIDBackend


class TensorRTReIDBackend(BaseReIDBackend):
    """
    TensorRT runtime backend that consumes a serialized engine file.

    Notes:
    - Expects the engine to have exactly one input and one output tensor.
    - Input tensor shape must be NCHW with N == 1; H, W are read from the
      engine so the shared preprocessor uses the correct crop size.

    Thread-safety:
    - The TRT execution context and its single set of host/device buffers
      are not safe to use concurrently, so inference is serialized with an
      instance lock.
    - pycuda binds a CUDA context to the thread that created it. ReID is
      called from per-camera worker threads (see PeopleCountingSession),
      not from the thread that built this backend, so we retain the
      device's primary context here and explicitly push/pop it around
      every CUDA call. Using ``pycuda.autoinit`` would only make the
      context current on the constructing thread, which is why the
      previous version failed at first inference from a worker thread
      ("invalid resource handle" / "no currently active context").
    """

    def __init__(self, engine_path: str, use_grayscale: bool = False):
        self.use_grayscale = use_grayscale
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT backend requested, but tensorrt/pycuda is not installed."
            ) from exc

        cuda.init()
        self._trt = trt
        self._cuda = cuda

        device = cuda.Device(0)
        # The primary context is reference-counted and shared across threads
        # via push/pop. Hold one reference for the lifetime of the backend.
        self._cuda_ctx = device.retain_primary_context()
        self._infer_lock = threading.Lock()

        self._cuda_ctx.push()
        try:
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
        finally:
            self._cuda_ctx.pop()

        print(f"[ReID] TensorRT backend ready. Embedding dim: {self._dim}")

    def _setup_bindings(self) -> None:
        trt = self._trt
        engine = self.engine
        context = self.context

        if not hasattr(engine, "num_io_tensors"):
            raise RuntimeError(
                "Unsupported TensorRT Python API version. Please use TensorRT 8.6+."
            )

        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        input_names = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        output_names = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

        if len(input_names) != 1 or len(output_names) != 1:
            raise RuntimeError(
                f"TensorRT engine must expose 1 input and 1 output tensor "
                f"(got inputs={input_names}, outputs={output_names})."
            )

        self.input_name = input_names[0]
        self.output_name = output_names[0]

        # Resolve dynamic shapes against the engine's optimization profile,
        # honouring the engine's own min/opt/max bounds when present.
        input_shape = tuple(context.get_tensor_shape(self.input_name))
        if -1 in input_shape:
            target_shape = self._resolve_dynamic_input_shape(input_shape)
            if not context.set_input_shape(self.input_name, target_shape):
                raise RuntimeError(
                    f"set_input_shape({target_shape}) rejected by engine "
                    f"(profile range may be incompatible)."
                )

        in_shape = tuple(context.get_tensor_shape(self.input_name))
        out_shape = tuple(context.get_tensor_shape(self.output_name))
        if -1 in in_shape or -1 in out_shape:
            raise RuntimeError(
                f"TensorRT shapes not fully resolved (input={in_shape}, output={out_shape})."
            )
        if len(in_shape) != 4 or in_shape[0] != 1 or in_shape[1] != 3:
            raise RuntimeError(
                f"Unexpected TensorRT input shape for ReID engine: {in_shape} "
                "(expected (1, 3, H, W))."
            )

        # Align the shared preprocessor with whatever crop size the engine
        # was built for (e.g. 256x128 for Market-1501 ReID, 224x224 for the
        # generic ResNet-50 ONNX zoo).
        self.input_hw = (int(in_shape[2]), int(in_shape[3]))

        in_dtype = trt.nptype(engine.get_tensor_dtype(self.input_name))
        out_dtype = trt.nptype(engine.get_tensor_dtype(self.output_name))

        # Pinned host buffers give faster + async-safe H2D/D2H transfers.
        self.host_input = self._cuda.pagelocked_empty(int(np.prod(in_shape)), dtype=in_dtype).reshape(in_shape)
        self.host_output = self._cuda.pagelocked_empty(int(np.prod(out_shape)), dtype=out_dtype).reshape(out_shape)
        self.dev_input = self._cuda.mem_alloc(self.host_input.nbytes)
        self.dev_output = self._cuda.mem_alloc(self.host_output.nbytes)

        context.set_tensor_address(self.input_name, int(self.dev_input))
        context.set_tensor_address(self.output_name, int(self.dev_output))

        self._dim = int(np.prod(self.host_output.shape))

    def _resolve_dynamic_input_shape(self, dynamic_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Build a concrete (1, 3, H, W) shape that satisfies the engine's
        optimization profile. Falls back to the BaseReIDBackend default
        (224x224) when the engine does not advertise a profile."""
        h_default, w_default = self.input_hw
        target = [
            1 if dim == -1 and i == 0 else
            3 if dim == -1 and i == 1 else
            h_default if dim == -1 and i == 2 else
            w_default if dim == -1 and i == 3 else
            int(dim)
            for i, dim in enumerate(dynamic_shape)
        ]

        engine = self.engine
        if engine.num_optimization_profiles > 0:
            try:
                min_s, opt_s, max_s = engine.get_tensor_profile_shape(self.input_name, 0)
                # Prefer the profile's "opt" shape — it is what the engine
                # was tuned for and is always within the min/max range.
                target = [int(v) for v in opt_s]
            except Exception:
                pass
        return tuple(target)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        inp = self._preprocess(frame, bbox)
        if inp is None:
            return None

        # Engines built in fp16 expose float16 input; the preprocessor
        # produces float32. Cast once here (cheap, contiguous) so the
        # np.copyto into a pinned host buffer never raises.
        if inp.dtype != self.host_input.dtype:
            inp = inp.astype(self.host_input.dtype, copy=False)

        with self._infer_lock:
            self._cuda_ctx.push()
            try:
                np.copyto(self.host_input, inp)
                self._cuda.memcpy_htod_async(self.dev_input, self.host_input, self.stream)
                if not self.context.execute_async_v3(stream_handle=self.stream.handle):
                    return None
                self._cuda.memcpy_dtoh_async(self.host_output, self.dev_output, self.stream)
                self.stream.synchronize()
                return self._postprocess(self.host_output)
            finally:
                self._cuda_ctx.pop()

    def __del__(self):
        # Free device memory under the same primary context that allocated
        # it; otherwise pycuda raises during interpreter shutdown.
        ctx = getattr(self, "_cuda_ctx", None)
        if ctx is None:
            return
        try:
            ctx.push()
            try:
                for attr in ("dev_input", "dev_output"):
                    buf = getattr(self, attr, None)
                    if buf is not None:
                        buf.free()
                        setattr(self, attr, None)
            finally:
                ctx.pop()
        except Exception:
            pass
