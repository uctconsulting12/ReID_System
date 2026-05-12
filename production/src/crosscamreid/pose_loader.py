"""
pose_loader.py
==============
Centralized YOLO pose-model loader that is robust to ``.engine`` files
exported without Ultralytics metadata.

Background
----------
``yolo export format=engine`` is supposed to embed a metadata blob (task,
class names, keypoint shape, stride, imgsz, ...) at the end of the engine
file. Older / hand-built engines and engines produced by some CLI flows
miss that blob. When ``AutoBackend`` finds no metadata it falls back to
the generic detect defaults: ``task="detect"`` and the 80 COCO class
names.

That alone would be survivable, but PosePredictor inherits
``postprocess`` from DetectionPredictor, and DetectionPredictor.postprocess
computes::

    nc = 0 if args.task == "detect" else len(self.model.names)

For ``task="pose"`` (which we force on the YOLO instance), this becomes
``nc = 80`` against pose's 56-channel output tensor and NMS hits::

    extra = preds.shape[1] - nc - 4   # = 56 - 80 - 4 = -28
    output = [torch.zeros((0, 6 + extra), ...)] * bs
    RuntimeError: zeros: Dimension size must be non-negative.

Fix: register an ``on_predict_start`` callback that overwrites the wrong
defaults *after* AutoBackend has built the runtime but *before* the first
postprocess() call. This keeps the engine usable without re-export.
"""

from __future__ import annotations

from ultralytics import YOLO


def _patch_pose_engine_metadata(predictor) -> None:
    """Force YOLO pose attributes regardless of engine metadata.

    Registered as an ``on_predict_start`` callback so it fires once the
    underlying ``AutoBackend`` exists (after ``setup_model``) and before
    the first ``postprocess`` invocation.
    """
    model = getattr(predictor, "model", None)
    if model is None:
        return

    model.task = "pose"
    names = getattr(model, "names", None)
    if names is None or len(names) != 1:
        model.names = {0: "person"}

    if not getattr(model, "kpt_shape", None):
        # Standard YOLO-pose layout: 17 COCO keypoints, (x, y, vis) per kp.
        model.kpt_shape = [17, 3]

    # Ensure the predictor's own task arg also says pose. PosePredictor
    # sets this in __init__, but a re-used predictor with stale overrides
    # can drift; pinning it here is a cheap belt-and-braces fix.
    if getattr(predictor, "args", None) is not None:
        predictor.args.task = "pose"


def load_pose_model(weights_path: str) -> YOLO:
    """Load a YOLO pose model that works for both ``.pt`` and ``.engine``.

    For ``.engine`` files we additionally register a metadata-patch
    callback so missing/stale embedded metadata cannot crash NMS at the
    first inference.
    """
    pose = YOLO(weights_path, task="pose")
    pose.add_callback("on_predict_start", _patch_pose_engine_metadata)
    return pose
