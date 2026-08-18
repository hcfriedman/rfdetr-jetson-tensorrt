# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np
from PIL import Image as PILImage
from numpy.typing import NDArray
import onnxruntime as ort
from supervision import Detections

def _bilinear_resize_half_pixel(src: NDArray[np.float32], out_h: int, out_w: int) -> NDArray[np.float32]:
    """Numpy bilinear resize matching ``F.interpolate(mode="bilinear", align_corners=False)``.

    Half-pixel center convention with no antialias filter — the same convention
    ``torchvision.transforms.functional.resize(..., antialias=False)`` and
    ``RFDETR.predict()`` use. Serves as the torch-free fallback for both image
    preprocessing and mask decoding.

    Args:
        src: Source array of shape ``(K, src_h, src_w)``.
        out_h: Target height in pixels.
        out_w: Target width in pixels.

    Returns:
        Float32 array of shape ``(K, out_h, out_w)``.

    Note:
        Replaces ``PIL.Image.resize(BILINEAR)``, which applies an adaptive
        antialias filter when downscaling and a corner-aligned half-pixel
        convention, both of which diverge from ``F.interpolate``.
    """
    src_h, src_w = src.shape[-2], src.shape[-1]
    src_y = (np.arange(out_h, dtype=np.float32) + 0.5) * (src_h / out_h) - 0.5
    src_x = (np.arange(out_w, dtype=np.float32) + 0.5) * (src_w / out_w) - 0.5
    src_y = np.clip(src_y, 0.0, src_h - 1)
    src_x = np.clip(src_x, 0.0, src_w - 1)
    y0 = np.floor(src_y).astype(np.int64)
    x0 = np.floor(src_x).astype(np.int64)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    dy = (src_y - y0)[:, None]
    dx = (src_x - x0)[None, :]
    a = src[..., y0[:, None], x0[None, :]]
    b = src[..., y0[:, None], x1[None, :]]
    c = src[..., y1[:, None], x0[None, :]]
    d = src[..., y1[:, None], x1[None, :]]
    out = (1 - dy) * ((1 - dx) * a + dx * b) + dy * ((1 - dx) * c + dx * d)
    return np.asarray(out, dtype=np.float32)

def _preprocess_pil_to_nchw(
    image: PILImage.Image,
    height: int,
    width: int,
    channels: int = 3,
    ) -> NDArray[np.float32]:
    """Resize and normalise a PIL image to an ``(1, C, H, W)`` float32 NCHW tensor.

    Resizes with RFDETR.predict()'s exact convention — bilinear, half-pixel centers, antialias=False — via the 
    pure-NumPy _bilinear_resize_half_pixel (matches predict() up to float32 op-order noise; torchvision branch 
    removed so this module never imports torch, whose pip CUDA libs break Jetson ORT's CUDA init). PIL resize 
    is not used: both its BILINEAR and BICUBIC filters apply adaptive antialiasing when downscaling and diverge
    from predict(), shifting confidence scores. Normalises with ImageNet statistics:
    ``mean=[0.485, 0.456, 0.406]``, ``std=[0.229, 0.224, 0.225]``.

    Args:
        image: Input PIL image; any mode — converted to ``"RGB"`` (3-channel) or ``"L"`` (1-channel) internally.
        height: Target spatial height expected by the model.
        width: Target spatial width expected by the model.
        channels: Number of channels the model expects (``1`` for grayscale, ``3`` for RGB).

    Returns:
        Float32 ndarray of shape ``(1, channels, height, width)``.

    Examples:
        .. code-block:: python

            inp = _preprocess_pil_to_nchw(image, height=640, width=640)
    """
    _imagenet_mean = [0.485, 0.456, 0.406]
    _imagenet_std = [0.229, 0.224, 0.225]
    pil_mode = "L" if channels == 1 else "RGB"
    pil_img = image.convert(pil_mode)
    mean_list = [_imagenet_mean[i % 3] for i in range(channels)]
    std_list = [_imagenet_std[i % 3] for i in range(channels)]

    # Torch-free fallback: same antialias-free half-pixel bilinear as predict(), in NumPy.
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    if arr.ndim == 2:  # "L" → (H, W); needs (H, W, 1)
        arr = arr[:, :, np.newaxis]
    chw = _bilinear_resize_half_pixel(arr.transpose(2, 0, 1), height, width)
    mean = np.array(mean_list, dtype=np.float32)[:, np.newaxis, np.newaxis]
    std = np.array(std_list, dtype=np.float32)[:, np.newaxis, np.newaxis]
    chw = (chw - mean) / std
    return np.expand_dims(chw, axis=0).astype(np.float32)  # (1, C, H, W)


def _create_onnx_session(model_path, providers):
    session = ort.InferenceSession(str(model_path), providers=providers)
    print(f"ORT providers in use: {session.get_providers()}")
    return session

DEFAULT_NUM_SELECT = 300

def _select_topk_multiclass(
    scores_all: NDArray[np.floating], threshold: float, num_select: int = DEFAULT_NUM_SELECT
    ) -> tuple[NDArray[np.floating], NDArray[np.int64], NDArray[np.int64]]:
    """Select the top ``num_select`` query/class pairs, then threshold.

    Mirrors ``PostProcess._select_topk`` (flatten ``(Q, C)`` to ``Q * C``, take the top
    ``num_select`` scoring pairs in deterministic descending-score order) followed by the
    caller's own ``scores > threshold`` filter — ``PostProcess`` never bakes thresholding into
    ``_select_topk`` itself, it is always applied by the caller afterwards.

    Uses a deterministic lexicographic order: descending score, then ascending flattened
    query/class index. ``PostProcess._select_topk`` uses the same stable tie rule so exported
    inference remains reproducible when scores are equal.

    Args:
        scores_all: Per-query, per-class sigmoid probabilities, shape ``(Q, C)``.
        threshold: Confidence threshold; pairs at or below this score are dropped.
        num_select: Maximum number of query/class pairs to consider before thresholding.

    Returns:
        A ``(scores, labels, query_indices)`` tuple, each 1-D and sorted by descending score,
        containing only pairs that cleared ``threshold``. ``query_indices`` selects rows from
        box/mask outputs and, unlike a per-query argmax, can repeat when a query has more than one
        detection.
    """
    if scores_all.ndim != 2:
        raise ValueError(f"scores_all must have shape (Q, C); got {scores_all.shape}")
    if num_select < 0:
        raise ValueError(f"num_select must be non-negative; got {num_select}")

    num_queries, num_classes = scores_all.shape
    flat_scores = scores_all.reshape(-1)
    if num_select == 0 or flat_scores.size == 0:
        return (
            flat_scores[:0],
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    num_to_select = min(num_select, flat_scores.shape[0])
    flat_idx = np.arange(flat_scores.shape[0], dtype=np.int64)
    # PyTorch ranks NaNs ahead of finite values for descending argsort; preserve that ordering
    # so the subsequent ``> threshold`` filter drops the same malformed scores rather than
    # allowing a lower finite score to occupy the cap.
    sort_scores = np.where(np.isnan(flat_scores), np.inf, flat_scores)
    top_idx = np.lexsort((flat_idx, -sort_scores))[:num_to_select]

    topk_scores = flat_scores[top_idx]
    topk_query = top_idx // num_classes
    topk_labels = top_idx % num_classes

    keep = topk_scores > threshold
    return topk_scores[keep], topk_labels[keep], topk_query[keep]

def _decode_raw_outputs(
    raw_outputs: list,
    output_names: list[str],
    original_size: tuple[int, int],
    threshold: float = 0.3,
    num_select: int | None = None
        ) -> Detections:
    """Vendored from ``_run_inference`` (decode portion only).

    Args:
        raw_outputs: Arrays returned by ``session.run`` — RF-DETR exports
            produce ``dets`` (1, Q, 4) normalised cxcywh boxes and
            ``labels`` (1, Q, num_classes + 1) logits.
        output_names: Output names from the session, in run order
            (``[o.name for o in session.get_outputs()]``). Used to identify
            which array is boxes vs logits; shape-based fallback if the
            names don't match.
        original_size: ``(width, height)`` of the source image; boxes are
            scaled into this pixel space.
        threshold: Confidence threshold; detections at or below it are dropped.
        num_select: Maximum query/class pairs considered before thresholding.
            ``None`` uses the model's query count.

    Returns:
        ``supervision.Detections`` with pixel-space ``xyxy`` boxes,
        ``confidence`` scores, and ``class_id`` labels.
    """

    # RF-DETR ONNX output names: "dets" = pred_boxes, "labels" = pred_logits.
    # Match by name so the code is robust to output reordering.
    boxes_idx = next((i for i, name in enumerate(output_names) if "dets" in name), None)
    logits_idx = next((i for i, name in enumerate(output_names) if "labels" in name), None)
    if boxes_idx is None or logits_idx is None:
        # Fall back to shape-based matching: boxes (*, 4) and logits (*, num_classes+1).
        print(
            f"Name-based ONNX output matching failed (available names: {output_names}). Falling back to shape-based matching."
        )
        shape_boxes_candidates = [
            i for i, arr_out in enumerate(raw_outputs) if arr_out.ndim == 3 and arr_out.shape[-1] == 4
        ]
        shape_logits_candidates = [
            i for i, arr_out in enumerate(raw_outputs) if arr_out.ndim == 3 and arr_out.shape[-1] != 4
        ]
        if len(shape_boxes_candidates) == 1 and len(shape_logits_candidates) == 1:
            boxes_idx = shape_boxes_candidates[0]
            logits_idx = shape_logits_candidates[0]
        elif len(raw_outputs) == 2:
            # Ambiguous shapes (e.g. num_classes==3 → logits dim==4 == boxes dim).
            # ONNX preserves output order: index 0 = dets (boxes), index 1 = labels (logits).
            print(
                "Shape-based ONNX output matching is ambiguous (both outputs have last dim==4, "
                "which happens when num_classes==3).  Falling back to positional order: "
                "output 0 = boxes ('dets'), output 1 = logits ('labels').  "
                "If detections look wrong, inspect output names with _create_onnx_session()."
            )
            boxes_idx = 0
            logits_idx = 1
        else:
            available_shapes = [list(arr_out.shape) for arr_out in raw_outputs]
            raise ValueError(
                f"Shape-based ONNX output matching failed. Expected exactly one rank-3 tensor with "
                f"last dim == 4 (boxes) and one rank-3 tensor with last dim != 4 (logits). "
                f"Available output shapes: {available_shapes}"
            )

    boxes_cwh = raw_outputs[boxes_idx][0]  # (Q, 4) normalised cxcywh
    # Drop last logit column: RF-DETR adds +1 to num_classes (no-object slot, criterion.py:323).
    # Keeping it causes class_id == len(class_names) → IndexError at display time.
    logits = raw_outputs[logits_idx][0, :, :-1]  # (Q, num_classes)

    one = np.asarray(1, dtype=logits.dtype)
    scores_all = one / (one + np.exp(-logits.clip(-88, 88)))
    # Flatten (Q, C) to Q*C query/class pairs and take the top-scoring ones before thresholding —
    # mirrors PostProcess._select_topk. A per-query argmax (the previous approach) keeps at most
    # one class per query, silently dropping legitimate detections whenever a query scores above
    # threshold on more than one class; see _topk.py for why that happens routinely here.
    selection_cap = boxes_cwh.shape[0] if num_select is None else num_select
    scores, cls, query_idx = _select_topk_multiclass(scores_all, threshold, num_select=selection_cap)

    cx, cy, bw, bh = boxes_cwh[query_idx].T
    ow, oh = original_size
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    xyxy *= np.array([ow, oh, ow, oh], dtype=np.float32)

    return Detections(xyxy=xyxy, confidence=scores, class_id=cls.astype(int))
