# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np
from PIL import Image as PILImage
from numpy.typing import NDArray
import onnxruntime as ort

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
