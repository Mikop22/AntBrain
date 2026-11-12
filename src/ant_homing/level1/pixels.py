"""Normalized pixel descriptors for visual-memory retrieval and local homing.

Each observation is z-score normalized independently across its own pixels,
removing uniform affine brightness and contrast shifts without fitted parameters.
"""

from __future__ import annotations

import numpy as np

NAME = "normalized"


def encode(images: np.ndarray) -> np.ndarray:
    """Flatten uint8 NHWC RGB observations into z-score normalized float32 rows."""
    if (
        not isinstance(images, np.ndarray)
        or images.dtype != np.uint8
        or images.ndim != 4
        or images.shape[-1] != 3
        or min(images.shape[1:3]) < 1
    ):
        raise ValueError("Expected uint8 RGB images with shape (N, H, W, 3) and H, W > 0")
    dimensions = int(np.prod(images.shape[1:]))
    values = images.astype(np.float32).reshape(len(images), dimensions) / np.float32(255.0)
    if len(values):
        constant = np.ptp(values, axis=1) == 0
        values -= values.mean(axis=1, keepdims=True)
        values /= np.maximum(values.std(axis=1, keepdims=True), np.float32(1e-6))
        values[constant] = 0.0
    return values


def normalized_image_distance(current: np.ndarray, reference: np.ndarray) -> float:
    """Mean squared difference between two z-score normalized RGB images."""
    if (
        not isinstance(current, np.ndarray)
        or not isinstance(reference, np.ndarray)
        or current.dtype != np.uint8
        or reference.dtype != np.uint8
        or current.ndim != 3
        or current.shape != reference.shape
        or current.shape[-1] != 3
    ):
        raise ValueError("Expected matching uint8 HWC RGB images")
    pair = encode(np.stack([current, reference]))
    difference = pair[0] - pair[1]
    return float(np.mean(difference * difference))
