"""Image-only pixel descriptors for the Level 1 recognition comparison.

These baselines have no fitted parameters. Normalization uses each individual
image, so gallery/query descriptors never depend on a split's other images.
The returned float32 descriptor cache costs four bytes per pixel channel;
storing the original raw RGB image requires only one byte per channel.
"""

from __future__ import annotations

import numpy as np

variants = ("raw", "normalized")


def encode(images: np.ndarray, variant: str = "raw") -> np.ndarray:
    """Flatten uint8 NHWC RGB observations into independent float32 rows.

    ``raw`` scales channels to [0, 1]. ``normalized`` additionally subtracts
    the image-wide mean and divides by its standard deviation, floored at
    1e-6. Squared Euclidean descriptor distances therefore reproduce the
    existing pixel baselines' retrieval rankings. Constant images normalize
    to zero and cannot be distinguished by this baseline.

    No pose, scene identity, label, or collection statistics are accepted.
    The input is never modified, including when it is a noncontiguous view.
    """
    if not isinstance(variant, str) or variant not in variants:
        raise ValueError(f"Unknown pixel descriptor variant: {variant!r}")
    if (
        not isinstance(images, np.ndarray)
        or images.dtype != np.uint8
        or images.ndim != 4
        or images.shape[-1] != 3
        or min(images.shape[1:3]) < 1
    ):
        raise ValueError("Expected uint8 RGB images with shape (N, H, W, 3) and H, W > 0")
    dimensions = int(np.prod(images.shape[1:]))
    values = images.astype(np.float32).reshape(len(images), dimensions)
    values /= np.float32(255.0)
    if variant == "normalized" and len(values):
        constant = np.ptp(values, axis=1) == 0
        values -= values.mean(axis=1, keepdims=True)
        values /= np.maximum(values.std(axis=1, keepdims=True), np.float32(1e-6))
        # Float32 summation can leave a residual mean even for constant pixels.
        values[constant] = 0.0
    return values
