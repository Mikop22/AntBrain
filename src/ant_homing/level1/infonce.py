"""The original spatial CNN with paired, symmetric InfoNCE supervision.

A batch contains places and repeated observations of each place. Each observation
is a complete four-camera strip; camera panels are never positive examples of
one another. This module defines the model and objective, not an optimizer.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

NAME = "infonce"
DESCRIPTOR_DIM = 64
TEMPERATURE = 0.1
IMAGE_SHAPE = (3, 32, 128)


def build_model() -> nn.Module:
    """Reproduce the existing encoder without importing its training pipeline."""
    return nn.Sequential(
        nn.Conv2d(3, 12, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(12, 24, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(24, 32, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((2, 8)),
        nn.Flatten(),
        nn.Linear(32 * 2 * 8, DESCRIPTOR_DIM),
    )


def _validate_images(images: torch.Tensor) -> None:
    if not isinstance(images, torch.Tensor):
        raise TypeError("Images must be a torch tensor")
    if images.ndim != 4 or tuple(images.shape[1:]) != IMAGE_SHAPE or len(images) == 0:
        raise ValueError("Expected a nonempty image batch with shape (N, 3, 32, 128)")
    if not images.is_floating_point():
        raise TypeError("Images must be floating point values on [0, 1]")
    if not bool(torch.isfinite(images).all()) or bool((images < 0).any() | (images > 1).any()):
        raise ValueError("Images must contain finite values on [0, 1]")


def encode(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return differentiable unit-length descriptors from image tensors only."""
    _validate_images(images)
    descriptors = model(images)
    if descriptors.shape != (len(images), DESCRIPTOR_DIM):
        raise ValueError("Encoder must produce one 64-dimensional descriptor per image")
    if not bool(torch.isfinite(descriptors).all()):
        raise ValueError("Encoder produced nonfinite descriptors")
    norms = torch.linalg.vector_norm(descriptors, dim=1)
    if bool((norms <= 1e-12).any()):
        raise ValueError("Encoder produced a zero-length descriptor")
    return F.normalize(descriptors, p=2, dim=1)


def training_loss(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Average symmetric InfoNCE over disjoint repeated-observation pairs.

    ``batch`` has shape (places, visits, 3, 32, 128), with at least two places
    and an even number of visits. For four visits, both (0, 1) and (2, 3) are
    evaluated so all approaches consume the same observations. The shared
    trainer constructs within-scene place batches; it also owns optimization.
    """
    if not isinstance(batch, torch.Tensor):
        raise TypeError("Training batch must be a torch tensor")
    if (
        batch.ndim != 5
        or batch.shape[0] < 2
        or batch.shape[1] < 2
        or batch.shape[1] % 2
        or tuple(batch.shape[2:]) != IMAGE_SHAPE
    ):
        raise ValueError("Expected (P, V, 3, 32, 128), with P >= 2 and even V >= 2")
    places, visits = batch.shape[:2]
    descriptors = encode(model, batch.reshape(places * visits, *IMAGE_SHAPE))
    descriptors = descriptors.reshape(places, visits, DESCRIPTOR_DIM)
    labels = torch.arange(places, device=descriptors.device)
    losses = []
    for visit in range(0, visits, 2):
        left, right = descriptors[:, visit], descriptors[:, visit + 1]
        logits = left @ right.T / TEMPERATURE
        losses.append((F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2)
    return torch.stack(losses).mean()
