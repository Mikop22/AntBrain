"""Camera-aware spatial descriptors and mined Multi-Similarity learning.

This is a small project-specific adaptation of Conv-AP, not the original
ResNet model. Each complete four-camera observation is one training example;
separate visits to a place provide its positive examples.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

NAME = "convap_ms"
DESCRIPTOR_DIM = 128
LOSS_CONFIG = {"alpha": 1.0, "beta": 50.0, "base": 0.0, "epsilon": 0.1}
LOSS_REFERENCE = {
    "library": "pytorch-metric-learning",
    "version": "2.3.0",
    "reduction": "mean_all_anchors",
    "source": "https://github.com/KevinMusgrave/pytorch-metric-learning/tree/v2.3.0",
}


def _validate_images(images: torch.Tensor) -> None:
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("Expected an NCHW image tensor")
    if tuple(images.shape[1:]) != (3, 32, 128) or not images.is_floating_point():
        raise ValueError("Expected floating RGB images with shape (N, 3, 32, 128)")
    if not torch.isfinite(images).all() or torch.any((images < 0) | (images > 1)):
        raise ValueError("Image values must be finite and in [0, 1]")


class CameraConvAP(nn.Module):
    """One shared camera encoder, ordered spatial features, and a compact head."""

    def __init__(self):
        super().__init__()
        self.camera_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.aggregation = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.projection = nn.Linear(4 * 32 * 2 * 2, DESCRIPTOR_DIM)

    def camera_features(self, images: torch.Tensor) -> torch.Tensor:
        """Preserve E,S,W,N order without convolving across camera seams."""
        _validate_images(images)
        count = images.shape[0]
        if count == 0:
            return images.new_empty((0, 4, 128))
        cameras = images.reshape(count, 3, 32, 4, 32)
        cameras = cameras.permute(0, 3, 1, 2, 4).reshape(count * 4, 3, 32, 32)
        features = self.aggregation(self.camera_encoder(cameras))
        return features.reshape(count, 4, -1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.camera_features(images)
        if images.shape[0] == 0:
            return images.new_empty((0, DESCRIPTOR_DIM))
        return F.normalize(self.projection(features.flatten(1)), p=2, dim=1)


def build_model() -> nn.Module:
    """Construct an untrained model; this function performs no optimization."""
    return CameraConvAP()


def encode(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Encode RGB only; callers control train/eval and gradient contexts."""
    _validate_images(images)
    return model(images)


def _mined_pair_masks(
    similarities: torch.Tensor, labels: torch.Tensor, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select informative pairs using the Multi-Similarity miner inequalities."""
    same_place = labels[:, None] == labels[None, :]
    positive = same_place & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    negative = ~same_place
    with torch.no_grad():
        maximum_negative = similarities.masked_fill(~negative, -torch.inf).max(dim=1).values
        minimum_positive = similarities.masked_fill(~positive, torch.inf).min(dim=1).values
        mined_positive = positive & (similarities < maximum_negative[:, None] + epsilon)
        mined_negative = negative & (similarities > minimum_positive[:, None] - epsilon)
    return mined_positive, mined_negative


def multi_similarity_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 50.0,
    base: float = 0.0,
    epsilon: float = 0.1,
) -> torch.Tensor:
    """Mined cosine Multi-Similarity loss, averaging all anchor losses.

    Pair formulas and mining follow the public pytorch-metric-learning
    v2.3.0 implementation; hyperparameters match the BoQ/SALAD VPR recipe. Its
    MeanReducer includes inactive anchors as zeros. Its GenericPairLoss guard
    also returns a differentiable zero when neither directed pair set contains
    more than one pair. This version is pinned because more recent library
    implementations changed which anchor rows reach the reducer.
    """
    if (
        not isinstance(embeddings, torch.Tensor)
        or embeddings.ndim != 2
        or not embeddings.is_floating_point()
        or not torch.isfinite(embeddings).all()
    ):
        raise ValueError("Embeddings must be a finite floating matrix")
    if (
        not isinstance(labels, torch.Tensor)
        or labels.ndim != 1
        or len(labels) != len(embeddings)
        or labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        or labels.device != embeddings.device
    ):
        raise ValueError("Integer labels must match embeddings in length and device")
    if (
        not all(math.isfinite(value) for value in (alpha, beta, base, epsilon))
        or min(alpha, beta) <= 0
        or epsilon < 0
    ):
        raise ValueError("Loss scales must be finite and positive; epsilon must be nonnegative")
    if len(embeddings) == 0:
        return embeddings.sum() * 0
    values = F.normalize(embeddings, p=2, dim=1)
    similarities = values @ values.T
    positive, negative = _mined_pair_masks(similarities, labels, epsilon)
    # GenericPairLoss v2.3.0 short-circuits when all four pair-index arrays
    # (positive anchors/partners, negative anchors/partners) have length <= 1.
    if positive.sum() <= 1 and negative.sum() <= 1:
        return embeddings.sum() * 0
    # Leading zero implements log(1 + sum(exp(...))) without overflow.
    zero = similarities.new_zeros((len(values), 1))
    positive_logits = (-alpha * (similarities - base)).masked_fill(~positive, -torch.inf)
    negative_logits = (beta * (similarities - base)).masked_fill(~negative, -torch.inf)
    positive_loss = torch.logsumexp(torch.cat((zero, positive_logits), dim=1), dim=1) / alpha
    negative_loss = torch.logsumexp(torch.cat((zero, negative_logits), dim=1), dim=1) / beta
    return (positive_loss + negative_loss).mean()


def training_loss(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Use repeat observations as positives and other same-scene places as negatives."""
    if not isinstance(batch, torch.Tensor) or batch.ndim != 5:
        raise ValueError("Expected a (places, observations, 3, 32, 128) tensor")
    places, visits = batch.shape[:2]
    if places < 2 or visits < 2:
        raise ValueError("At least two places and two observations per place are required")
    embeddings = encode(model, batch.flatten(0, 1))
    labels = torch.arange(places, device=batch.device).repeat_interleave(visits)
    return multi_similarity_loss(embeddings, labels, **LOSS_CONFIG)
