"""The unchanged InfoNCE encoder trained with the reviewed MS loss recipe."""

from __future__ import annotations

import torch
from torch import nn

from .convap import LOSS_CONFIG, LOSS_REFERENCE, multi_similarity_loss
from .infonce import DESCRIPTOR_DIM, IMAGE_SHAPE, build_model, encode

NAME = "infonce_ms"
__all__ = [
    "DESCRIPTOR_DIM",
    "LOSS_CONFIG",
    "LOSS_REFERENCE",
    "NAME",
    "build_model",
    "encode",
    "training_loss",
]


def training_loss(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Use all repeat visits as positives, with the original MS miner/settings."""
    if (
        not isinstance(batch, torch.Tensor)
        or batch.ndim != 5
        or batch.shape[0] < 2
        or batch.shape[1] != 4
        or tuple(batch.shape[2:]) != IMAGE_SHAPE
    ):
        raise ValueError("Expected (P, 4, 3, 32, 128), with P >= 2")
    places, visits = batch.shape[:2]
    embeddings = encode(model, batch.reshape(places * visits, *IMAGE_SHAPE))
    labels = torch.arange(places, device=batch.device).repeat_interleave(visits)
    return multi_similarity_loss(embeddings, labels, **LOSS_CONFIG)
