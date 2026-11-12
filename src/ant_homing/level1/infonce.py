"""Spatial CNN with paired, symmetric InfoNCE supervision.

A batch contains places and repeated observations of each place. Each observation
is a complete four-camera strip; camera panels are never positive examples of
one another. Saved Level 1 checkpoints live in ``models/level1/``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..simulation import CAMERA_CONVENTION

NAME = "infonce"
DESCRIPTOR_DIM = 64
TEMPERATURE = 0.1
IMAGE_SHAPE = (3, 32, 128)
MODELS_DIR = Path(__file__).resolve().parents[3] / "models" / "level1"


def build_model() -> nn.Module:
    """Construct the 42,728-parameter spatial CNN encoder."""
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
    """Average symmetric InfoNCE over disjoint repeated-observation pairs."""
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


def default_checkpoint(seed: int = 0) -> Path:
    """Return the path to a saved Level 1 InfoNCE checkpoint."""
    path = MODELS_DIR / f"infonce-seed-{seed}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Saved InfoNCE checkpoint not found: {path}")
    return path


def load_checkpoint(checkpoint: str | Path | int = 0) -> tuple[nn.Module, dict]:
    """Load a trained InfoNCE encoder and its metadata."""
    path = default_checkpoint(checkpoint) if isinstance(checkpoint, int) else Path(checkpoint)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 1 or saved.get("module") != NAME:
        raise ValueError("Unsupported InfoNCE checkpoint format or module")
    if saved.get("camera_convention") != CAMERA_CONVENTION:
        raise ValueError("Checkpoint camera convention is incompatible")
    model = build_model()
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    info = dict(saved)
    manifest_path = MODELS_DIR / "manifest.json"
    if manifest_path.is_file() and "threshold" not in info:
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest.get("models", {}).get("infonce", {}).get("checkpoints", []):
            if entry.get("seed") == info.get("seed"):
                info["threshold"] = entry.get("threshold")
                break
    return model, info


def load_image_scorer(
    checkpoint: str | Path | int = 0,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return an image-only scorer measuring squared 64D descriptor distance."""
    model, saved = load_checkpoint(checkpoint)
    expected_shape = tuple(saved.get("image_shape", (32, 128, 3)))

    def score(current_rgb: np.ndarray, reference_rgb: np.ndarray) -> float:
        if current_rgb.shape != expected_shape or reference_rgb.shape != expected_shape:
            raise ValueError(f"Checkpoint expects RGB images with shape {expected_shape}")
        pair = np.stack([current_rgb, reference_rgb]).transpose(0, 3, 1, 2)
        tensor = torch.from_numpy(np.ascontiguousarray(pair)).float() / 255.0
        with torch.inference_mode():
            descriptors = encode(model, tensor).numpy()
        difference = descriptors[0] - descriptors[1]
        return float(np.dot(difference, difference))

    return score
