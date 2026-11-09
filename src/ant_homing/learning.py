"""Small image-only contrastive model and honest held-out retrieval baselines.

PyTorch is optional until training or a neural scorer is requested. This module
does not claim that retrieval accuracy establishes a working homing controller.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .dataset import load_dataset


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Neural features require the ML extra: uv sync --extra ml.") from exc
    return torch


def _build_encoder(embedding_dim: int = 64):
    torch = _torch()
    nn = torch.nn
    return nn.Sequential(
        nn.Conv2d(3, 12, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(12, 24, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(24, 32, 3, stride=2, padding=1),
        nn.ReLU(),
        # Retain coarse spatial layout; global pooling would discard bearings.
        nn.AdaptiveAvgPool2d((2, 8)),
        nn.Flatten(),
        nn.Linear(32 * 2 * 8, embedding_dim),
    )


def _image_tensor(images: np.ndarray):
    torch = _torch()
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Expected an NHWC uint8 RGB image batch.")
    return torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2))).float() / 255.0


def _embed(model, images: np.ndarray, batch_size: int = 128) -> np.ndarray:
    torch = _torch()
    model.eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            values = model(_image_tensor(images[start : start + batch_size]))
            batches.append(torch.nn.functional.normalize(values, dim=1).cpu().numpy())
    return np.concatenate(batches)


def _pixel_features(images: np.ndarray, normalized: bool) -> np.ndarray:
    values = images.astype(np.float32).reshape(len(images), -1) / 255.0
    if normalized:
        values -= values.mean(axis=1, keepdims=True)
        values /= np.maximum(values.std(axis=1, keepdims=True), 1e-6)
    return values


def _distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # The expanded quadratic form avoids allocating N x N x image_size arrays.
    return np.maximum(
        (left * left).sum(1)[:, None] + (right * right).sum(1)[None, :] - 2 * left @ right.T, 0
    )


def _retrieval_metrics(arrays: dict, split: str, feature_fn: Callable, tolerance: float) -> dict:
    errors, correct = [], []
    scenes = np.unique(arrays["scene_ids"][arrays["splits"] == split])
    if not len(scenes):
        raise ValueError(f"No samples in split {split!r}.")
    for scene in scenes:
        rows = np.flatnonzero((arrays["scene_ids"] == scene) & (arrays["splits"] == split))
        features_g = feature_fn(arrays["gallery"][rows])
        features_q = feature_fn(arrays["query"][rows])
        selected = np.argmin(_distances(features_q, features_g), axis=1)
        predicted = arrays["gallery_poses"][rows[selected], :2]
        actual = arrays["query_poses"][rows, :2]
        errors.extend(np.linalg.norm(predicted - actual, axis=1).tolist())
        correct.extend((selected == np.arange(len(rows))).tolist())
    values = np.asarray(errors)
    return {
        "query_count": len(errors),
        "scene_count": len(scenes),
        "top1_location_accuracy": float(np.mean(correct)),
        "recall_within_tolerance": float(np.mean(values <= tolerance)),
        "tolerance_m": tolerance,
        "median_location_error_m": float(np.median(values)),
        "mean_location_error_m": float(np.mean(values)),
    }


def train_model(
    dataset: str | Path,
    output: str | Path,
    *,
    epochs: int = 8,
    batch_size: int = 32,
    seed: int = 0,
    learning_rate: float = 0.002,
) -> dict:
    """Train with within-scene negatives; select a checkpoint using validation only."""
    if (
        epochs < 1
        or batch_size < 2
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
        or seed < 0
    ):
        raise ValueError(
            "Require epochs >= 1, batch_size >= 2, positive learning_rate and seed >= 0."
        )
    torch = _torch()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    arrays, metadata = load_dataset(dataset)
    target = Path(output)
    if (target / "model.pt").exists():
        raise FileExistsError(f"Checkpoint exists in {target}; choose a new output directory.")
    target.mkdir(parents=True, exist_ok=True)
    model = _build_encoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scenes = np.unique(arrays["scene_ids"][arrays["splits"] == "train"])
    history, best_score = [], -float("inf")
    for epoch in range(epochs):
        model.train()
        losses = []
        for scene in rng.permutation(scenes):
            rows = np.flatnonzero((arrays["scene_ids"] == scene) & (arrays["splits"] == "train"))
            rows = rng.permutation(rows)
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                if len(batch) < 2:
                    continue
                left = torch.nn.functional.normalize(
                    model(_image_tensor(arrays["gallery"][batch])), dim=1
                )
                right = torch.nn.functional.normalize(
                    model(_image_tensor(arrays["query"][batch])), dim=1
                )
                logits = left @ right.T / 0.1
                labels = torch.arange(len(batch))
                loss = (
                    torch.nn.functional.cross_entropy(logits, labels)
                    + torch.nn.functional.cross_entropy(logits.T, labels)
                ) / 2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
        if not losses:
            raise ValueError("No batches with within-scene negatives were available.")
        validation = _retrieval_metrics(
            arrays, "val", lambda imgs: _embed(model, imgs), metadata["retrieval_tolerance_m"]
        )
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation": validation}
        history.append(row)
        score = validation["top1_location_accuracy"]
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "format_version": 1,
                    "architecture": "spatial_cnn_v1",
                    "embedding_dim": 64,
                    "state_dict": model.state_dict(),
                    "dataset_sha256": metadata["data_sha256"],
                    "training_scene_seeds": np.unique(
                        arrays["scene_ids"][arrays["splits"] == "train"]
                    ).tolist(),
                    "validation_scene_seeds": np.unique(
                        arrays["scene_ids"][arrays["splits"] == "val"]
                    ).tolist(),
                    "image_shape": metadata["image_shape"],
                    "camera_convention": metadata["camera_convention"],
                    "epoch": epoch + 1,
                    "seed": seed,
                },
                target / "model.pt",
            )
    report = {
        "architecture": "spatial_cnn_v1",
        "objective": "symmetric within-scene InfoNCE; matched gallery/query locations are positives",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dataset_sha256": metadata["data_sha256"],
        "checkpoint_selection": "highest validation top1_location_accuracy; earliest tie",
        "history": history,
        "checkpoint": str(target / "model.pt"),
        "note": "Pipeline validation does not establish navigation performance or statistical superiority.",
    }
    (target / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _load_encoder(checkpoint: str | Path):
    from .simulation import CAMERA_CONVENTION

    torch = _torch()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 1 or saved.get("architecture") != "spatial_cnn_v1":
        raise ValueError("Unsupported neural checkpoint architecture or version.")
    if saved.get("camera_convention") != CAMERA_CONVENTION:
        raise ValueError("Checkpoint camera convention is incompatible; retrain the model.")
    model = _build_encoder(saved["embedding_dim"])
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    return model, saved


def load_image_scorer(checkpoint: str | Path) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return an image-only score; lower squared embedding distance is more familiar."""
    model, saved = _load_encoder(checkpoint)
    shape = tuple(saved["image_shape"])

    def score(current_rgb: np.ndarray, reference_rgb: np.ndarray) -> float:
        if current_rgb.shape != shape or reference_rgb.shape != shape:
            raise ValueError(f"Checkpoint expects RGB images with shape {shape}.")
        embeddings = _embed(model, np.stack([current_rgb, reference_rgb]))
        return float(np.square(embeddings[0] - embeddings[1]).sum())

    return score


def evaluate_retrieval(
    dataset: str | Path,
    checkpoint: str | Path | None = None,
    output: str | Path | None = None,
    *,
    split: str = "test",
) -> dict:
    """Compare representation methods with the same gallery/query evaluation."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val or test.")
    if output is not None and Path(output).exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    arrays, metadata = load_dataset(dataset)
    methods = {
        "raw_pixels": lambda imgs: _pixel_features(imgs, False),
        "normalized_pixels": lambda imgs: _pixel_features(imgs, True),
    }
    if checkpoint is not None:
        model, saved = _load_encoder(checkpoint)
        # Check actual membership even when the evaluation collection has the
        # training dataset hash: test status is a property of scene separation.
        evaluated = set(arrays["scene_ids"][arrays["splits"] == split].tolist())
        used = set(saved["training_scene_seeds"]) | set(saved["validation_scene_seeds"])
        if evaluated & used and split == "test":
            raise ValueError("Test scenes overlap checkpoint training/validation scenes.")
        if list(arrays["gallery"].shape[1:]) != saved["image_shape"]:
            raise ValueError("Checkpoint and evaluation image shapes differ.")
        methods["cnn"] = lambda imgs: _embed(model, imgs)
    report = {
        "split": split,
        "dataset_sha256": metadata["data_sha256"],
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "metrics": {
            name: _retrieval_metrics(arrays, split, feature_fn, metadata["retrieval_tolerance_m"])
            for name, feature_fn in methods.items()
        },
        "scope": "within-scene image retrieval; not a navigation success measurement",
    }
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
    return report
