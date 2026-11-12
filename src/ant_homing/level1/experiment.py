"""Training and held-out evaluation for the retained Level 1 visual-memory models."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ant_homing.artifacts import new_output, provenance, write_json

from . import infonce
from .data import load_dataset
from .evaluation import (
    evaluate_split,
    observation_latency,
    paired_scene_interval,
    validation_threshold,
)
from .pixels import encode as pixel_encode

CANONICAL_NEGATIVE_GUARD_M = 0.9
ACTUAL_NEGATIVE_GUARD_M = 0.5


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def place_batches(
    arrays: dict, epoch: int, seed: int, places_per_batch: int = 16
) -> list[np.ndarray]:
    """Build within-scene place batches with an explicit spatial negative guard."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    scenes = np.unique(arrays["gallery_scene_ids"][arrays["gallery_splits"] == "train"])
    batches = []
    for scene in rng.permutation(scenes):
        all_rows = np.flatnonzero(
            (arrays["gallery_scene_ids"] == scene) & (arrays["gallery_splits"] == "train")
        )
        remaining = list(rng.permutation(all_rows))
        while remaining:
            selected = [remaining.pop(0)]
            for candidate in remaining.copy():
                separation = np.linalg.norm(
                    arrays["gallery_positions"][selected, :2]
                    - arrays["gallery_positions"][candidate, :2],
                    axis=1,
                )
                if np.all(separation >= CANONICAL_NEGATIVE_GUARD_M):
                    selected.append(candidate)
                    remaining.remove(candidate)
                    if len(selected) == places_per_batch:
                        break
            if len(selected) == 1:
                support = [
                    row
                    for row in all_rows
                    if row != selected[0]
                    and np.linalg.norm(
                        arrays["gallery_positions"][row, :2]
                        - arrays["gallery_positions"][selected[0], :2]
                    )
                    >= CANONICAL_NEGATIVE_GUARD_M
                ]
                if not support:
                    raise ValueError("Scene lacks separated places for valid negatives")
                selected.append(int(support[0]))
            batch = np.asarray(selected, dtype=np.int64)
            positions = arrays["observation_positions"][batch, :, :2]
            for left in range(len(batch)):
                for right in range(left):
                    distances = np.linalg.norm(
                        positions[left, :, None] - positions[right, None, :], axis=-1
                    )
                    if np.min(distances) <= ACTUAL_NEGATIVE_GUARD_M:
                        raise ValueError("Training batch contains ambiguous spatial negatives")
            batches.append(batch)
    return batches


def tensor(images: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2))).float() / 255


def neural_features(module, model, images: np.ndarray) -> np.ndarray:
    model.eval()
    features = []
    with torch.inference_mode():
        for start in range(0, len(images), 128):
            features.append(module.encode(model, tensor(images[start : start + 128])).numpy())
    return np.concatenate(features)


def train_one(
    arrays: dict,
    metadata: dict,
    output: Path,
    *,
    seed: int,
    epochs: int = 20,
    learning_rate: float = 0.001,
) -> dict:
    """Train one seeded InfoNCE encoder and calibrate its validation threshold."""
    torch.manual_seed(seed)
    model = infonce.build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    checkpoint = output / "model.pt"
    history, best = [], -np.inf
    started = perf_counter()
    for epoch in range(epochs):
        model.train()
        losses, exposures = [], 0
        batches = place_batches(arrays, epoch, seed)
        for batch_index, rows in enumerate(batches):
            images = arrays["observations"][rows].copy()
            visit_rng = np.random.default_rng(
                np.random.SeedSequence([seed, epoch, batch_index, 583])
            )
            for place in range(len(images)):
                images[place] = images[place, visit_rng.permutation(images.shape[1])]
            places, visits = images.shape[:2]
            batch = tensor(images.reshape(-1, *images.shape[2:])).reshape(
                places, visits, 3, 32, 128
            )
            loss = infonce.training_loss(model, batch)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            exposures += places * visits
        validation = evaluate_split(
            arrays,
            "val",
            lambda x: neural_features(infonce, model, x),
            tolerance=metadata["retrieval_tolerance_m"],
        )
        metric = validation["metrics"]["recall_at_1"]
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "updates": len(batches),
                "image_exposures": exposures,
                "validation": validation["metrics"],
            }
        )
        if metric > best:
            best = metric
            torch.save(
                {
                    "format_version": 1,
                    "module": infonce.NAME,
                    "seed": seed,
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "dataset_sha256": metadata["data_sha256"],
                    "scene_seeds": metadata["scene_seeds"],
                    "image_shape": metadata["image_shape"],
                    "camera_convention": metadata["camera_convention"],
                },
                checkpoint,
            )
        print(
            json.dumps(
                {
                    "method": infonce.NAME,
                    "seed": seed,
                    "epoch": epoch + 1,
                    "loss": history[-1]["loss"],
                    "val_recall_at_1": metric,
                }
            ),
            flush=True,
        )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    validation = evaluate_split(
        arrays,
        "val",
        lambda x: neural_features(infonce, model, x),
        tolerance=metadata["retrieval_tolerance_m"],
    )
    report = {
        "method": infonce.NAME,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "selected_epoch": saved["epoch"],
        "checkpoint_selection": "highest validation spatial Recall@1; earliest tie",
        "threshold": validation_threshold(validation["rows"]),
        "threshold_calibration": "validation familiar-query 95th distance percentile",
        "history": history,
        "training_seconds": perf_counter() - started,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "model_state_bytes": sum(p.numel() * p.element_size() for p in model.state_dict().values()),
        "descriptor_dimension": infonce.DESCRIPTOR_DIM,
        "loss_configuration": {"temperature": infonce.TEMPERATURE},
        "checkpoint_sha256": digest(checkpoint),
    }
    write_json(output / "training.json", report)
    return report


def train_all(data: Path, output: Path, *, seeds=(0, 1, 2), epochs=20, learning_rate=0.001) -> dict:
    if (
        epochs < 1
        or not seeds
        or len(set(seeds)) != len(seeds)
        or min(seeds) < 0
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("Invalid training budget or seeds")
    torch.set_num_threads(1)
    arrays, metadata = load_dataset(data)
    if metadata["image_shape"] != [32, 128, 3] or metadata["observations_per_place"] != 4:
        raise ValueError(
            "Level 1 comparison requires native 32x128 RGB and four observations per place"
        )
    target = new_output(output)
    validation = evaluate_split(
        arrays,
        "val",
        pixel_encode,
        tolerance=metadata["retrieval_tolerance_m"],
    )
    report = {
        "dataset_sha256": metadata["data_sha256"],
        "metadata_sha256": digest(data / "metadata.json"),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "seeds": list(seeds),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "negative_guard_m": ACTUAL_NEGATIVE_GUARD_M,
        "canonical_guard_m": CANONICAL_NEGATIVE_GUARD_M,
        "provenance": provenance(),
        "pixels": {"normalized": {"threshold": validation_threshold(validation["rows"])}},
        "runs": [],
    }
    for seed in seeds:
        run_dir = target / f"infonce-seed-{seed}"
        run_dir.mkdir()
        report["runs"].append(
            train_one(
                arrays,
                metadata,
                run_dir,
                seed=seed,
                epochs=epochs,
                learning_rate=learning_rate,
            )
        )
    report["frozen_at_utc"] = datetime.now(UTC).isoformat()
    write_json(target / "trained.json", report)
    return report


def evaluate_all(data: Path, run_dir: Path, output: Path) -> dict:
    """Score held-out test queries using frozen checkpoints and thresholds."""
    torch.set_num_threads(1)
    arrays, metadata = load_dataset(data)
    frozen = json.loads((run_dir / "trained.json").read_text())
    if frozen["dataset_sha256"] != metadata["data_sha256"] or not frozen.get("frozen_at_utc"):
        raise ValueError("Dataset mismatch or unfinished training")
    if frozen["metadata_sha256"] != digest(data / "metadata.json"):
        raise ValueError("Dataset metadata changed after training")
    target = new_output(output)
    latency_images = arrays["gallery"][np.flatnonzero(arrays["gallery_splits"] == "train")[:32]]

    norm_result = evaluate_split(
        arrays,
        "test",
        pixel_encode,
        tolerance=metadata["retrieval_tolerance_m"],
        threshold=frozen["pixels"]["normalized"]["threshold"],
    )
    norm_result.update(
        method="normalized",
        seed=None,
        parameter_count=0,
        model_state_bytes=0,
        descriptor_dimension=int(np.prod(metadata["image_shape"])),
        observation_latency=observation_latency(pixel_encode, latency_images),
    )
    results = [norm_result]

    for run in frozen["runs"]:
        checkpoint = run_dir / f"infonce-seed-{run['seed']}" / "model.pt"
        if digest(checkpoint) != run["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after freezing")
        model, saved = infonce.load_checkpoint(checkpoint)
        if (
            saved["dataset_sha256"] != metadata["data_sha256"]
            or saved["seed"] != run["seed"]
            or saved["scene_seeds"] != metadata["scene_seeds"]
        ):
            raise ValueError("Checkpoint provenance differs from the frozen run")
        warmup = np.flatnonzero(arrays["gallery_splits"] == "train")[:8]
        neural_features(infonce, model, arrays["gallery"][warmup])
        result = evaluate_split(
            arrays,
            "test",
            partial(neural_features, infonce, model),
            tolerance=metadata["retrieval_tolerance_m"],
            threshold=run["threshold"],
        )
        result.update(
            {
                key: run[key]
                for key in (
                    "method",
                    "seed",
                    "parameter_count",
                    "model_state_bytes",
                    "descriptor_dimension",
                )
            }
        )
        result["observation_latency"] = observation_latency(
            partial(neural_features, infonce, model), latency_images
        )
        results.append(result)

    reference = {
        int(scene): values["recall_at_1"] for scene, values in norm_result["scenes"].items()
    }
    infonce_runs = [row for row in results if row["method"] == "infonce"]
    mean_scenes = {
        scene: float(np.mean([row["scenes"][str(scene)]["recall_at_1"] for row in infonce_runs]))
        for scene in reference
    }
    report = {
        "dataset_sha256": metadata["data_sha256"],
        "test_scene_ids": metadata["scene_seeds"]["test"],
        "frozen_at_utc": frozen["frozen_at_utc"],
        "results": results,
        "paired_recall_vs_normalized": {"infonce": paired_scene_interval(mean_scenes, reference)},
        "reference_rgb_bytes": int(np.prod(metadata["image_shape"])),
        "cache_bytes_per_reference": {
            row["method"]: row["descriptor_dimension"] * 4 for row in results
        },
        "provenance": provenance(),
    }
    write_json(target / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--seeds", default="0,1,2")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        train_all(
            args.data,
            args.output,
            seeds=tuple(int(x) for x in args.seeds.split(",")),
            epochs=args.epochs,
        )
    else:
        report = evaluate_all(args.data, args.run_dir, args.output)
        print(json.dumps({"paired": report["paired_recall_vs_normalized"]}, indent=2))


if __name__ == "__main__":
    main()
