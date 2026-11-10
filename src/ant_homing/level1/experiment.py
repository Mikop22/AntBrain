"""Review-gated training and frozen held-out comparison of Level 1 methods."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ant_homing.artifacts import new_output, provenance, write_json

from .data import load_dataset
from .evaluation import (
    evaluate_split,
    observation_latency,
    paired_scene_interval,
    validation_threshold,
)
from .pixels import encode as pixel_encode

METHODS = ("infonce", "convap")
CANONICAL_NEGATIVE_GUARD_M = 0.9
ACTUAL_NEGATIVE_GUARD_M = 0.5
REVIEWED_SOURCE_PATHS = {
    f"src/ant_homing/level1/{name}.py"
    for name in ("__init__", "data", "pixels", "infonce", "convap", "evaluation", "experiment")
} | {"src/ant_homing/simulation.py", "src/ant_homing/artifacts.py"}


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_reviews(path: Path, dataset_hash: str, metadata_hash: str, project: Path) -> dict:
    """Fail closed if reviews are incomplete or reviewed source/data changed."""
    manifest = json.loads(path.read_text())
    if set(manifest.get("approvals", {})) != {
        "pixels_data",
        "infonce_training",
        "convap_evaluation",
    }:
        raise ValueError("Three independent pre-training approvals are required")
    if any(value != "PASS" for value in manifest["approvals"].values()):
        raise ValueError("Pre-training review has unresolved findings")
    if manifest.get("dataset_sha256") != dataset_hash:
        raise ValueError("Review manifest does not match the dataset")
    if manifest.get("metadata_sha256") != metadata_hash:
        raise ValueError("Review manifest does not match the dataset metadata")
    if not manifest.get("source_sha256") or not manifest.get("reviewed_at_utc"):
        raise ValueError("Review manifest must identify reviewed source and time")
    if not REVIEWED_SOURCE_PATHS.issubset(manifest["source_sha256"]):
        raise ValueError("Review manifest omits required source files")
    for name, expected in manifest["source_sha256"].items():
        if digest(project / name) != expected:
            raise ValueError(f"Reviewed source changed: {name}")
    return manifest


def place_batches(
    arrays: dict, epoch: int, seed: int, places_per_batch: int = 16
) -> list[np.ndarray]:
    """Same scene/place batches for both models, with an explicit spatial guard."""
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
    module_name: str,
    output: Path,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    review_hash: str,
) -> dict:
    """Called only by train_all after the complete source/data review gate."""
    module = importlib.import_module(f"ant_homing.level1.{module_name}")
    torch.manual_seed(seed)
    model = module.build_model()
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
            loss = module.training_loss(model, batch)
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
            lambda x: neural_features(module, model, x),
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
                    "module": module_name,
                    "seed": seed,
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "dataset_sha256": metadata["data_sha256"],
                    "review_manifest_sha256": review_hash,
                    "scene_seeds": metadata["scene_seeds"],
                    "image_shape": metadata["image_shape"],
                    "camera_convention": metadata["camera_convention"],
                },
                checkpoint,
            )
        print(
            json.dumps(
                {
                    "method": module_name,
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
        lambda x: neural_features(module, model, x),
        tolerance=metadata["retrieval_tolerance_m"],
    )
    report = {
        "method": module_name,
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
        "descriptor_dimension": module.DESCRIPTOR_DIM,
        "loss_configuration": getattr(
            module,
            "LOSS_CONFIG",
            {"temperature": module.TEMPERATURE} if hasattr(module, "TEMPERATURE") else {},
        ),
        "loss_reference": getattr(module, "LOSS_REFERENCE", "paired symmetric InfoNCE"),
        "visit_order": "per-place permutation seeded by training seed, epoch and batch index",
        "checkpoint_sha256": digest(checkpoint),
    }
    write_json(output / "training.json", report)
    return report


def train_all(
    data: Path, output: Path, reviews: Path, *, seeds=(0, 1, 2), epochs=20, learning_rate=0.001
) -> dict:
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
    project = Path(__file__).resolve().parents[3]
    metadata_hash = digest(data / "metadata.json")
    verify_reviews(reviews, metadata["data_sha256"], metadata_hash, project)
    target = new_output(output)
    (target / "review-manifest.json").write_bytes(reviews.read_bytes())
    review_hash = digest(reviews)
    started_at = datetime.now(UTC).isoformat()
    report = {
        "dataset_sha256": metadata["data_sha256"],
        "metadata_sha256": metadata_hash,
        "review_manifest_sha256": review_hash,
        "started_at_utc": started_at,
        "seeds": list(seeds),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "negative_guard_m": ACTUAL_NEGATIVE_GUARD_M,
        "canonical_guard_m": CANONICAL_NEGATIVE_GUARD_M,
        "provenance": provenance(),
        "runs": [],
        "pixels": {},
    }
    for variant in ("raw", "normalized"):
        validation = evaluate_split(
            arrays,
            "val",
            lambda x, v=variant: pixel_encode(x, v),
            tolerance=metadata["retrieval_tolerance_m"],
        )
        report["pixels"][variant] = {"threshold": validation_threshold(validation["rows"])}
    for method in METHODS:
        for seed in seeds:
            run = target / f"{method}-seed-{seed}"
            run.mkdir()
            report["runs"].append(
                train_one(
                    arrays,
                    metadata,
                    method,
                    run,
                    seed=seed,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    review_hash=review_hash,
                )
            )
    report["frozen_at_utc"] = datetime.now(UTC).isoformat()
    write_json(target / "trained.json", report)
    return report


def evaluate_all(data: Path, run_dir: Path, output: Path) -> dict:
    """Score test queries only after every run/checkpoint/threshold has been frozen."""
    torch.set_num_threads(1)
    arrays, metadata = load_dataset(data)
    frozen = json.loads((run_dir / "trained.json").read_text())
    if frozen["dataset_sha256"] != metadata["data_sha256"] or not frozen.get("frozen_at_utc"):
        raise ValueError("Dataset mismatch or unfinished training")
    project = Path(__file__).resolve().parents[3]
    manifest = run_dir / "review-manifest.json"
    verify_reviews(manifest, metadata["data_sha256"], digest(data / "metadata.json"), project)
    if frozen["metadata_sha256"] != digest(data / "metadata.json"):
        raise ValueError("Dataset metadata changed after training")
    if digest(manifest) != frozen["review_manifest_sha256"]:
        raise ValueError("Review manifest changed")
    target = new_output(output)
    results = []
    latency_images = arrays["gallery"][np.flatnonzero(arrays["gallery_splits"] == "train")[:32]]
    for variant in ("raw", "normalized"):
        result = evaluate_split(
            arrays,
            "test",
            lambda x, v=variant: pixel_encode(x, v),
            tolerance=metadata["retrieval_tolerance_m"],
            threshold=frozen["pixels"][variant]["threshold"],
        )
        result.update(
            method=variant,
            seed=None,
            parameter_count=0,
            model_state_bytes=0,
            descriptor_dimension=int(np.prod(metadata["image_shape"])),
            observation_latency=observation_latency(
                lambda x, v=variant: pixel_encode(x, v), latency_images
            ),
        )
        results.append(result)
    for run in frozen["runs"]:
        checkpoint = run_dir / f"{run['method']}-seed-{run['seed']}" / "model.pt"
        if digest(checkpoint) != run["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after freezing")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["dataset_sha256"] != metadata["data_sha256"]:
            raise ValueError("Checkpoint trained on a different dataset")
        if (
            saved["module"] != run["method"]
            or saved["seed"] != run["seed"]
            or saved["scene_seeds"] != metadata["scene_seeds"]
            or saved["image_shape"] != metadata["image_shape"]
            or saved["camera_convention"] != metadata["camera_convention"]
            or saved["review_manifest_sha256"] != frozen["review_manifest_sha256"]
        ):
            raise ValueError("Checkpoint provenance differs from the frozen run")
        module = importlib.import_module(f"ant_homing.level1.{run['method']}")
        model = module.build_model()
        model.load_state_dict(saved["state_dict"])
        # Warm-up uses training images, not validation/test labels or statistics.
        warmup = np.flatnonzero(arrays["gallery_splits"] == "train")[:8]
        neural_features(module, model, arrays["gallery"][warmup])
        result = evaluate_split(
            arrays,
            "test",
            partial(neural_features, module, model),
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
            partial(neural_features, module, model), latency_images
        )
        results.append(result)
    paired = {}
    normalized = next(row for row in results if row["method"] == "normalized")
    reference = {
        int(scene): values["recall_at_1"] for scene, values in normalized["scenes"].items()
    }
    for method in METHODS:
        runs = [row for row in results if row["method"] == method]
        mean_scenes = {
            scene: float(np.mean([row["scenes"][str(scene)]["recall_at_1"] for row in runs]))
            for scene in reference
        }
        paired[method] = paired_scene_interval(mean_scenes, reference)
    report = {
        "dataset_sha256": metadata["data_sha256"],
        "test_scene_ids": metadata["scene_seeds"]["test"],
        "frozen_at_utc": frozen["frozen_at_utc"],
        "results": results,
        "paired_recall_vs_normalized": paired,
        "interval_scope": "paired scene bootstrap conditional on these training seeds",
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
    train.add_argument("--reviews", type=Path, required=True)
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
            args.reviews,
            seeds=tuple(int(x) for x in args.seeds.split(",")),
            epochs=args.epochs,
        )
    else:
        report = evaluate_all(args.data, args.run_dir, args.output)
        print(json.dumps({"paired": report["paired_recall_vs_normalized"]}, indent=2))


if __name__ == "__main__":
    main()
