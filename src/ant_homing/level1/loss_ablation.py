"""Review-gated, matched-encoder loss experiment; original pilot stays frozen."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch

from ant_homing.artifacts import new_output, provenance, write_json

from .data import load_dataset
from .evaluation import evaluate_split, paired_scene_interval
from .experiment import REVIEWED_SOURCE_PATHS, digest, neural_features, train_one

METHODS = ("infonce", "infonce_ms")
SEEDS = (0, 1, 2)
EXTRA_SOURCES = {
    "src/ant_homing/level1/infonce_ms.py",
    "src/ant_homing/level1/loss_ablation.py",
    "docs/level1/loss-ablation/PROTOCOL.md",
    "tests/test_loss_ablation.py",
}


def verify_gate(reviews: Path, data: Path, metadata: dict) -> dict:
    """Bind three independent approvals, source, protocol, data and metadata."""
    root = Path(__file__).resolve().parents[3]
    record = json.loads(reviews.read_text())
    if record.get("approvals") != dict.fromkeys(("loss", "training", "evaluation"), "PASS"):
        raise ValueError("Three passing independent reviews are required")
    if record.get("dataset_sha256") != metadata["data_sha256"] or record.get(
        "metadata_sha256"
    ) != digest(data / "metadata.json"):
        raise ValueError("Reviewed data or metadata changed")
    hashes = record.get("source_sha256", {})
    if not (REVIEWED_SOURCE_PATHS | EXTRA_SOURCES).issubset(hashes):
        raise ValueError("Required reviewed source omitted")
    for name, expected in hashes.items():
        if digest(root / name) != expected:
            raise ValueError(f"Reviewed source changed: {name}")
    if set(record.get("reports", {})) != set(record["approvals"]):
        raise ValueError("Review reports missing")
    for report in record["reports"].values():
        if digest(root / report["path"]) != report["sha256"]:
            raise ValueError("Review report changed")
    if record.get("budget") != {"seeds": list(SEEDS), "epochs": 20, "learning_rate": 0.001}:
        raise ValueError("Reviewed budget differs")
    if metadata["scene_seeds"]["test"] != list(range(400, 412)):
        raise ValueError("Fresh test scenes must be 400 through 411")
    if metadata["image_shape"] != [32, 128, 3] or metadata["observations_per_place"] != 4:
        raise ValueError("Native camera resolution and four visits required")
    return record


def initial_state_hash(module, seed: int) -> str:
    """Fingerprint identical initialization without taking an optimizer step."""
    torch.manual_seed(seed)
    model = module.build_model()
    state = hashlib.sha256()
    for name, value in model.state_dict().items():
        state.update(name.encode())
        state.update(value.numpy().tobytes())
    return state.hexdigest()


def train(data: Path, output: Path, reviews: Path) -> dict:
    torch.set_num_threads(1)
    arrays, metadata = load_dataset(data)
    verify_gate(reviews, data, metadata)
    target = new_output(output)
    (target / "review-manifest.json").write_bytes(reviews.read_bytes())
    review_hash = digest(reviews)
    report = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "dataset_sha256": metadata["data_sha256"],
        "metadata_sha256": digest(data / "metadata.json"),
        "review_manifest_sha256": review_hash,
        "seeds": list(SEEDS),
        "epochs": 20,
        "learning_rate": 0.001,
        "provenance": provenance(),
        "runs": [],
    }
    for seed in SEEDS:
        paired = []
        for method in METHODS:
            module = importlib.import_module(f"ant_homing.level1.{method}")
            initial = initial_state_hash(module, seed)
            directory = target / f"{method}-seed-{seed}"
            directory.mkdir()
            run = train_one(
                arrays,
                metadata,
                method,
                directory,
                seed=seed,
                epochs=20,
                learning_rate=0.001,
                review_hash=review_hash,
            )
            run["initial_state_sha256"] = initial
            if run["parameter_count"] != 42728 or run["descriptor_dimension"] != 64:
                raise ValueError("Encoder architecture differs")
            paired.append(run)
            report["runs"].append(run)
        if paired[0]["initial_state_sha256"] != paired[1]["initial_state_sha256"]:
            raise ValueError("Paired initial weights differ")
        budgets = [[(e["updates"], e["image_exposures"]) for e in r["history"]] for r in paired]
        if budgets[0] != budgets[1]:
            raise ValueError("Paired training budgets differ")
    verify_gate(reviews, data, metadata)
    report["frozen_at_utc"] = datetime.now(UTC).isoformat()
    write_json(target / "trained.json", report)
    (target / "trained.sha256").write_text(digest(target / "trained.json") + "\n")
    return report


def evaluate(data: Path, run_dir: Path, output: Path) -> dict:
    torch.set_num_threads(1)
    arrays, metadata = load_dataset(data)
    marker = run_dir / "trained.sha256"
    if not marker.is_file() or marker.read_text().strip() != digest(run_dir / "trained.json"):
        raise ValueError("Incomplete or changed frozen experiment")
    frozen = json.loads((run_dir / "trained.json").read_text())
    manifest = run_dir / "review-manifest.json"
    verify_gate(manifest, data, metadata)
    if (
        not frozen.get("frozen_at_utc")
        or frozen["review_manifest_sha256"] != digest(manifest)
        or frozen["dataset_sha256"] != metadata["data_sha256"]
        or frozen["metadata_sha256"] != digest(data / "metadata.json")
        or {(r["method"], r["seed"]) for r in frozen["runs"]}
        != {(m, s) for m in METHODS for s in SEEDS}
        or len(frozen["runs"]) != 6
    ):
        raise ValueError("Incomplete or changed frozen experiment")
    # Validate every checkpoint before exposing the test split to any model.
    for run in frozen["runs"]:
        path = run_dir / f"{run['method']}-seed-{run['seed']}" / "model.pt"
        if digest(path) != run["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        expected = {
            "module": run["method"],
            "seed": run["seed"],
            "epoch": run["selected_epoch"],
            "dataset_sha256": metadata["data_sha256"],
            "review_manifest_sha256": frozen["review_manifest_sha256"],
            "scene_seeds": metadata["scene_seeds"],
            "image_shape": metadata["image_shape"],
            "camera_convention": metadata["camera_convention"],
        }
        if any(saved[k] != value for k, value in expected.items()):
            raise ValueError("Checkpoint provenance differs")
    target = new_output(output)
    results = []
    for run in frozen["runs"]:
        module = importlib.import_module(f"ant_homing.level1.{run['method']}")
        model = module.build_model()
        saved = torch.load(
            run_dir / f"{run['method']}-seed-{run['seed']}" / "model.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(saved["state_dict"])
        result = evaluate_split(
            arrays,
            "test",
            partial(neural_features, module, model),
            tolerance=metadata["retrieval_tolerance_m"],
            threshold=run["threshold"],
        )
        result.update(method=run["method"], seed=run["seed"])
        results.append(result)
    paired = {}
    for metric in (
        "recall_at_1",
        "unfamiliar_false_acceptance",
        "familiar_acceptance",
        "familiarity_auroc",
    ):
        scenes = {}
        for method in METHODS:
            runs = [r for r in results if r["method"] == method]
            scenes[method] = {
                int(s): float(np.mean([r["scenes"][s][metric] for r in runs]))
                for s in runs[0]["scenes"]
            }
        paired[metric] = paired_scene_interval(scenes["infonce_ms"], scenes["infonce"])
    report = {
        "dataset_sha256": metadata["data_sha256"],
        "review_manifest_sha256": frozen["review_manifest_sha256"],
        "frozen_at_utc": frozen["frozen_at_utc"],
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "test_scene_ids": metadata["scene_seeds"]["test"],
        "results": results,
        "paired_ms_minus_infonce": paired,
        "interval_scope": "paired scene means conditional on three training seeds; AUROC interval is for mean scene AUROC, not pooled AUROC",
    }
    write_json(target / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--reviews" if name == "train" else "--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.output, args.reviews)
    else:
        evaluate(args.data, args.run_dir, args.output)


if __name__ == "__main__":
    main()
