"""Shared image-only retrieval and familiarity evaluation for all approaches."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import numpy as np


def squared_distances(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.square(query).sum(1)[:, None]
        + np.square(gallery).sum(1)[None, :]
        - 2 * query @ gallery.T,
        0,
    )


def summarize(rows: list[dict], threshold: float | None = None) -> dict:
    known = [row for row in rows if row["familiar"]]
    novel = [row for row in rows if not row["familiar"]]
    distances_known = np.array([row["distance"] for row in known])
    distances_novel = np.sort([row["distance"] for row in novel])
    auroc = None
    if len(known) and len(novel):
        left = np.searchsorted(distances_novel, distances_known, side="left")
        right = np.searchsorted(distances_novel, distances_known, side="right")
        auroc = float(np.mean((len(novel) - right + 0.5 * (right - left)) / len(novel)))
    result = {
        "queries": len(rows),
        "familiar_queries": len(known),
        "unfamiliar_queries": len(novel),
        "recall_at_1": float(np.mean([r["within_tolerance"] for r in known])) if known else None,
        "recall_at_5": float(np.mean([r["within_tolerance_at_5"] for r in known]))
        if known
        else None,
        "place_identity_accuracy": float(np.mean([r["correct_place"] for r in known]))
        if known
        else None,
        "median_error_m": float(np.median([r["error_m"] for r in known])) if known else None,
        "p90_error_m": float(np.quantile([r["error_m"] for r in known], 0.9)) if known else None,
        "familiarity_auroc": auroc,
    }
    if threshold is not None:
        result.update(
            threshold=float(threshold),
            familiar_acceptance=float(np.mean(distances_known <= threshold)) if known else None,
            unfamiliar_false_acceptance=float(np.mean(distances_novel <= threshold))
            if novel
            else None,
            correct_and_accepted=float(
                np.mean([r["within_tolerance"] and r["distance"] <= threshold for r in known])
            )
            if known
            else None,
        )
    return result


def validation_threshold(rows: list[dict], acceptance: float = 0.95) -> float:
    """Calibrate using validation familiar examples, never test observations."""
    values = [row["distance"] for row in rows if row["familiar"]]
    if not values or not 0 < acceptance <= 1:
        raise ValueError(
            "Threshold calibration needs familiar validation queries and valid acceptance"
        )
    return float(np.quantile(values, acceptance, method="higher"))


def observation_latency(feature_fn: Callable, training_images: np.ndarray) -> dict:
    """Time sequential observations after warm-up on identical training images."""
    if not len(training_images):
        raise ValueError("Latency measurement requires training images")
    feature_fn(training_images[:8])
    times = []
    for image in training_images[:32]:
        start = perf_counter()
        feature_fn(image[None])
        times.append(1000 * (perf_counter() - start))
    return {
        "observations": len(times),
        "median_ms": float(np.median(times)),
        "p90_ms": float(np.quantile(times, 0.9)),
    }


def evaluate_split(
    arrays: dict,
    split: str,
    feature_fn: Callable[[np.ndarray], np.ndarray],
    *,
    tolerance: float = 0.25,
    threshold: float | None = None,
) -> dict:
    """Query within each scene's gallery; expose only RGB images to the encoder."""
    if split not in {"val", "test"} or tolerance <= 0:
        raise ValueError("Evaluation requires val/test and a positive spatial tolerance")
    rows, elapsed, images_count = [], 0.0, 0
    scenes = np.unique(arrays["gallery_scene_ids"][arrays["gallery_splits"] == split])
    if not len(scenes):
        raise ValueError(f"No gallery scenes in split {split}")
    for scene in scenes:
        gallery_indices = np.flatnonzero(
            (arrays["gallery_scene_ids"] == scene) & (arrays["gallery_splits"] == split)
        )
        query_indices = np.flatnonzero(
            (arrays["query_scene_ids"] == scene) & (arrays["query_splits"] == split)
        )
        before = perf_counter()
        gallery_features = feature_fn(arrays["gallery"][gallery_indices])
        query_features = feature_fn(arrays["query_images"][query_indices])
        elapsed += perf_counter() - before
        images_count += len(gallery_indices) + len(query_indices)
        if not np.isfinite(gallery_features).all() or not np.isfinite(query_features).all():
            raise ValueError("Nonfinite model descriptors")
        distances = squared_distances(query_features, gallery_features)
        ranked = np.argsort(distances, axis=1, kind="stable")
        for local, index in enumerate(query_indices):
            selected = gallery_indices[ranked[local, :5]]
            pose_errors = np.linalg.norm(
                arrays["gallery_positions"][selected, :2] - arrays["query_positions"][index, :2],
                axis=1,
            )
            familiar = bool(arrays["query_familiar"][index])
            rows.append(
                {
                    "query_index": int(index),
                    "scene": int(scene),
                    "condition": str(arrays["query_conditions"][index]),
                    "familiar": familiar,
                    "distance": float(distances[local, ranked[local, 0]]),
                    "predicted_place": int(arrays["place_ids"][selected[0]]),
                    "target_place": int(arrays["query_place_ids"][index]),
                    "correct_place": familiar
                    and bool(arrays["place_ids"][selected[0]] == arrays["query_place_ids"][index]),
                    "error_m": float(pose_errors[0]),
                    "within_tolerance": familiar and bool(pose_errors[0] <= tolerance),
                    "within_tolerance_at_5": familiar and bool(np.any(pose_errors <= tolerance)),
                }
            )
    return {
        "split": split,
        "tolerance_m": tolerance,
        "metrics": summarize(rows, threshold),
        "conditions": {
            condition: summarize([r for r in rows if r["condition"] == condition], threshold)
            for condition in sorted({r["condition"] for r in rows})
        },
        "scenes": {
            str(scene): summarize([r for r in rows if r["scene"] == scene], threshold)
            for scene in scenes
        },
        "batched_encoding_ms_per_image": 1000 * elapsed / images_count,
        "rows": rows,
    }


def paired_scene_interval(left: dict[int, float], right: dict[int, float], seed: int = 0) -> dict:
    """Bootstrap paired scene differences, not individual correlated frames."""
    scenes = sorted(set(left) & set(right))
    if not scenes:
        raise ValueError("No paired scenes")
    difference = np.array([left[s] - right[s] for s in scenes])
    rng = np.random.default_rng(seed)
    replicates = difference[rng.integers(0, len(scenes), size=(5000, len(scenes)))].mean(1)
    return {
        "scenes": len(scenes),
        "mean_difference": float(difference.mean()),
        "ci95": np.quantile(replicates, [0.025, 0.975]).tolist(),
    }
