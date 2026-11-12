"""Batching and evaluation checks for Level 1 visual-memory experiments."""

import numpy as np
import pytest

from ant_homing.level1.evaluation import (
    evaluate_split,
    paired_scene_interval,
    summarize,
    validation_threshold,
)
from ant_homing.level1.experiment import place_batches


def test_shared_batches_cover_training_places_and_respect_actual_negative_guard():
    positions = np.array([[x * 0.6, y * 0.6, 0] for x in range(6) for y in range(6)])
    arrays = {
        "gallery_positions": positions,
        "observation_positions": np.repeat(positions[:, None], 4, axis=1),
        "gallery_scene_ids": np.zeros(36, dtype=int),
        "gallery_splits": np.array(["train"] * 36),
    }
    batches = place_batches(arrays, 0, 3)
    assert set(np.concatenate(batches)) == set(range(36))
    for batch, repeat in zip(batches, place_batches(arrays, 0, 3), strict=True):
        assert np.array_equal(batch, repeat)
        distances = np.linalg.norm(positions[batch, None, :2] - positions[None, batch, :2], axis=-1)
        np.fill_diagonal(distances, np.inf)
        assert distances.min() >= 0.9


def test_retrieval_uses_within_scene_gallery_and_rgb_only():
    gallery = np.zeros((3, 1, 1, 3), dtype=np.uint8)
    gallery[1] = 100
    gallery[2] = 240
    queries = gallery[[0, 2]]
    arrays = {
        "gallery": gallery,
        "gallery_scene_ids": np.array([30, 30, 99]),
        "gallery_splits": np.array(["test", "test", "train"]),
        "gallery_positions": np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
        "place_ids": np.array([10, 11, 12]),
        "query_images": queries,
        "query_positions": np.array([[0.1, 0, 0], [2, 0, 0]]),
        "query_scene_ids": np.array([30, 30]),
        "query_splits": np.array(["test", "test"]),
        "query_conditions": np.array(["clean", "clean"]),
        "query_familiar": np.array([True, False]),
        "query_place_ids": np.array([10, -1]),
    }

    def features(images):
        assert images.dtype == np.uint8 and images.ndim == 4
        return images.reshape(len(images), -1).astype(np.float32)

    report = evaluate_split(arrays, "test", features)
    assert report["metrics"]["recall_at_1"] == 1
    assert report["rows"][1]["predicted_place"] == 11  # Train-scene place 12 is not eligible.
    assert report["metrics"]["familiarity_auroc"] == 1


def test_rejection_auroc_ties_and_threshold_do_not_use_novel_queries():
    common = {
        "within_tolerance": True,
        "within_tolerance_at_5": True,
        "correct_place": True,
        "error_m": 0.1,
    }
    rows = [
        {**common, "familiar": True, "distance": 1},
        {**common, "familiar": False, "distance": 1},
    ]
    assert summarize(rows)["familiarity_auroc"] == 0.5
    assert validation_threshold(rows) == 1
    rows[1]["distance"] = 10000
    assert validation_threshold(rows) == 1
    assert summarize(rows, 1)["unfamiliar_false_acceptance"] == 0


def test_paired_scene_interval_preserves_constant_difference():
    interval = paired_scene_interval({1: 0.7, 2: 0.9}, {1: 0.5, 2: 0.7})
    assert interval["mean_difference"] == pytest.approx(0.2)
    assert interval["ci95"] == pytest.approx([0.2, 0.2])
