"""Pre-training checks: no optimizer updates occur in this test module."""

import json

import numpy as np
import pytest

from ant_homing.level1.evaluation import (
    evaluate_split,
    paired_scene_interval,
    summarize,
    validation_threshold,
)
from ant_homing.level1.experiment import (
    REVIEWED_SOURCE_PATHS,
    digest,
    place_batches,
    verify_reviews,
)


def test_review_gate_rejects_missing_approvals_changed_source_and_metadata(tmp_path):
    hashes = {}
    for name in REVIEWED_SOURCE_PATHS:
        source = tmp_path / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("reviewed code")
        hashes[name] = digest(source)
    record = {
        "approvals": {
            name: "PASS" for name in ("pixels_data", "infonce_training", "convap_evaluation")
        },
        "dataset_sha256": "data",
        "metadata_sha256": "metadata",
        "source_sha256": hashes,
        "reviewed_at_utc": "2026-11-10T00:00:00+00:00",
    }
    manifest = tmp_path / "reviews.json"
    manifest.write_text(json.dumps(record))
    assert verify_reviews(manifest, "data", "metadata", tmp_path) == record
    with pytest.raises(ValueError, match="metadata"):
        verify_reviews(manifest, "data", "changed", tmp_path)
    record["approvals"].pop("pixels_data")
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Three independent"):
        verify_reviews(manifest, "data", "metadata", tmp_path)
    record["approvals"]["pixels_data"] = "PASS"
    manifest.write_text(json.dumps(record))
    (tmp_path / next(iter(hashes))).write_text("unreviewed change")
    with pytest.raises(ValueError, match="source changed"):
        verify_reviews(manifest, "data", "metadata", tmp_path)


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
