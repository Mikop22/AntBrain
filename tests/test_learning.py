"""Tests exercise leakage controls, image metrics and actual optimization."""

import hashlib
import json

import numpy as np
import pytest

from ant_homing.dataset import generate_dataset, load_dataset
from ant_homing.learning import (
    _pixel_features,
    evaluate_retrieval,
    load_image_scorer,
    train_model,
)
from ant_homing.simulation import CAMERA_CONVENTION


def make_dataset(path):
    """Create distinguishable synthetic views; no simulator is required here."""
    path.mkdir()
    rng = np.random.default_rng(9)
    images = rng.integers(20, 180, (12, 16, 64, 3), dtype=np.uint8)
    positions = np.tile(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.0]]), (3, 1))
    scenes = np.repeat([10, 20, 30], 4)
    np.savez_compressed(
        path / "images.npz",
        gallery=images,
        query=images.copy(),
        gallery_poses=positions,
        query_poses=positions.copy(),
        scene_ids=scenes,
        splits=np.repeat(["train", "val", "test"], 4),
        location_ids=np.tile(np.arange(4), 3),
        gains=np.ones((12, 2)),
    )
    metadata = {
        "format_version": 1,
        "seed": 0,
        "scene_seeds": {"train": [10], "val": [20], "test": [30]},
        "image_shape": [16, 64, 3],
        "camera_convention": CAMERA_CONVENTION,
        "retrieval_tolerance_m": 0.25,
        "data_sha256": hashlib.sha256((path / "images.npz").read_bytes()).hexdigest(),
    }
    (path / "metadata.json").write_text(json.dumps(metadata))
    return path


def test_retrieval_uses_correct_scene_and_measures_location_error(tmp_path):
    dataset = make_dataset(tmp_path / "data")
    report = evaluate_retrieval(dataset)
    assert report["split"] == "test"
    for metrics in report["metrics"].values():
        assert metrics["query_count"] == 4
        assert metrics["scene_count"] == 1
        assert metrics["top1_location_accuracy"] == 1.0
        assert metrics["median_location_error_m"] == 0


def test_normalized_baseline_removes_uniform_affine_brightness():
    rng = np.random.default_rng(5)
    images = rng.integers(0, 90, (3, 8, 32, 3), dtype=np.uint8)
    brighter = images * 2 + 10
    np.testing.assert_allclose(
        _pixel_features(images, True), _pixel_features(brighter, True), atol=2e-6
    )
    assert not np.allclose(_pixel_features(images, False), _pixel_features(brighter, False))


def test_hash_detects_modified_data(tmp_path):
    dataset = make_dataset(tmp_path / "data")
    with (dataset / "images.npz").open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_dataset(dataset)


def test_split_leakage_is_rejected(tmp_path):
    dataset = make_dataset(tmp_path / "data")
    with np.load(dataset / "images.npz") as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays["scene_ids"][4] = 10
    np.savez_compressed(dataset / "images.npz", **arrays)
    metadata = json.loads((dataset / "metadata.json").read_text())
    metadata["data_sha256"] = hashlib.sha256((dataset / "images.npz").read_bytes()).hexdigest()
    (dataset / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="more than one split"):
        load_dataset(dataset)


def test_actual_training_checkpoint_and_image_only_scorer(tmp_path):
    torch = pytest.importorskip("torch")
    from ant_homing.learning import _build_encoder

    dataset = make_dataset(tmp_path / "data")
    torch.manual_seed(0)
    initial = {k: v.clone() for k, v in _build_encoder().state_dict().items()}
    report = train_model(dataset, tmp_path / "model", epochs=2, batch_size=4)
    assert len(report["history"]) == 2
    assert all(np.isfinite(row["train_loss"]) for row in report["history"])
    checkpoint = tmp_path / "model" / "model.pt"
    fitted = torch.load(checkpoint, weights_only=True)["state_dict"]
    assert any(not torch.equal(initial[key], fitted[key]) for key in initial)
    scorer = load_image_scorer(checkpoint)
    arrays, _ = load_dataset(dataset)
    assert scorer(arrays["gallery"][0], arrays["gallery"][0]) == pytest.approx(0, abs=1e-10)
    assert scorer(arrays["gallery"][0], arrays["gallery"][1]) > 0
    evaluated = evaluate_retrieval(dataset, checkpoint, tmp_path / "evaluation.json")
    assert evaluated["metrics"]["cnn"]["top1_location_accuracy"] == 1
    with pytest.raises(ValueError, match="shape"):
        scorer(np.zeros((5, 5, 3), np.uint8), arrays["gallery"][0])


def test_metadata_cannot_misrepresent_training_scenes(tmp_path):
    dataset = make_dataset(tmp_path / "data")
    metadata = json.loads((dataset / "metadata.json").read_text())
    metadata["scene_seeds"]["train"] = [99]
    (dataset / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Scene membership"):
        load_dataset(dataset)


def test_stale_dataset_camera_convention_is_rejected(tmp_path):
    dataset = make_dataset(tmp_path / "data")
    metadata = json.loads((dataset / "metadata.json").read_text())
    metadata["camera_convention"] = "old-noncontiguous-strip"
    (dataset / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="camera convention"):
        load_dataset(dataset)


def test_stale_checkpoint_camera_convention_is_rejected(tmp_path):
    torch = pytest.importorskip("torch")
    dataset = make_dataset(tmp_path / "data")
    train_model(dataset, tmp_path / "model", epochs=1, batch_size=4)
    checkpoint = tmp_path / "model" / "model.pt"
    saved = torch.load(checkpoint, weights_only=True)
    saved["camera_convention"] = "old-noncontiguous-strip"
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="camera convention"):
        load_image_scorer(checkpoint)


def test_evaluation_rejects_checkpoint_training_scene_relabelled_as_test(tmp_path):
    pytest.importorskip("torch")
    dataset = make_dataset(tmp_path / "data")
    train_model(dataset, tmp_path / "model", epochs=1, batch_size=4)
    with np.load(dataset / "images.npz") as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays["scene_ids"][:4] = 11
    arrays["scene_ids"][8:] = 10
    np.savez_compressed(dataset / "images.npz", **arrays)
    metadata = json.loads((dataset / "metadata.json").read_text())
    metadata["scene_seeds"] = {"train": [11], "val": [20], "test": [10]}
    metadata["data_sha256"] = hashlib.sha256((dataset / "images.npz").read_bytes()).hexdigest()
    (dataset / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Test scenes overlap"):
        evaluate_retrieval(dataset, tmp_path / "model" / "model.pt")


def test_seeded_training_is_repeatable_and_uses_validation_only(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from ant_homing import learning

    dataset = make_dataset(tmp_path / "data")
    seen_splits = []
    original = learning._retrieval_metrics

    def record_split(arrays, split, feature_fn, tolerance):
        seen_splits.append(split)
        return original(arrays, split, feature_fn, tolerance)

    monkeypatch.setattr(learning, "_retrieval_metrics", record_split)
    first = train_model(dataset, tmp_path / "first", epochs=1, batch_size=4, seed=5)
    second = train_model(dataset, tmp_path / "second", epochs=1, batch_size=4, seed=5)
    assert first["history"] == second["history"]
    assert seen_splits == ["val", "val"]


def test_generator_rejects_seed_leakage_before_rendering(tmp_path):
    pytest.importorskip("pybullet")
    with pytest.raises(ValueError, match="split-disjoint"):
        generate_dataset(tmp_path / "data", scene_seeds={"train": [1], "val": [1], "test": [2]})


def test_collect_real_views_and_validate(tmp_path):
    pytest.importorskip("pybullet")
    generated = generate_dataset(
        tmp_path / "data",
        scene_seeds={"train": [1], "val": [2], "test": [3]},
        locations_per_scene=3,
        camera_size=16,
        seed=4,
    )
    arrays, loaded = load_dataset(tmp_path / "data")
    assert generated == loaded
    assert arrays["gallery"].shape == (9, 16, 64, 3)
    assert np.std(arrays["gallery"]) > 10
    assert not np.array_equal(arrays["gallery"], arrays["query"])
    assert (
        np.max(
            np.linalg.norm(arrays["query_poses"][:, :2] - arrays["gallery_poses"][:, :2], axis=1)
        )
        <= 0.06 * np.sqrt(2) + 1e-8
    )
