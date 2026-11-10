"""Check actual captures, disjoint splits, geometry and frozen-data provenance."""

import json

import numpy as np
import pytest

from ant_homing.level1.data import generate_dataset, load_dataset, validate_dataset


@pytest.fixture(scope="module")
def collection(tmp_path_factory):
    path = tmp_path_factory.mktemp("level1-data")
    generate_dataset(
        path,
        seed=3,
        scene_seeds={"train": [100], "val": [200], "test": [300]},
        places_per_scene=4,
        unfamiliar_per_scene=4,
        camera_size=8,
    )
    return path


def test_captured_schema_counts_and_spatial_labels(collection):
    arrays, metadata = load_dataset(collection)
    assert arrays["gallery"].shape == (12, 8, 32, 3)
    assert arrays["observations"].shape == (12, 4, 8, 32, 3)
    assert arrays["query_images"].shape == (60, 8, 32, 3)
    assert arrays["query_familiar"].sum() == 48
    for split in ("train", "val", "test"):
        assert metadata["counts"][split] == {
            "scenes": 1,
            "gallery": 4,
            "queries": 20,
            "familiar_queries": 16,
            "unfamiliar_queries": 4,
        }
    assert np.isfinite(arrays["observation_positions"]).all()
    offsets = np.linalg.norm(
        arrays["observation_positions"][..., :2] - arrays["gallery_positions"][:, None, :2], axis=2
    )
    assert np.all((offsets >= 0.04 - 1e-8) & (offsets <= 0.16 + 1e-8))


def test_generation_replays_without_reusing_query_captures(collection, tmp_path):
    first, metadata = load_dataset(collection)
    generate_dataset(
        tmp_path,
        seed=3,
        scene_seeds=metadata["scene_seeds"],
        places_per_scene=4,
        unfamiliar_per_scene=4,
        camera_size=8,
    )
    second, _ = load_dataset(tmp_path)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    assert np.any(first["query_positions"][:4] != first["observation_positions"][0])
    with pytest.raises(FileExistsError):
        generate_dataset(collection)


def test_loader_rejects_changed_data_hash(collection, tmp_path):
    (tmp_path / "metadata.json").write_bytes((collection / "metadata.json").read_bytes())
    (tmp_path / "images.npz").write_bytes((collection / "images.npz").read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_dataset(tmp_path)


def test_split_labels_and_unknown_geometry_cannot_be_relabelled(collection):
    arrays, metadata = load_dataset(collection)
    bad_metadata = json.loads(json.dumps(metadata))
    bad_metadata["scene_seeds"]["test"] = bad_metadata["scene_seeds"]["train"]
    with pytest.raises(ValueError, match="overlap"):
        validate_dataset(arrays, bad_metadata)
    unknown = np.flatnonzero(~arrays["query_familiar"])[0]
    arrays["query_positions"][unknown] = arrays["gallery_positions"][0]
    with pytest.raises(ValueError, match="Unfamiliar"):
        validate_dataset(arrays, metadata)


def test_familiar_labels_must_belong_to_same_scene(collection):
    arrays, metadata = load_dataset(collection)
    arrays["query_place_ids"][0] = arrays["place_ids"][-1]
    with pytest.raises(ValueError, match="another scene"):
        validate_dataset(arrays, metadata)


@pytest.mark.parametrize("corruption", ["shape", "float_ids", "threshold", "counts", "config"])
def test_validator_rejects_misleading_schema_and_provenance(collection, corruption):
    arrays, metadata = load_dataset(collection)
    if corruption == "shape":
        arrays["gallery"] = arrays["gallery"][:, :, :-1]
        arrays["observations"] = arrays["observations"][:, :, :, :-1]
        arrays["query_images"] = arrays["query_images"][:, :, :-1]
        metadata["image_shape"] = [8, 31, 3]
    elif corruption == "float_ids":
        arrays["place_ids"] = arrays["place_ids"].astype(float) + 0.25
    elif corruption == "threshold":
        metadata["retrieval_tolerance_m"] = float("nan")
    elif corruption == "counts":
        metadata["counts"]["val"]["gallery"] += 1
    else:
        metadata["scene_configs"][0]["camera_size"] = 9
    with pytest.raises(ValueError):
        validate_dataset(arrays, metadata)
