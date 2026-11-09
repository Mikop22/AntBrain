"""Reproducible visual-memory data with scene-disjoint evaluation splits.

True poses are labels for collection and evaluation only. The inference API in
``learning`` consumes RGB images, never these labels.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

DEFAULT_SCENES = {"train": [10, 11, 12, 13], "val": [20, 21], "test": [30, 31]}
FORMAT_VERSION = 1


def _sensor_session(image: np.ndarray, gain: float, rng: np.random.Generator) -> np.ndarray:
    """Apply a documented exposure/noise condition, not physical relighting."""
    noisy = image.astype(np.float32) * gain + rng.normal(0, 1.5, image.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def generate_dataset(
    output: str | Path,
    *,
    scene_seeds: Mapping[str, Sequence[int]] | None = None,
    locations_per_scene: int = 24,
    seed: int = 0,
    camera_size: int = 32,
    query_jitter: float = 0.06,
) -> dict:
    """Render gallery/query sessions and persist their exact provenance.

    Each scene contributes matched locations and within-scene distractors. Scenes
    are disjoint between train, validation and test. Queries are captured close
    to each gallery location, with independent brightness/noise perturbations.
    This tests repeat-visit matching; it is not yet a navigation experiment.
    """
    from .simulation import CAMERA_CONVENTION, Arena, Pose, SimConfig

    if locations_per_scene < 2:
        raise ValueError("At least two locations per scene are required for negatives.")
    if camera_size < 8 or not np.isfinite(query_jitter) or query_jitter < 0:
        raise ValueError("camera_size must be at least 8 and query_jitter nonnegative.")
    specification = DEFAULT_SCENES if scene_seeds is None else scene_seeds
    scenes = {k: [int(s) for s in v] for k, v in specification.items()}
    if set(scenes) != {"train", "val", "test"} or any(not v for v in scenes.values()):
        raise ValueError("scene_seeds must contain nonempty train, val and test lists.")
    all_seeds = [s for group in scenes.values() for s in group]
    if min(all_seeds) < 0 or seed < 0 or len(set(all_seeds)) != len(all_seeds):
        raise ValueError("Scene seeds must be nonnegative, unique and split-disjoint.")
    target = Path(output)
    if (target / "images.npz").exists() or (target / "metadata.json").exists():
        raise FileExistsError(f"Dataset already exists in {target}; choose a new directory.")

    gallery, queries, gallery_poses, query_poses = [], [], [], []
    scene_ids, split_ids, location_ids, gains, configs = [], [], [], [], []
    for split in ("train", "val", "test"):
        for scene_seed in scenes[split]:
            rng = np.random.default_rng(np.random.SeedSequence([seed, scene_seed]))
            config = SimConfig(seed=scene_seed, camera_size=camera_size)
            configs.append(asdict(config))
            with Arena(config) as arena:
                poses = []
                for location in range(locations_per_scene):
                    for _ in range(2000):
                        pose = arena.sample_free_pose(rng)
                        if all(np.hypot(pose.x - p.x, pose.y - p.y) >= 0.25 for p in poses):
                            break
                    else:
                        raise ValueError("Too many locations for the available free arena area.")
                    poses.append(pose)
                    arena.reset(pose)
                    gallery_image = arena.observe().rgb
                    query_pose = pose
                    for _ in range(20):
                        delta = rng.uniform(-query_jitter, query_jitter, 2)
                        candidate = Pose(
                            pose.x + float(delta[0]), pose.y + float(delta[1]), pose.yaw
                        )
                        try:
                            arena.reset(candidate)
                        except ValueError:
                            continue
                        query_pose = candidate
                        break
                    arena.reset(query_pose)
                    query_image = arena.observe().rgb
                    gallery_gain = float(rng.uniform(0.9, 1.1))
                    query_gain = float(rng.uniform(0.65, 1.35))
                    gallery.append(_sensor_session(gallery_image, gallery_gain, rng))
                    queries.append(_sensor_session(query_image, query_gain, rng))
                    gallery_poses.append([pose.x, pose.y, pose.yaw])
                    query_poses.append([query_pose.x, query_pose.y, query_pose.yaw])
                    scene_ids.append(scene_seed)
                    split_ids.append(split)
                    location_ids.append(location)
                    gains.append([gallery_gain, query_gain])

    target.mkdir(parents=True, exist_ok=True)
    arrays = {
        "gallery": np.stack(gallery),
        "query": np.stack(queries),
        "gallery_poses": np.asarray(gallery_poses, dtype=np.float64),
        "query_poses": np.asarray(query_poses, dtype=np.float64),
        "scene_ids": np.asarray(scene_ids, dtype=np.int64),
        "splits": np.asarray(split_ids, dtype="U5"),
        "location_ids": np.asarray(location_ids, dtype=np.int64),
        "gains": np.asarray(gains, dtype=np.float32),
    }
    np.savez_compressed(target / "images.npz", **arrays)
    metadata = {
        "format_version": FORMAT_VERSION,
        "seed": seed,
        "scene_seeds": scenes,
        "locations_per_scene": locations_per_scene,
        "image_shape": list(arrays["gallery"].shape[1:]),
        "query_jitter_per_axis_m": query_jitter,
        "session_model": "independent exposure gain plus Gaussian RGB sensor noise (std=1.5/255)",
        "gallery_gain_range": [0.9, 1.1],
        "query_gain_range": [0.65, 1.35],
        "heading_assumption": "compass-aligned panorama",
        "camera_convention": CAMERA_CONVENTION,
        "retrieval_tolerance_m": 0.25,
        "scene_configs": configs,
        "pair_count": len(gallery),
        "data_sha256": hashlib.sha256((target / "images.npz").read_bytes()).hexdigest(),
        "scope": "within-scene repeat-visit retrieval, evaluated on scenes unseen during training",
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_dataset(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load and validate arrays, hashes and split membership without pickle."""
    from .simulation import CAMERA_CONVENTION

    directory = Path(path)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported dataset format.")
    if metadata.get("camera_convention") != CAMERA_CONVENTION:
        raise ValueError("Dataset camera convention is incompatible; regenerate the collection.")
    digest = hashlib.sha256((directory / "images.npz").read_bytes()).hexdigest()
    if digest != metadata.get("data_sha256"):
        raise ValueError("Dataset hash mismatch: images.npz changed after collection.")
    with np.load(directory / "images.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    required = {
        "gallery",
        "query",
        "gallery_poses",
        "query_poses",
        "scene_ids",
        "splits",
        "location_ids",
        "gains",
    }
    if not required.issubset(arrays):
        raise ValueError("Dataset is missing required arrays.")
    count = len(arrays["gallery"])
    if not count or any(len(value) != count for value in arrays.values()):
        raise ValueError("Dataset arrays have inconsistent sample counts.")
    if arrays["gallery"].shape != arrays["query"].shape or arrays["gallery"].ndim != 4:
        raise ValueError("Gallery and query must be matching NHWC arrays.")
    if arrays["gallery"].shape[-1] != 3 or any(
        arrays[k].dtype != np.uint8 for k in ("gallery", "query")
    ):
        raise ValueError("Images must contain uint8 RGB values.")
    if list(arrays["gallery"].shape[1:]) != metadata.get("image_shape"):
        raise ValueError("Image shape differs from dataset metadata.")
    for name in ("gallery_poses", "query_poses"):
        if arrays[name].shape != (count, 3) or not np.isfinite(arrays[name]).all():
            raise ValueError("Pose labels must be finite arrays with shape (N, 3).")
    if arrays["gains"].shape != (count, 2) or not np.isfinite(arrays["gains"]).all():
        raise ValueError("Session gains must be finite arrays with shape (N, 2).")
    if any(arrays[name].shape != (count,) for name in ("scene_ids", "splits", "location_ids")):
        raise ValueError("Scene, split and location labels must be one-dimensional.")
    tolerance = metadata.get("retrieval_tolerance_m")
    if not isinstance(tolerance, (int, float)) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Retrieval tolerance must be finite and positive.")
    if set(arrays["splits"].tolist()) != {"train", "val", "test"}:
        raise ValueError("Dataset must contain all three evaluation splits.")
    for scene in np.unique(arrays["scene_ids"]):
        rows = arrays["scene_ids"] == scene
        if len(set(arrays["splits"][rows].tolist())) != 1:
            raise ValueError("A scene occurs in more than one split.")
        if rows.sum() < 2:
            raise ValueError("Each scene requires at least two locations.")
        if len(np.unique(arrays["location_ids"][rows])) != int(rows.sum()):
            raise ValueError("Location IDs must be unique within a scene.")
    declared_scenes = metadata.get("scene_seeds", {})
    for split in ("train", "val", "test"):
        actual = set(arrays["scene_ids"][arrays["splits"] == split].tolist())
        if actual != set(declared_scenes.get(split, [])):
            raise ValueError("Scene membership differs from dataset metadata.")
    return arrays, metadata
