"""Shared, scene-disjoint data for the Level 1 visual-memory comparison.

This format keeps images separate from evaluator-only spatial labels. Exposure
and occlusion conditions are synthetic sensor perturbations, not relighting or
new scene geometry. No model fitting occurs during collection or validation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..simulation import CAMERA_CONVENTION, Arena, Pose, SimConfig

FORMAT_VERSION = 1
CONDITIONS = ("clean", "exposure", "occlusion", "combined")
DEFAULT_SCENES = {
    "train": list(range(100, 124)),
    "val": list(range(200, 208)),
    "test": list(range(300, 312)),
}


def _perturb(image: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    """Apply a declared condition independently of location and split."""
    pixels = image.astype(np.float32)
    if condition in {"exposure", "combined"}:
        gain = rng.uniform(0.55, 1.45)
        contrast = rng.uniform(0.75, 1.25)
        channel_gain = rng.uniform(0.80, 1.20, (1, 1, 3))
        pixels = ((pixels - 127.5) * contrast + 127.5) * gain * channel_gain
        pixels += rng.normal(0, 2.5, image.shape)
    if condition in {"occlusion", "combined"}:
        # One or two partial perspective views are obstructed. The other views
        # remain available; rectangles never stitch across a camera boundary.
        height, width = image.shape[:2]
        face_width = width // 4
        for face in rng.choice(4, size=int(rng.integers(1, 3)), replace=False):
            patch_height = int(rng.integers(max(2, height // 3), max(3, height * 3 // 4)))
            patch_width = int(rng.integers(max(2, face_width // 3), max(3, face_width * 3 // 4)))
            y = int(rng.integers(0, height - patch_height + 1))
            x = int(face) * face_width + int(rng.integers(0, face_width - patch_width + 1))
            color = rng.uniform(20, 210, 3)
            pixels[y : y + patch_height, x : x + patch_width] = color
    return np.clip(np.rint(pixels), 0, 255).astype(np.uint8)


def _locations(arena: Arena, count: int, rng: np.random.Generator, spacing: float) -> list[Pose]:
    """Sample nearby pairs as informative spatial distractors, without overlap."""
    poses: list[Pose] = []
    for index in range(count):
        for _ in range(10000):
            if index % 2 and poses:
                angle = rng.uniform(-np.pi, np.pi)
                distance = rng.uniform(spacing + 0.02, spacing + 0.18)
                anchor = poses[-1]
                candidate = Pose(
                    anchor.x + distance * np.cos(angle), anchor.y + distance * np.sin(angle), 0.0
                )
                if not arena._is_free(candidate.x, candidate.y):
                    continue
            else:
                sampled = arena.sample_free_pose(rng)
                candidate = Pose(sampled.x, sampled.y, 0.0)
            if all(np.hypot(candidate.x - p.x, candidate.y - p.y) >= spacing for p in poses):
                poses.append(candidate)
                break
        else:
            raise ValueError("Could not place the requested gallery with minimum spacing")
    return poses


def _jittered_capture(arena: Arena, pose: Pose, radius: float, rng: np.random.Generator):
    for _ in range(1000):
        angle = rng.uniform(-np.pi, np.pi)
        distance = rng.uniform(0.04, radius)
        candidate = Pose(pose.x + distance * np.cos(angle), pose.y + distance * np.sin(angle), 0.0)
        if arena._is_free(candidate.x, candidate.y):
            return candidate, arena.reset(candidate).rgb
    raise ValueError("Could not capture a collision-free repeat view")


def generate_dataset(
    output: str | Path,
    *,
    seed: int = 2026,
    scene_seeds: Mapping[str, Sequence[int]] | None = None,
    places_per_scene: int = 48,
    observations_per_place: int = 4,
    unfamiliar_per_scene: int = 24,
    camera_size: int = 32,
) -> dict:
    """Collect one fixed comparison dataset without fitting or scoring models."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if places_per_scene < 2 or observations_per_place < 2 or unfamiliar_per_scene < 1:
        raise ValueError("Require at least two places/views and one unfamiliar query per scene")
    if not isinstance(camera_size, int) or camera_size < 8:
        raise ValueError("camera_size must be an integer of at least eight")
    scenes = {
        k: [int(v) for v in values]
        for k, values in (DEFAULT_SCENES if scene_seeds is None else scene_seeds).items()
    }
    if set(scenes) != {"train", "val", "test"} or any(not v for v in scenes.values()):
        raise ValueError("Require nonempty train, val and test scene lists")
    flattened = [v for values in scenes.values() for v in values]
    if any(v < 0 for v in flattened) or len(set(flattened)) != len(flattened):
        raise ValueError("Scene seeds must be unique, nonnegative, and split-disjoint")
    destination = Path(output)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    values: dict[str, list] = {
        k: []
        for k in (
            "gallery",
            "observations",
            "gallery_positions",
            "observation_positions",
            "gallery_scene_ids",
            "gallery_splits",
            "place_ids",
            "query_images",
            "query_positions",
            "query_scene_ids",
            "query_splits",
            "query_conditions",
            "query_familiar",
            "query_place_ids",
        )
    }
    configs = []
    for split in ("train", "val", "test"):
        for scene_seed in scenes[split]:
            config = SimConfig(
                seed=scene_seed, arena_size=8.0, camera_size=camera_size, landmark_count=24
            )
            configs.append(asdict(config))
            geometry_rng = np.random.default_rng(np.random.SeedSequence([seed, scene_seed, 0]))
            observation_rng = np.random.default_rng(np.random.SeedSequence([seed, scene_seed, 1]))
            query_rng = np.random.default_rng(np.random.SeedSequence([seed, scene_seed, 2]))
            with Arena(config) as arena:
                poses = _locations(arena, places_per_scene, geometry_rng, 0.50)
                for pose in poses:
                    place_id = len(values["gallery"])
                    values["gallery"].append(arena.reset(pose).rgb)
                    observations, observed_poses = [], []
                    for visit in range(observations_per_place):
                        p, image = _jittered_capture(arena, pose, 0.16, observation_rng)
                        observations.append(_perturb(image, CONDITIONS[visit % 4], observation_rng))
                        observed_poses.append([p.x, p.y, p.yaw])
                    values["observations"].append(np.stack(observations))
                    values["observation_positions"].append(observed_poses)
                    values["gallery_positions"].append([pose.x, pose.y, pose.yaw])
                    values["gallery_scene_ids"].append(scene_seed)
                    values["gallery_splits"].append(split)
                    values["place_ids"].append(place_id)
                    for condition in CONDITIONS:
                        p, image = _jittered_capture(arena, pose, 0.20, query_rng)
                        _append_query(
                            values,
                            _perturb(image, condition, query_rng),
                            p,
                            scene_seed,
                            split,
                            condition,
                            True,
                            place_id,
                        )
                reference_positions = np.asarray([[p.x, p.y] for p in poses])
                for index in range(unfamiliar_per_scene):
                    for _ in range(10000):
                        sampled = arena.sample_free_pose(query_rng)
                        p = Pose(sampled.x, sampled.y, 0.0)
                        if np.linalg.norm(reference_positions - [p.x, p.y], axis=1).min() >= 0.60:
                            break
                    else:
                        raise ValueError(
                            "Could not sample an unfamiliar location outside every memory"
                        )
                    condition = CONDITIONS[index % len(CONDITIONS)]
                    image = _perturb(arena.reset(p).rgb, condition, query_rng)
                    _append_query(values, image, p, scene_seed, split, condition, False, -1)
            print(f"Collected {split} scene {scene_seed}: {places_per_scene} places", flush=True)

    arrays = {}
    for name, items in values.items():
        if name in {"gallery", "observations", "query_images"}:
            arrays[name] = np.stack(items).astype(np.uint8)
        elif name.endswith("positions"):
            arrays[name] = np.asarray(items, dtype=np.float64)
        elif name.endswith("splits") or name == "query_conditions":
            arrays[name] = np.asarray(items, dtype="U12")
        elif name == "query_familiar":
            arrays[name] = np.asarray(items, dtype=np.bool_)
        else:
            arrays[name] = np.asarray(items, dtype=np.int64)
    np.savez_compressed(destination / "images.npz", **arrays)
    metadata = {
        "format_version": FORMAT_VERSION,
        "seed": seed,
        "scene_seeds": scenes,
        "camera_convention": CAMERA_CONVENTION,
        "image_shape": list(arrays["gallery"].shape[1:]),
        "places_per_scene": places_per_scene,
        "observations_per_place": observations_per_place,
        "unfamiliar_per_scene": unfamiliar_per_scene,
        "gallery_min_spacing_m": 0.50,
        "observation_max_jitter_m": 0.16,
        "query_max_jitter_m": 0.20,
        "unfamiliar_min_distance_m": 0.60,
        "retrieval_tolerance_m": 0.25,
        "conditions": list(CONDITIONS),
        "sensor_model": {
            "exposure_gain_range": [0.55, 1.45],
            "contrast_range": [0.75, 1.25],
            "channel_gain_range": [0.80, 1.20],
            "rgb_noise_std": 2.5,
            "occluded_camera_count": [1, 2],
            "patch_side_fraction_range": [1 / 3, 3 / 4],
            "scope": "post-render sensor perturbations, not physical relighting or obstacles",
        },
        "scene_configs": configs,
        "counts": {
            split: {
                "scenes": len(scenes[split]),
                "gallery": int(np.sum(arrays["gallery_splits"] == split)),
                "queries": int(np.sum(arrays["query_splits"] == split)),
                "familiar_queries": int(
                    np.sum((arrays["query_splits"] == split) & arrays["query_familiar"])
                ),
                "unfamiliar_queries": int(
                    np.sum((arrays["query_splits"] == split) & ~arrays["query_familiar"])
                ),
            }
            for split in scenes
        },
        "data_sha256": hashlib.sha256((destination / "images.npz").read_bytes()).hexdigest(),
        "scope": "within-scene retrieval and familiarity rejection, with scene-disjoint representation training",
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    validate_dataset(arrays, metadata)
    return metadata


def _append_query(values, image, pose, scene, split, condition, familiar, place_id):
    values["query_images"].append(image)
    values["query_positions"].append([pose.x, pose.y, pose.yaw])
    values["query_scene_ids"].append(scene)
    values["query_splits"].append(split)
    values["query_conditions"].append(condition)
    values["query_familiar"].append(familiar)
    values["query_place_ids"].append(place_id)


def validate_dataset(arrays: dict, metadata: dict) -> None:
    """Validate schema and spatial/split contracts without computing model scores."""
    if (
        metadata.get("format_version") != FORMAT_VERSION
        or metadata.get("camera_convention") != CAMERA_CONVENTION
    ):
        raise ValueError("Unsupported dataset format or camera convention")
    required = {
        "gallery",
        "observations",
        "gallery_positions",
        "observation_positions",
        "gallery_scene_ids",
        "gallery_splits",
        "place_ids",
        "query_images",
        "query_positions",
        "query_scene_ids",
        "query_splits",
        "query_conditions",
        "query_familiar",
        "query_place_ids",
    }
    if not required.issubset(arrays):
        raise ValueError("Dataset is missing required arrays")
    n, m = len(arrays["gallery"]), len(arrays["query_images"])
    shape = tuple(metadata.get("image_shape", ()))
    if (
        len(shape) != 3
        or any(not isinstance(value, int) or isinstance(value, bool) for value in shape)
        or shape[0] < 8
        or shape != (shape[0], 4 * shape[0], 3)
    ):
        raise ValueError("Image shape must describe four square RGB camera views")
    v = metadata.get("observations_per_place")
    if not isinstance(v, int) or isinstance(v, bool) or v < 2:
        raise ValueError("Require at least two observations per place")
    for key in ("places_per_scene", "unfamiliar_per_scene"):
        if (
            not isinstance(metadata.get(key), int)
            or isinstance(metadata[key], bool)
            or metadata[key] < 1
        ):
            raise ValueError(f"Invalid generation count: {key}")
    spatial_keys = (
        "gallery_min_spacing_m",
        "observation_max_jitter_m",
        "query_max_jitter_m",
        "unfamiliar_min_distance_m",
        "retrieval_tolerance_m",
    )
    for key in spatial_keys:
        value = metadata.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Spatial thresholds must be finite and positive: {key}")
    if not (
        metadata["observation_max_jitter_m"]
        <= metadata["query_max_jitter_m"]
        <= metadata["retrieval_tolerance_m"]
        < metadata["unfamiliar_min_distance_m"]
        and metadata["gallery_min_spacing_m"] > 2 * metadata["query_max_jitter_m"]
    ):
        raise ValueError("Spatial thresholds have inconsistent relationships")
    for name, expected in (
        ("gallery", (n, *shape)),
        ("observations", (n, v, *shape)),
        ("query_images", (m, *shape)),
    ):
        if arrays[name].dtype != np.uint8 or arrays[name].shape != expected:
            raise ValueError(f"Invalid RGB array shape or dtype: {name}")
    for name, expected in (
        ("gallery_positions", (n, 3)),
        ("observation_positions", (n, v, 3)),
        ("query_positions", (m, 3)),
    ):
        if arrays[name].shape != expected or not np.isfinite(arrays[name]).all():
            raise ValueError(f"Invalid spatial labels: {name}")
    for name in ("gallery_scene_ids", "gallery_splits", "place_ids"):
        if arrays[name].shape != (n,):
            raise ValueError(f"Invalid row labels: {name}")
    for name in (
        "query_scene_ids",
        "query_splits",
        "query_conditions",
        "query_familiar",
        "query_place_ids",
    ):
        if arrays[name].shape != (m,):
            raise ValueError(f"Invalid query labels: {name}")
    for name in ("gallery_scene_ids", "query_scene_ids", "place_ids", "query_place_ids"):
        if not np.issubdtype(arrays[name].dtype, np.integer):
            raise ValueError(f"IDs must use integer arrays: {name}")
    if any(
        np.any(arrays[name] < 0) for name in ("gallery_scene_ids", "query_scene_ids", "place_ids")
    ):
        raise ValueError("Scene and gallery place IDs must be nonnegative")
    if not n or not m or len(np.unique(arrays["place_ids"])) != n:
        raise ValueError("Empty dataset or nonunique place IDs")
    if arrays["query_familiar"].dtype != np.bool_ or set(
        arrays["query_conditions"].tolist()
    ) != set(CONDITIONS):
        raise ValueError("Invalid familiarity labels or query conditions")
    scenes = metadata["scene_seeds"]
    flattened = [scene for values in scenes.values() for scene in values]
    if set(scenes) != {"train", "val", "test"} or len(flattened) != len(set(flattened)):
        raise ValueError("Scene splits overlap or are missing")
    if any(not values for values in scenes.values()) or any(
        not isinstance(scene, int) or isinstance(scene, bool) or scene < 0 for scene in flattened
    ):
        raise ValueError("Declared scenes must be nonempty nonnegative integer lists")
    configs = metadata.get("scene_configs", [])
    if len(configs) != len(flattened) or {config.get("seed") for config in configs} != set(
        flattened
    ):
        raise ValueError("Scene configs differ from declared scenes")
    for config in configs:
        try:
            validated = SimConfig(**config)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid scene configuration") from error
        if validated.camera_size != shape[0]:
            raise ValueError("Scene camera size differs from image shape")
    for split in ("train", "val", "test"):
        g = arrays["gallery_splits"] == split
        q = arrays["query_splits"] == split
        if set(arrays["gallery_scene_ids"][g]) != set(scenes[split]) or set(
            arrays["query_scene_ids"][q]
        ) != set(scenes[split]):
            raise ValueError("Scene membership differs from metadata")
        expected_counts = {
            "scenes": len(scenes[split]),
            "gallery": int(g.sum()),
            "queries": int(q.sum()),
            "familiar_queries": int(np.sum(q & arrays["query_familiar"])),
            "unfamiliar_queries": int(np.sum(q & ~arrays["query_familiar"])),
        }
        if metadata.get("counts", {}).get(split) != expected_counts:
            raise ValueError("Sample counts differ from declared metadata")
    if set(arrays["gallery_splits"].tolist()) != set(scenes) or set(
        arrays["query_splits"].tolist()
    ) != set(scenes):
        raise ValueError("Unknown split label")
    id_to_row = {int(value): row for row, value in enumerate(arrays["place_ids"])}
    for scene in flattened:
        g = np.flatnonzero(arrays["gallery_scene_ids"] == scene)
        q = np.flatnonzero(arrays["query_scene_ids"] == scene)
        positions = arrays["gallery_positions"][g, :2]
        if (
            len(g) != metadata["places_per_scene"]
            or len(q) != len(g) * len(CONDITIONS) + metadata["unfamiliar_per_scene"]
        ):
            raise ValueError("Per-scene generation counts are inconsistent")
        distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        if np.min(distances) < metadata["gallery_min_spacing_m"] - 1e-8:
            raise ValueError("Gallery spacing contract violated")
        offsets = np.linalg.norm(
            arrays["observation_positions"][g, :, :2] - positions[:, None], axis=2
        )
        if np.max(offsets) > metadata["observation_max_jitter_m"] + 1e-8:
            raise ValueError("Observation jitter contract violated")
        for row in q:
            place_id = int(arrays["query_place_ids"][row])
            if arrays["query_familiar"][row]:
                if place_id not in id_to_row:
                    raise ValueError("Familiar query refers to a missing place")
                reference = id_to_row[place_id]
                if arrays["gallery_scene_ids"][reference] != scene:
                    raise ValueError("Familiar query refers to another scene")
                error = np.linalg.norm(
                    arrays["query_positions"][row, :2] - arrays["gallery_positions"][reference, :2]
                )
                if error > metadata["query_max_jitter_m"] + 1e-8:
                    raise ValueError("Familiar query jitter contract violated")
            elif (
                place_id != -1
                or np.linalg.norm(positions - arrays["query_positions"][row, :2], axis=1).min()
                < metadata["unfamiliar_min_distance_m"] - 1e-8
            ):
                raise ValueError("Unfamiliar query is inside the excluded memory radius")
        for gallery_row in g:
            reference_queries = q[arrays["query_place_ids"][q] == arrays["place_ids"][gallery_row]]
            if len(reference_queries) != len(CONDITIONS) or set(
                arrays["query_conditions"][reference_queries]
            ) != set(CONDITIONS):
                raise ValueError(
                    "Each gallery place requires exactly one familiar query per condition"
                )


def load_dataset(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    directory = Path(path)
    metadata = json.loads((directory / "metadata.json").read_text())
    digest = hashlib.sha256((directory / "images.npz").read_bytes()).hexdigest()
    if digest != metadata.get("data_sha256"):
        raise ValueError("Dataset hash mismatch")
    with np.load(directory / "images.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    validate_dataset(arrays, metadata)
    return arrays, metadata
