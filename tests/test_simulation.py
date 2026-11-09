"""Real renderer and movement-contract tests, including evaluator isolation."""

from dataclasses import fields
from math import atan2, hypot, pi

import numpy as np
import pytest

from ant_homing.simulation import (
    CAMERA_CONVENTION,
    Arena,
    Motion,
    NoiseConfig,
    Observation,
    Pose,
    SimConfig,
)


def test_real_panorama_is_reproducible_compass_aligned_and_position_dependent():
    with Arena(SimConfig(seed=7)) as first, Arena(SimConfig(seed=7)) as second:
        rgb = first.observe().rgb
        assert rgb.shape == (32, 128, 3)
        assert rgb.dtype == np.uint8 and rgb.std() > 10
        np.testing.assert_array_equal(rgb, second.observe().rgb)
        rotated = first.step(Motion(0, pi / 3))
        np.testing.assert_array_equal(rgb, rotated.rgb)
        assert rotated.compass_heading == pytest.approx(pi / 3)
        moved = first.step(Motion(0.4, -pi / 3))
        assert np.mean(np.abs(moved.rgb.astype(float) - rgb.astype(float))) > 1
        assert first.render_overhead(96).shape == (96, 96, 3)


def test_observation_has_no_true_coordinates_and_reset_odometry_is_relative():
    with Arena() as arena:
        observation = arena.reset(Pose(0.6, 0.4, 0.2))
        assert {field.name for field in fields(Observation)} == {
            "rgb",
            "odometry",
            "compass_heading",
            "collision",
        }
        assert observation.odometry == Pose(0, 0, pytest.approx(0.2))
        assert arena.true_pose.x == 0.6
        arena.step(Motion(0.3, 0))
        assert arena.observe().odometry.x < arena.true_pose.x


def test_panorama_tiles_follow_contiguous_compass_bearings(monkeypatch):
    with Arena() as arena:
        bearings = []

        def track_render(eye, target, up, size, fov=90):
            bearings.append(atan2(target[1] - eye[1], target[0] - eye[0]))
            return np.full((size, size, 3), len(bearings), dtype=np.uint8)

        monkeypatch.setattr(arena, "_render", track_render)
        image = arena.observe().rgb
        assert bearings == pytest.approx([0, -pi / 2, pi, pi / 2])
        assert image[0, ::32, 0].tolist() == [1, 2, 3, 4]
        assert CAMERA_CONVENTION == "compass-east-south-west-north-v1"


def test_wall_collision_rejects_whole_translation_but_applies_turn():
    with Arena() as arena:
        observation = arena.step(Motion(100, pi / 2))
        assert observation.collision
        assert arena.true_pose == Pose(0, 0, pi / 2)
        assert observation.odometry == Pose(0, 0, pi / 2)


def test_swept_landmark_collision_prevents_tunnelling_with_free_endpoints():
    with Arena(SimConfig(seed=19)) as arena:
        found = False
        # The evaluator may inspect geometry to construct a collision test;
        # controllers never receive these bounds.
        for x0, x1, y0, y1 in arena._obstacles:
            candidates = [
                (Pose(x0 - 0.3, (y0 + y1) / 2), Pose(x1 + 0.3, (y0 + y1) / 2)),
                (Pose((x0 + x1) / 2, y0 - 0.3), Pose((x0 + x1) / 2, y1 + 0.3)),
            ]
            for start, end in candidates:
                if arena._is_free(start.x, start.y) and arena._is_free(end.x, end.y):
                    bearing = atan2(end.y - start.y, end.x - start.x)
                    arena.reset(Pose(start.x, start.y, bearing))
                    observation = arena.step(Motion(hypot(end.x - start.x, end.y - start.y), 0))
                    assert observation.collision
                    assert arena.true_pose.x == start.x and arena.true_pose.y == start.y
                    found = True
                    break
            if found:
                break
        assert found, "Test scene must contain an obstacle with free opposing endpoints"


def test_odometry_noise_does_not_move_ground_truth_and_repeats_from_seed():
    noise = NoiseConfig(distance_std=0.02, heading_std=0.03, distance_bias=0.1, heading_bias=0.01)
    with Arena(noise=noise) as first, Arena(noise=noise) as second:
        a, b = first.step(Motion(0.5, 0.2)), second.step(Motion(0.5, 0.2))
        assert a.odometry == b.odometry
        assert first.true_pose == second.true_pose
        assert a.odometry != first.true_pose
        first.reset()
        assert first.step(Motion(0.5, 0.2)).odometry == a.odometry


def test_noise_seed_can_vary_without_changing_scene_or_true_movement():
    with (
        Arena(SimConfig(seed=8), noise=NoiseConfig(distance_std=0.1, seed=1)) as first,
        Arena(SimConfig(seed=8), noise=NoiseConfig(distance_std=0.1, seed=2)) as second,
    ):
        np.testing.assert_array_equal(first.observe().rgb, second.observe().rgb)
        a, b = first.step(Motion(0.5, 0.2)), second.step(Motion(0.5, 0.2))
        assert first.true_pose == second.true_pose
        assert a.odometry != b.odometry
        np.testing.assert_array_equal(a.rgb, b.rgb)


def test_invalid_poses_actions_and_closed_arena_fail_explicitly():
    arena = Arena()
    with pytest.raises(ValueError):
        arena.reset(Pose(10, 10, 0))
    with pytest.raises(ValueError):
        arena.step(Motion(float("nan"), 0))
    arena.close()
    arena.close()
    with pytest.raises(RuntimeError):
        arena.observe()


@pytest.mark.parametrize("config", [{"arena_size": 0}, {"camera_size": 0}, {"seed": -1}])
def test_invalid_config_fails_before_rendering(config):
    with pytest.raises(ValueError):
        SimConfig(**config)
