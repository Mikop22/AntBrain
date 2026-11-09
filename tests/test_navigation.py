"""Controller checks include physical probing and evaluator isolation."""

import hashlib
from math import cos, hypot, sin
from types import SimpleNamespace

import numpy as np
import pytest

from ant_homing.navigation import (
    normalized_image_distance,
    odometry_home,
    raw_image_distance,
    run_episode,
    visual_home,
)
from ant_homing.simulation import Motion, Pose, SimConfig


class SensorOnlyAgent:
    """Small actuator fake that rejects any evaluator-state access."""

    def __init__(self, x=0.3, y=0.15):
        self.x, self.y, self.heading = x, y, 0.0
        self.actions = []
        self.distance = 0.0

    @property
    def true_pose(self):
        raise AssertionError("Controller attempted evaluator-state access")

    def observe(self):
        image = np.empty((4, 4, 3), dtype=np.uint8)
        image[:] = (round(128 + self.x * 50), round(128 + self.y * 50), 20)
        return SimpleNamespace(
            rgb=image,
            odometry=Pose(self.x, self.y, self.heading),
            compass_heading=self.heading,
            collision=False,
        )

    def step(self, motion):
        self.heading += motion.turn
        self.x += motion.forward * cos(self.heading)
        self.y += motion.forward * sin(self.heading)
        self.distance += abs(motion.forward)
        self.actions.append(motion)
        if len(self.actions) > 250:
            raise AssertionError("Controller did not terminate")
        return self.observe()


def test_pixel_distances_and_brightness_normalization():
    image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3) + 10
    assert raw_image_distance(image, image) == 0.0
    assert normalized_image_distance(image, image + 20) < 1e-10
    assert raw_image_distance(image, image + 20) > 0.0


def test_odometry_controller_uses_sensor_interface_only():
    agent = SensorOnlyAgent()
    assert odometry_home(agent, stop_radius=0.01) == "odometry_arrival"
    assert hypot(agent.x, agent.y) <= 0.01
    assert agent.actions


def test_visual_controller_executes_and_counts_return_probes():
    agent = SensorOnlyAgent()
    home = SensorOnlyAgent(0.0, 0.0).observe().rgb
    direct_distance = hypot(agent.x, agent.y)
    reason = visual_home(agent, home, step_distance=0.15)
    assert reason == "visual_match"
    assert hypot(agent.x, agent.y) < 1e-6
    assert any(action.forward < 0 for action in agent.actions)
    assert agent.distance > direct_distance * 3


def test_flat_visual_landscape_stops_without_inventing_direction():
    agent = SensorOnlyAgent()
    reason = visual_home(agent, agent.observe().rgb, scorer=lambda *_: 1.0)
    assert reason == "no_visual_improvement"
    assert len(agent.actions) == 8  # Four physical forward/backward probes.
    assert hypot(agent.x - 0.3, agent.y - 0.15) < 1e-6


def test_episode_budget_is_enforced_and_trajectory_is_reproducible():
    first = run_episode(SimConfig(seed=21), "raw", max_steps=4)
    second = run_episode(SimConfig(seed=21), "raw", max_steps=4)
    assert first["summary"]["steps"] == 4
    assert first["summary"]["stop_reason"] == "step_limit"
    assert first["trajectory"] == second["trajectory"]
    assert first["summary"]["path_length"] >= 1.0
    assert first["summary"]["memory_bytes"] == first["home_rgb"].nbytes


def test_zero_noise_odometry_returns_and_records_all_movement():
    result = run_episode(SimConfig(seed=22), "odometry", capture_frames=True)
    assert result["summary"]["success"]
    assert result["summary"]["endpoint_error"] < 0.05
    assert result["summary"]["stop_reason"] == "odometry_arrival"
    assert result["summary"]["memory_bytes"] == 0
    assert len(result["frames"]) == len(result["trajectory"])
    assert result["summary"]["path_length"] > 1.5


def test_success_does_not_stop_controller_at_hidden_goal():
    # A zero-length outbound route begins at home. The evaluator must not
    # preempt the controller: it still reports the sensor-based arrival reason.
    result = run_episode(SimConfig(seed=22), "odometry", outbound=[])
    assert result["summary"]["stop_reason"] == "odometry_arrival"
    assert result["summary"]["steps"] == 0


@pytest.mark.parametrize("kwargs", [{"max_steps": 0}, {"step_distance": 0}, {"success_radius": 0}])
def test_invalid_episode_parameters_fail_early(kwargs):
    with pytest.raises(ValueError):
        run_episode(SimConfig(), **kwargs)


def test_blocked_outbound_at_home_is_not_a_success():
    result = run_episode(SimConfig(seed=22), "odometry", outbound=[Motion(100.0, 0.0)])
    assert result["summary"]["stop_reason"] == "outbound_blocked"
    assert result["summary"]["goal_within_radius"]
    assert not result["summary"]["outbound_completed"]
    assert not result["summary"]["success"]


def test_custom_outbound_route_is_preserved_in_summary():
    route = [Motion(0.3, 0.0), Motion(0.15, 0.5)]
    result = run_episode(SimConfig(seed=22), "odometry", outbound=route)
    assert result["summary"]["outbound_route"] == [
        {"forward": 0.3, "turn": 0.0},
        {"forward": 0.15, "turn": 0.5},
    ]
    assert result["summary"]["checkpoint_sha256"] is None


def test_cnn_checkpoint_content_hash_is_recorded(tmp_path, monkeypatch):
    # Scoring is mocked here to isolate provenance from neural-model training.
    checkpoint = tmp_path / "model.pt"
    content = b"checkpoint provenance fixture"
    checkpoint.write_bytes(content)
    monkeypatch.setattr("ant_homing.learning.load_image_scorer", lambda _path: raw_image_distance)
    result = run_episode(SimConfig(seed=22), "cnn", checkpoint=checkpoint, max_steps=4)
    assert result["summary"]["checkpoint_sha256"] == hashlib.sha256(content).hexdigest()
    assert result["summary"]["checkpoint_file_bytes"] == len(content)
