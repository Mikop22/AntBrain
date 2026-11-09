"""Baseline homing controllers and evaluator-separated episode recording.

Controllers receive camera images, dead reckoning and a compass only. The episode
recorder accesses true poses to calculate metrics, never to choose actions.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import asdict
from math import atan2, hypot, pi
from pathlib import Path
from typing import Protocol

import numpy as np

from .simulation import Arena, Motion, NoiseConfig, Observation, Pose, SimConfig

ImageScorer = Callable[[np.ndarray, np.ndarray], float]


class AgentInterface(Protocol):
    """The restricted simulator interface exposed to navigation controllers."""

    def observe(self) -> Observation: ...

    def step(self, motion: Motion) -> Observation: ...


def raw_image_distance(current: np.ndarray, reference: np.ndarray) -> float:
    """Mean squared RGB difference on the [0, 1] intensity scale."""
    difference = (current.astype(np.float32) - reference.astype(np.float32)) / 255.0
    return float(np.mean(difference * difference))


def normalized_image_distance(current: np.ndarray, reference: np.ndarray) -> float:
    """RGB difference after global brightness/contrast normalization."""

    def normalize(image: np.ndarray) -> np.ndarray:
        pixels = image.astype(np.float32) / 255.0
        return (pixels - pixels.mean()) / max(float(pixels.std()), 1e-6)

    difference = normalize(current) - normalize(reference)
    return float(np.mean(difference * difference))


def _turn_to(heading: float, target: float) -> float:
    return float((target - heading + pi) % (2.0 * pi) - pi)


def odometry_home(
    agent: AgentInterface,
    *,
    step_distance: float = 0.15,
    stop_radius: float = 0.04,
) -> str:
    """Drive towards the dead-reckoned origin; no true position is consulted.

    The measured position estimate is relative to the episode's starting pose.
    An external action budget must bound the supplied interface's ``step``.
    """
    if (
        not np.isfinite(step_distance)
        or step_distance <= 0
        or not np.isfinite(stop_radius)
        or stop_radius < 0
    ):
        raise ValueError("step_distance must be positive and stop_radius nonnegative")
    while True:
        observation = agent.observe()
        estimate = observation.odometry
        distance = hypot(estimate.x, estimate.y)
        if distance <= stop_radius:
            return "odometry_arrival"
        heading = atan2(-estimate.y, -estimate.x)
        observation = agent.step(
            Motion(
                forward=min(step_distance, distance),
                turn=_turn_to(observation.compass_heading, heading),
            )
        )
        if observation.collision:
            return "blocked"


def visual_home(
    agent: AgentInterface,
    reference: np.ndarray,
    *,
    scorer: ImageScorer = raw_image_distance,
    step_distance: float = 0.15,
    match_threshold: float = 0.0,
    min_improvement: float = 1e-7,
) -> str:
    """Descend a visual-familiarity score using physically executed probes.

    Each candidate is reached by an actual action and followed by an actual
    backward action. No images are queried at unvisited locations. Probe and
    recovery costs are therefore part of the measured episode trajectory.
    """
    if (
        not np.isfinite(step_distance)
        or step_distance <= 0
        or not np.isfinite(match_threshold)
        or match_threshold < 0
        or not np.isfinite(min_improvement)
        or min_improvement < 0
    ):
        raise ValueError(
            "Require positive finite step distance and nonnegative finite score tolerances"
        )
    bearings = (0.0, pi / 2.0, pi, -pi / 2.0)
    while True:
        observation = agent.observe()
        base_score = scorer(observation.rgb, reference)
        if not np.isfinite(base_score):
            return "invalid_visual_score"
        if base_score <= match_threshold:
            return "visual_match"
        best_score = base_score
        best_bearing = None
        for bearing in bearings:
            observation = agent.observe()
            probe = agent.step(
                Motion(step_distance, _turn_to(observation.compass_heading, bearing))
            )
            # Arena rejects an entire colliding translation, so no reverse
            # translation is needed when the forward probe is blocked.
            if probe.collision:
                continue
            score = scorer(probe.rgb, reference)
            returned = agent.step(Motion(-step_distance, 0.0))
            if returned.collision:
                return "probe_recovery_blocked"
            if not np.isfinite(score):
                return "invalid_visual_score"
            if score < best_score - min_improvement:
                best_score = score
                best_bearing = bearing
        if best_bearing is None:
            return "no_visual_improvement"
        observation = agent.observe()
        selected = agent.step(
            Motion(step_distance, _turn_to(observation.compass_heading, best_bearing))
        )
        if selected.collision:
            return "blocked"


class _BudgetExhausted(Exception):
    pass


class _EpisodeRecorder:
    """Evaluator wrapper; controllers use its observation/action interface only."""

    def __init__(self, arena: Arena, max_steps: int, capture_frames: bool):
        self._arena = arena
        self.max_steps = max_steps
        self.capture_frames = capture_frames
        self.phase = "outbound"
        self.trajectory: list[dict] = []
        self.frames: list[np.ndarray] = []
        self.path_length = 0.0
        self.collisions = 0
        self._previous_pose = arena.true_pose
        self._record(arena.observe(), None)

    def observe(self) -> Observation:
        return self._arena.observe()

    def step(self, motion: Motion) -> Observation:
        if len(self.trajectory) - 1 >= self.max_steps:
            raise _BudgetExhausted
        observation = self._arena.step(motion)
        self._record(observation, motion)
        return observation

    def _record(self, observation: Observation, motion: Motion | None) -> None:
        pose = self._arena.true_pose
        self.path_length += hypot(pose.x - self._previous_pose.x, pose.y - self._previous_pose.y)
        self._previous_pose = pose
        self.collisions += int(observation.collision and motion is not None)
        self.trajectory.append(
            {
                "step": len(self.trajectory),
                "phase": self.phase,
                "true_pose": asdict(pose),
                "odometry": asdict(observation.odometry),
                "collision": bool(observation.collision),
                "action": asdict(motion) if motion is not None else None,
            }
        )
        if self.capture_frames:
            self.frames.append(observation.rgb.copy())


def run_episode(
    config: SimConfig,
    method: str = "odometry",
    *,
    noise: NoiseConfig | None = None,
    max_steps: int = 160,
    step_distance: float = 0.15,
    success_radius: float = 0.25,
    checkpoint: str | Path | None = None,
    capture_frames: bool = False,
    outbound: Sequence[Motion] | None = None,
    match_threshold: float = 0.0,
) -> dict:
    """Run a prescribed route and return metrics, trace and optional RGB frames.

    ``goal_within_radius`` reports final geometry. ``success`` additionally
    requires completed outbound motion and a nonfailure controller stop. Reaching
    a hidden goal never stops the controller.
    All actions, including outbound motion and visual probes, count against
    ``max_steps``. The default route stays in the arena's reserved open centre.
    """
    if method not in {"odometry", "raw", "visual", "normalized", "cnn", "hybrid"}:
        raise ValueError(f"Unknown navigation method: {method}")
    if (
        max_steps < 1
        or not np.isfinite(step_distance)
        or step_distance <= 0
        or not np.isfinite(success_radius)
        or success_radius <= 0
        or not np.isfinite(match_threshold)
        or match_threshold < 0
    ):
        raise ValueError("max_steps, step_distance and success_radius must be positive")
    if method == "cnn" and checkpoint is None:
        raise ValueError("CNN navigation requires a trained checkpoint")
    scorer: ImageScorer = raw_image_distance
    checkpoint_sha256 = None
    checkpoint_file_bytes = 0
    if method == "normalized":
        scorer = normalized_image_distance
    elif method == "cnn":
        from .learning import load_image_scorer

        scorer = load_image_scorer(checkpoint)
        with Path(checkpoint).open("rb") as checkpoint_file:
            checkpoint_sha256 = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
        checkpoint_file_bytes = Path(checkpoint).stat().st_size
    route = list(outbound) if outbound is not None else [Motion(0.60, 0.0), Motion(0.45, pi / 2.0)]
    with Arena(config, noise=noise) as arena:
        arena.reset(Pose(0.0, 0.0, 0.0))
        home = arena.true_pose
        home_rgb = arena.observe().rgb.copy()
        recorder = _EpisodeRecorder(arena, max_steps, capture_frames)
        stop_reason = "not_started"
        outbound_completed = False
        homing_started = False
        try:
            for motion in route:
                observation = recorder.step(motion)
                if observation.collision:
                    stop_reason = "outbound_blocked"
                    break
            else:
                outbound_completed = True
                homing_started = True
                recorder.phase = "homing"
                if method == "odometry":
                    stop_reason = odometry_home(recorder, step_distance=step_distance)
                else:
                    if method == "hybrid":
                        stop_reason = odometry_home(
                            recorder, step_distance=step_distance, stop_radius=0.35
                        )
                    if stop_reason != "blocked":
                        stop_reason = visual_home(
                            recorder,
                            home_rgb,
                            scorer=scorer,
                            step_distance=step_distance,
                            match_threshold=match_threshold,
                        )
        except _BudgetExhausted:
            stop_reason = "step_limit"
        final_pose = arena.true_pose
        endpoint_error = hypot(final_pose.x - home.x, final_pose.y - home.y)
        final_rgb = arena.observe().rgb.copy()
        homing_records = [p for p in recorder.trajectory if p["phase"] == "homing"]
        goal_within_radius = bool(endpoint_error <= success_radius)
        failed_stops = {
            "step_limit",
            "outbound_blocked",
            "blocked",
            "probe_recovery_blocked",
            "invalid_visual_score",
        }
        summary = {
            "schema_version": 1,
            "method": method,
            "outbound_route": [asdict(motion) for motion in route],
            "config": asdict(config),
            "noise": asdict(noise) if noise is not None else asdict(NoiseConfig()),
            "outbound_completed": outbound_completed,
            "homing_started": homing_started,
            "goal_within_radius": goal_within_radius,
            "success": bool(
                outbound_completed
                and homing_started
                and goal_within_radius
                and stop_reason not in failed_stops
            ),
            "success_radius": float(success_radius),
            "endpoint_error": float(endpoint_error),
            "stop_reason": stop_reason,
            "steps": len(recorder.trajectory) - 1,
            "max_steps": max_steps,
            "homing_actions": len(homing_records),
            "path_length": float(recorder.path_length),
            "collisions": recorder.collisions,
            "snapshots": 0 if method == "odometry" else 1,
            "memory_bytes": 0 if method == "odometry" else int(home_rgb.nbytes),
            "memory_scope": "stored RGB snapshot only; excludes controller state and model weights",
            "checkpoint_file_bytes": checkpoint_file_bytes,
            "checkpoint_sha256": checkpoint_sha256,
            "step_distance": float(step_distance),
            "match_threshold": float(match_threshold),
            "checkpoint": str(checkpoint) if method == "cnn" else None,
            "assumptions": [
                "planar kinematic motion",
                "perfect compass",
                "compass-aligned RGB panorama",
            ],
        }
        return {
            "summary": summary,
            "trajectory": recorder.trajectory,
            "frames": recorder.frames,
            "home_rgb": home_rgb,
            "final_rgb": final_rgb,
        }
