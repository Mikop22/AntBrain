"""Ideal foraging showcase: scripted search, visible pickup, noiseless homing.

This module demonstrates the engine and a perfect path integrator. It does not
train a policy, model ant biomechanics, or replace the research controllers.
"""

from __future__ import annotations

from dataclasses import asdict
from math import atan2, ceil, cos, hypot, pi, sin

import numpy as np
import pybullet as bullet

from .simulation import Arena, Motion, NoiseConfig, Pose, SimConfig

_HOME = Pose(-0.9, -0.6, 1.7)
_FOOD = (0.95, 0.1)
_STEP = 0.055
_TURN = 0.18
_DT = 0.18


def _angle(value: float) -> float:
    return float((value + pi) % (2 * pi) - pi)


def _outbound_points() -> list[np.ndarray]:
    """Smooth a reproducible demonstration route with Catmull-Rom segments."""
    controls = np.asarray(
        [
            [_HOME.x, _HOME.y],
            [-1.0, 0.2],
            [-0.55, 0.95],
            [0.35, 1.0],
            [0.95, 0.45],
            [0.75, -0.35],
            [0.05, -0.75],
            [-0.5, -0.15],
            [0.1, 0.35],
            list(_FOOD),
        ]
    )
    padded = np.vstack([controls[0], controls, controls[-1]])
    dense = []
    for index in range(1, len(padded) - 2):
        a, b, c, d = padded[index - 1 : index + 3]
        for t in np.linspace(0, 1, 80, endpoint=False):
            dense.append(
                0.5
                * (
                    (2 * b)
                    + (-a + c) * t
                    + (2 * a - 5 * b + 4 * c - d) * t**2
                    + (-a + 3 * b - 3 * c + d) * t**3
                )
            )
    dense.append(controls[-1])
    # Resample by arclength so apparent speed is consistent along each curve.
    dense = np.asarray(dense)
    lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    samples = np.linspace(0, lengths[-1], ceil(lengths[-1] / _STEP) + 1)
    return [
        np.array([np.interp(s, lengths, dense[:, axis]) for axis in (0, 1)]) for s in samples[1:]
    ]


class _ShowcaseScene:
    """Visual-only ant, nest, and cargo owned by one showcase arena client."""

    def __init__(self, arena: Arena):
        self.arena = arena
        self.client = arena._client  # Isolated scene decoration; never used by a controller.
        self.parts: list[tuple[int, float, float, float]] = []
        for x, radius, color in (
            (-0.10, 0.075, [0.12, 0.08, 0.06]),
            (0.0, 0.05, [0.18, 0.09, 0.05]),
            (0.085, 0.055, [0.24, 0.10, 0.045]),
        ):
            self.parts.append((self._shape(bullet.GEOM_SPHERE, color, radius=radius), x, 0, 0.085))
        self.legs = [
            self._shape(bullet.GEOM_CAPSULE, [0.12, 0.07, 0.035], radius=0.012, length=0.14)
            for _ in range(6)
        ]
        self.antennae = [
            self._shape(bullet.GEOM_CAPSULE, [0.18, 0.10, 0.04], radius=0.008, length=0.10)
            for _ in range(2)
        ]
        self.food = self._shape(bullet.GEOM_SPHERE, [1.0, 0.68, 0.03], radius=0.075)
        nest = self._shape(bullet.GEOM_CYLINDER, [0.33, 0.19, 0.08], radius=0.21, length=0.012)
        hole = self._shape(bullet.GEOM_CYLINDER, [0.055, 0.035, 0.02], radius=0.12, length=0.014)
        for body in (nest, hole):
            self._place(body, _HOME.x, _HOME.y, 0.01, 0.0)

    def _shape(self, kind: int, color: list, **kwargs) -> int:
        visual = bullet.createVisualShape(
            kind, rgbaColor=[*color, 1.0], physicsClientId=self.client, **kwargs
        )
        return bullet.createMultiBody(
            baseMass=0, baseVisualShapeIndex=visual, physicsClientId=self.client
        )

    def _place(
        self, body: int, x: float, y: float, z: float, yaw: float, *, capsule: bool = False
    ) -> None:
        rotation = bullet.getQuaternionFromEuler([0, pi / 2 if capsule else 0, yaw])
        bullet.resetBasePositionAndOrientation(
            body, [x, y, z], rotation, physicsClientId=self.client
        )

    def update(self, pose: Pose, state: str, step: int) -> dict:
        def world(x, y):
            return pose.x + x * cos(pose.yaw) - y * sin(pose.yaw), pose.y + x * sin(
                pose.yaw
            ) + y * cos(pose.yaw)

        for body, x, y, z in self.parts:
            wx, wy = world(x, y)
            self._place(body, wx, wy, z, pose.yaw)
        for index, body in enumerate(self.legs):
            side = 1 if index < 3 else -1
            row = index % 3
            swing = 0.12 * sin(step * 0.8 + (index % 2) * pi)
            angle = side * (pi / 2 + (row - 1) * 0.55) + swing
            x, y = world(0.04 - row * 0.065 + 0.055 * cos(angle), 0.055 * sin(angle))
            self._place(body, x, y, 0.035, pose.yaw + angle, capsule=True)
        for index, body in enumerate(self.antennae):
            angle = (1 if index else -1) * 0.45
            x, y = world(0.145, (index * 2 - 1) * 0.04)
            self._place(body, x, y, 0.10, pose.yaw + angle, capsule=True)
        if state == "carried":
            x, y = world(0.19, 0.0)
            position = {"x": x, "y": y, "z": 0.105}
        elif state == "delivered":
            position = {"x": _HOME.x + 0.11, "y": _HOME.y + 0.04, "z": 0.08}
        else:
            position = {"x": _FOOD[0], "y": _FOOD[1], "z": 0.075}
        self._place(self.food, **position, yaw=0.0)
        return position


def run_showcase(config: SimConfig | None = None, *, capture_frames: bool = True) -> dict:
    """Run a successful ideal pickup-and-delivery scenario through real steps.

    The scripted outbound path locates the food. Once acquired, only the exact
    accumulated displacement and compass determine the return heading. True
    position is retained for scene decoration, pickup checks, and evaluation.
    """
    config = config or SimConfig()
    records, frames, overviews = [], [], []
    state, phase = "on_ground", "outbound"
    total, outbound_length, return_length = 0.0, 0.0, 0.0
    with Arena(config, noise=NoiseConfig()) as arena:
        arena.reset(_HOME)
        scene = _ShowcaseScene(arena)

        def record(motion: Motion | None = None):
            nonlocal total, outbound_length, return_length
            pose = arena.true_pose
            if records:
                old = records[-1]["true_pose"]
                distance = hypot(pose.x - old["x"], pose.y - old["y"])
                total += distance
                if phase == "outbound":
                    outbound_length += distance
                elif phase == "returning":
                    return_length += distance
            position = scene.update(pose, state, len(records))
            observation = arena.observe()
            records.append(
                {
                    "step": len(records),
                    "time_s": len(records) * _DT,
                    "phase": phase,
                    "true_pose": asdict(pose),
                    "odometry": asdict(observation.odometry),
                    "collision": observation.collision,
                    "action": asdict(motion) if motion else None,
                    "carrying": state == "carried",
                    "food_state": state,
                    "food_position": position,
                }
            )
            if capture_frames:
                frames.append(observation.rgb.copy())
                overviews.append(arena.render_overhead(size=520))

        def step(motion: Motion):
            observation = arena.step(motion)
            if observation.collision:
                raise ValueError("Showcase route blocked; select a different scene configuration")
            record(motion)
            if len(records) > 500:
                raise RuntimeError("Showcase exceeded its fixed action budget")

        def turn_to(heading: float):
            while abs(delta := _angle(heading - arena.observe().compass_heading)) > _TURN:
                step(Motion(0.0, float(np.clip(delta, -_TURN, _TURN))))
            return delta

        record()
        home_rgb = arena.observe().rgb.copy()
        # Outbound coordinates belong to this explicit scripted engine showcase.
        for target in _outbound_points():
            pose = arena.true_pose
            delta = target - np.array([pose.x, pose.y])
            remaining_turn = turn_to(atan2(delta[1], delta[0]))
            step(Motion(float(np.linalg.norm(delta)), remaining_turn))
        if hypot(arena.true_pose.x - _FOOD[0], arena.true_pose.y - _FOOD[1]) > 0.12:
            raise RuntimeError("Showcase did not reach its pickup radius")
        phase, state = "pickup", "carried"
        for _ in range(3):
            step(Motion())
        pickup_position = asdict(arena.true_pose)
        phase = "returning"
        # Home is the origin of measured displacement, not an oracle map input.
        while True:
            estimate = arena.observe().odometry
            distance = hypot(estimate.x, estimate.y)
            if distance <= 1e-8:
                break
            heading = atan2(-estimate.y, -estimate.x)
            remaining_turn = turn_to(heading)
            step(Motion(min(_STEP, distance), remaining_turn))
        phase, state = "delivered", "delivered"
        for _ in range(4):
            step(Motion())
        endpoint = hypot(arena.true_pose.x - _HOME.x, arena.true_pose.y - _HOME.y)
        final_rgb = arena.observe().rgb.copy()
    summary = {
        "schema_version": 1,
        "method": "ideal_path_integration",
        "demonstration_mode": "ideal_foraging",
        "config": asdict(config),
        "noise": asdict(NoiseConfig()),
        "home_pose": asdict(_HOME),
        "food_initial_position": {"x": _FOOD[0], "y": _FOOD[1], "z": 0.075},
        "pickup_position": pickup_position,
        "pickup_completed": True,
        "delivery_completed": True,
        "food_state": state,
        "success": endpoint <= 0.05,
        "success_radius": 0.05,
        "endpoint_error": endpoint,
        "stop_reason": "food_delivered",
        "steps": len(records) - 1,
        "max_steps": 500,
        "path_length": total,
        "outbound_path_length": outbound_length,
        "return_path_length": return_length,
        "collisions": 0,
        "snapshots": 1,
        "memory_bytes": int(home_rgb.nbytes),
        "ideal_assumptions": [
            "scripted curved outbound route",
            "noiseless displacement integration",
            "perfect compass",
            "ideal actuation",
            "visual-only cargo attachment",
            "engine demonstration; no learned policy or biological validation",
        ],
    }
    return {
        "summary": summary,
        "trajectory": records,
        "frames": frames,
        "overview_frames": overviews,
        "home_rgb": home_rgb,
        "final_rgb": final_rgb,
    }
