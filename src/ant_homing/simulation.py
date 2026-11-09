"""Deterministic, headless 3D vision with planar kinematic movement.

The agent observes RGB, dead reckoning and an explicitly ideal compass. Exact
poses and the overview renderer belong to data collection/evaluation only.
This is deliberately not a wheel/terrain physics model. PyBullet renders the
scene; conservative swept square-footprint checks prevent wall tunnelling.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Self

import numpy as np
import pybullet as bullet

CAMERA_CONVENTION = "compass-east-south-west-north-v1"


def _angle(value: float) -> float:
    return float((value + pi) % (2 * pi) - pi)


@dataclass(frozen=True)
class Pose:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


@dataclass(frozen=True)
class Motion:
    """Turn in radians, then translate in metres along the new heading."""

    forward: float = 0.0
    turn: float = 0.0


@dataclass(frozen=True)
class SimConfig:
    seed: int = 0
    arena_size: float = 6.0
    camera_size: int = 32
    camera_height: float = 0.30
    agent_radius: float = 0.12
    landmark_count: int = 14

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.camera_size, int) or self.camera_size < 8:
            raise ValueError("camera_size must be an integer of at least 8")
        if not isinstance(self.landmark_count, int) or self.landmark_count < 0:
            raise ValueError("landmark_count must be nonnegative")
        if (
            not np.isfinite([self.arena_size, self.camera_height, self.agent_radius]).all()
            or self.arena_size < 4.0
            or self.camera_height <= 0
            or not 0 < self.agent_radius < 0.4
        ):
            raise ValueError(
                "arena_size >= 4, camera_height > 0 and 0 < agent_radius < .4 required"
            )


@dataclass(frozen=True)
class NoiseConfig:
    """Dead-reckoning error only; true movement remains kinematic.

    Distance std is metres per accepted translation; distance bias is a scale
    fraction. Heading std/bias are radians per nonzero action. The independent
    exact compass remains available, but does not silently correct odometry.
    The noise seed is independent of the scene seed. Reset replays its stream;
    use different seeds for independent repeated trials in the same scene.
    """

    distance_std: float = 0.0
    heading_std: float = 0.0
    distance_bias: float = 0.0
    heading_bias: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Noise seed must be a nonnegative integer")
        values = [self.distance_std, self.heading_std, self.distance_bias, self.heading_bias]
        if not np.isfinite(values).all() or min(self.distance_std, self.heading_std) < 0:
            raise ValueError("Noise parameters must be finite with nonnegative standard deviations")


@dataclass(frozen=True)
class Observation:
    rgb: np.ndarray
    odometry: Pose
    compass_heading: float
    collision: bool


class Arena:
    """Own one isolated Bullet client; use as a context manager or call close()."""

    def __init__(self, config: SimConfig | None = None, noise: NoiseConfig | None = None):
        self.config = config or SimConfig()
        self.noise = noise or NoiseConfig()
        self._client = bullet.connect(bullet.DIRECT)
        if self._client < 0:
            raise RuntimeError("Could not create headless PyBullet client")
        self._rng = np.random.default_rng(np.random.SeedSequence([self.noise.seed, 9127]))
        self._obstacles: list[tuple[float, float, float, float]] = []
        self._pose = Pose()
        self._odometry = Pose()
        self._collision = False
        self._build_scene()
        self.reset()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        if self._client is not None:
            bullet.disconnect(physicsClientId=self._client)
            self._client = None

    def _require_open(self) -> None:
        if self._client is None:
            raise RuntimeError("Arena is closed")

    def _box(self, position, half_extents, color) -> None:
        visual = bullet.createVisualShape(
            bullet.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[*color, 1.0],
            physicsClientId=self._client,
        )
        bullet.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual,
            basePosition=position,
            physicsClientId=self._client,
        )

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.config.seed)
        half = self.config.arena_size / 2
        self._box([0, 0, -0.06], [half + 0.1, half + 0.1, 0.06], [0.58, 0.57, 0.53])
        # Differently coloured walls and seeded landmarks avoid a trivially
        # rotationally symmetric arena while retaining a reproducible scene.
        wall_colors = [
            [0.72, 0.78, 0.83],
            [0.82, 0.74, 0.64],
            [0.69, 0.77, 0.67],
            [0.77, 0.69, 0.77],
        ]
        for i, (x, y, hx, hy) in enumerate(
            [
                (half + 0.05, 0, 0.05, half + 0.1),
                (-half - 0.05, 0, 0.05, half + 0.1),
                (0, half + 0.05, half + 0.1, 0.05),
                (0, -half - 0.05, half + 0.1, 0.05),
            ]
        ):
            self._box([x, y, 0.65], [hx, hy, 0.65], wall_colors[i])
        for i in range(self.config.landmark_count):
            angle = 2 * pi * (i + rng.uniform(-0.28, 0.28)) / max(1, self.config.landmark_count)
            # Keep the default outbound route and local homing probes clear.
            radius = rng.uniform(max(1.65, half * 0.65), half - 0.33)
            x, y = radius * cos(angle), radius * sin(angle)
            hx, hy = rng.uniform(0.10, 0.23, 2)
            height = float(rng.uniform(0.45, 1.15))
            color = rng.uniform(0.10, 0.95, 3)
            self._box([x, y, height / 2], [hx, hy, height / 2], color.tolist())
            # A contrasting upper band gives nearby views vertical structure.
            self._box([x, y, height * 0.80], [hx + 0.005, hy + 0.005, 0.04], (1 - color).tolist())
            self._obstacles.append(
                (
                    float(x - hx - 0.005),
                    float(x + hx + 0.005),
                    float(y - hy - 0.005),
                    float(y + hy + 0.005),
                )
            )

    @property
    def true_pose(self) -> Pose:
        """Evaluator/data-collector ground truth; never pass this to a controller."""
        self._require_open()
        return self._pose

    def _is_free(self, x: float, y: float) -> bool:
        r = self.config.agent_radius
        limit = self.config.arena_size / 2 - r
        if abs(x) >= limit or abs(y) >= limit:
            return False
        return not any(
            x0 - r <= x <= x1 + r and y0 - r <= y <= y1 + r for x0, x1, y0, y1 in self._obstacles
        )

    def reset(self, pose: Pose | None = None) -> Observation:
        """Place the agent and start dead reckoning at relative position zero."""
        self._require_open()
        pose = pose or Pose()
        if not np.isfinite([pose.x, pose.y, pose.yaw]).all() or not self._is_free(pose.x, pose.y):
            raise ValueError("Reset pose must be finite and collision-free")
        self._pose = Pose(float(pose.x), float(pose.y), _angle(pose.yaw))
        self._odometry = Pose(0.0, 0.0, self._pose.yaw)
        self._collision = False
        self._rng = np.random.default_rng(np.random.SeedSequence([self.noise.seed, 9127]))
        return self.observe()

    def sample_free_pose(self, rng: np.random.Generator) -> Pose:
        """Collection-only helper; pose coordinates are not an agent observation."""
        self._require_open()
        limit = self.config.arena_size / 2 - self.config.agent_radius - 0.05
        for _ in range(10000):
            x, y = rng.uniform(-limit, limit, 2)
            if self._is_free(float(x), float(y)):
                return Pose(float(x), float(y), float(rng.uniform(-pi, pi)))
        raise RuntimeError("Could not sample a free pose")

    def _segment_collides(self, start: Pose, end: Pose) -> bool:
        if not self._is_free(end.x, end.y):
            return True
        # Slab intersection against expanded rectangles catches large moves
        # through an obstacle even when both endpoints are collision-free.
        r = self.config.agent_radius
        dx, dy = end.x - start.x, end.y - start.y
        for x0, x1, y0, y1 in self._obstacles:
            lower, upper = 0.0, 1.0
            for origin, delta, lo, hi in (
                (start.x, dx, x0 - r, x1 + r),
                (start.y, dy, y0 - r, y1 + r),
            ):
                if abs(delta) < 1e-12:
                    if origin < lo or origin > hi:
                        lower, upper = 1.0, 0.0
                        break
                else:
                    a, b = (lo - origin) / delta, (hi - origin) / delta
                    lower, upper = max(lower, min(a, b)), min(upper, max(a, b))
            if lower <= upper:
                return True
        return False

    def step(self, motion: Motion) -> Observation:
        self._require_open()
        if not np.isfinite([motion.forward, motion.turn]).all():
            raise ValueError("Motion must contain finite numbers")
        yaw = _angle(self._pose.yaw + motion.turn)
        candidate = Pose(
            self._pose.x + motion.forward * cos(yaw), self._pose.y + motion.forward * sin(yaw), yaw
        )
        self._collision = self._segment_collides(self._pose, candidate)
        distance = 0.0 if self._collision else motion.forward
        self._pose = Pose(self._pose.x, self._pose.y, yaw) if self._collision else candidate
        changed = motion.forward != 0.0 or motion.turn != 0.0
        turn_error = (
            (self.noise.heading_bias + self._rng.normal(0, self.noise.heading_std))
            if changed
            else 0.0
        )
        measured_yaw = _angle(self._odometry.yaw + motion.turn + turn_error)
        measured_distance = distance
        if distance:
            measured_distance = distance * (1 + self.noise.distance_bias) + self._rng.normal(
                0, self.noise.distance_std
            )
        self._odometry = Pose(
            self._odometry.x + measured_distance * cos(measured_yaw),
            self._odometry.y + measured_distance * sin(measured_yaw),
            measured_yaw,
        )
        return self.observe()

    def _render(self, eye, target, up, size: int, fov: float = 90.0) -> np.ndarray:
        self._require_open()
        view = bullet.computeViewMatrix(eye, target, up)
        projection = bullet.computeProjectionMatrixFOV(fov, 1.0, 0.02, self.config.arena_size * 4)
        result = bullet.getCameraImage(
            width=size,
            height=size,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=bullet.ER_TINY_RENDERER,
            lightDirection=[-1, -2, 5],
            shadow=0,
            physicsClientId=self._client,
        )
        return np.asarray(result[2], dtype=np.uint8).reshape(size, size, 4)[..., :3].copy()

    def observe(self) -> Observation:
        self._require_open()
        eye = [self._pose.x, self._pose.y, self.config.camera_height]
        views = []
        # Cardinal views are aligned by an ideal external compass. The camera
        # panorama therefore does not rotate with the agent's body heading.
        # PyBullet's right-hand view coordinate makes each image run from
        # heading+45 degrees at left to heading-45 at right. E,S,W,N order
        # therefore gives adjacent seams instead of a scrambled 4-view strip.
        for yaw in (0.0, -pi / 2, pi, pi / 2):
            target = [eye[0] + cos(yaw), eye[1] + sin(yaw), eye[2]]
            views.append(self._render(eye, target, [0, 0, 1], self.config.camera_size))
        return Observation(
            np.concatenate(views, axis=1), self._odometry, self._pose.yaw, self._collision
        )

    def render_overhead(self, size: int = 384) -> np.ndarray:
        """Evaluator-only RGB overview; not supplied to the navigation agent."""
        return self._render([0, 0, self.config.arena_size * 0.95], [0, 0, 0], [0, 1, 0], size, 64)
