"""Ideal map-informed foraging: hidden food, visible pickup and obstacle detours.

This module demonstrates the engine using exact scene geometry and a visibility
graph planner. It does not train a policy, model ant biomechanics, or replace
the research controllers.
"""

from __future__ import annotations

from dataclasses import asdict
from heapq import heappop, heappush
from itertools import pairwise
from math import atan2, ceil, cos, hypot, pi, sin

import numpy as np
import pybullet as bullet

from .simulation import Arena, Motion, NoiseConfig, Pose, SimConfig

_HOME = Pose(-0.9, -0.6, 1.7)
_CLEARANCE = 0.28
_STEP = 0.055
_TURN = 0.18
_DT = 0.18


def _angle(value: float) -> float:
    return float((value + pi) % (2 * pi) - pi)


def _intersects(a, b, bounds) -> bool:
    """Closed segment/rectangle intersection using a two-dimensional slab test."""
    lower, upper = 0.0, 1.0
    for origin, delta, lo, hi in (
        (a[0], b[0] - a[0], bounds[0], bounds[1]),
        (a[1], b[1] - a[1], bounds[2], bounds[3]),
    ):
        if abs(delta) < 1e-12:
            if origin < lo or origin > hi:
                return False
        else:
            x, y = (lo - origin) / delta, (hi - origin) / delta
            lower, upper = max(lower, min(x, y)), min(upper, max(x, y))
    return lower <= upper


class _Planner:
    """Visibility-graph planner for this explicitly map-informed ideal showcase.

    Obstacles are expanded to enclose the ant and its carried food for every
    heading, including in-place turns. This is not a learned navigation policy.
    """

    def __init__(self, arena: Arena):
        self.pillars = list(arena._obstacles)
        self.clearance = max(_CLEARANCE, arena.config.agent_radius)
        self.limit = arena.config.arena_size / 2 - self.clearance - 0.02
        r = self.clearance
        self.boxes = [(x0 - r, x1 + r, y0 - r, y1 + r) for x0, x1, y0, y1 in self.pillars]
        corners = []
        for x0, x1, y0, y1 in self.boxes:
            for x in (x0 - 0.015, x1 + 0.015):
                for y in (y0 - 0.015, y1 + 0.015):
                    if self.free((x, y)):
                        corners.append(np.asarray([x, y], dtype=float))
        self.corners = corners
        self.edges = [[] for _ in corners]
        for i, a in enumerate(corners):
            for j in range(i):
                b = corners[j]
                if self.visible(a, b):
                    distance = float(np.linalg.norm(a - b))
                    self.edges[i].append((j, distance))
                    self.edges[j].append((i, distance))

    def free(self, p) -> bool:
        return (
            abs(p[0]) < self.limit
            and abs(p[1]) < self.limit
            and not any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for x0, x1, y0, y1 in self.boxes)
        )

    def visible(self, a, b) -> bool:
        return (
            self.free(a) and self.free(b) and not any(_intersects(a, b, box) for box in self.boxes)
        )

    def route(self, start, goal) -> list[np.ndarray]:
        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        if not self.free(start) or not self.free(goal):
            raise ValueError("Showcase waypoint lacks ant/cargo clearance")
        if self.visible(start, goal):
            return [start, goal]
        points = self.corners + [start, goal]
        n = len(self.corners)
        edges = [list(e) for e in self.edges] + [[], []]
        for i in (n, n + 1):
            for j in range(i):
                if self.visible(points[i], points[j]):
                    d = float(np.linalg.norm(points[i] - points[j]))
                    edges[i].append((j, d))
                    edges[j].append((i, d))
        queue = [(0.0, n)]
        best, previous = {n: 0.0}, {}
        while queue:
            cost, current = heappop(queue)
            if cost > best[current]:
                continue
            if current == n + 1:
                indices = [current]
                while indices[-1] != n:
                    indices.append(previous[indices[-1]])
                return [points[i] for i in reversed(indices)]
            for neighbor, distance in edges[current]:
                candidate = cost + distance
                if candidate < best.get(neighbor, float("inf")):
                    best[neighbor] = candidate
                    previous[neighbor] = current
                    heappush(queue, (candidate, neighbor))
        raise ValueError("No collision-free showcase route through this scene")


def _select_food(planner: _Planner) -> tuple[np.ndarray, int]:
    home = np.asarray([_HOME.x, _HOME.y])
    candidates = []
    for index, bounds in enumerate(planner.pillars):
        x0, x1, y0, y1 = bounds
        center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        angle = atan2(*(center - home)[::-1])
        for offset in (0.0, -0.15, 0.15, -0.3, 0.3, -0.45, 0.45):
            direction = np.array([cos(angle + offset), sin(angle + offset)])
            exit_distance = min(
                ((x1 - x0) / 2 + planner.clearance) / max(abs(direction[0]), 1e-12),
                ((y1 - y0) / 2 + planner.clearance) / max(abs(direction[1]), 1e-12),
            )
            food = center + direction * (exit_distance + 0.045)
            # Hidden means a sightline intersects the actual pillar, not merely
            # the expanded navigation footprint.
            if planner.free(food) and _intersects(home, food, bounds):
                candidates.append((index, food))
    for index, food in candidates:
        try:
            planner.route(home, food)
        except ValueError:
            continue
        return food, index
    raise ValueError(
        "No accessible far-side food position has safe ant/cargo clearance in this scene"
    )


def _outbound_route(planner: _Planner, food: np.ndarray) -> list[np.ndarray]:
    """Weave across the arena, routing every leg around actual pillars."""
    extent = min(2.05, planner.limit - 0.15)
    desired = [
        (-0.8, -0.75),
        (-0.85, 0.7),
        (0.75, 0.8),
        (0.8, -0.7),
        (-0.4, -0.55),
        (-0.75, 0.4),
        (0.4, 0.55),
    ]
    # Nearby free grid samples provide deterministic fallbacks when a desired
    # exploration point coincides with a seeded pillar.
    samples = [
        np.array([x, y])
        for x in np.linspace(-extent, extent, 19)
        for y in np.linspace(-extent, extent, 19)
        if planner.free((x, y))
    ]
    result = [np.array([_HOME.x, _HOME.y])]
    for fraction in desired:
        target = np.asarray(fraction) * extent
        for waypoint in sorted(samples, key=lambda p: float(np.linalg.norm(p - target))):
            try:
                leg = planner.route(result[-1], waypoint)
            except ValueError:
                continue
            result.extend(leg[1:])
            break
    result.extend(planner.route(result[-1], food)[1:])
    return result


def _movement_points(route) -> list[np.ndarray]:
    points = []
    for start, end in pairwise(route):
        count = max(1, ceil(float(np.linalg.norm(end - start)) / _STEP))
        points.extend(start + (end - start) * t for t in np.linspace(0, 1, count + 1)[1:])
    return points


class _ShowcaseScene:
    """Visual-only ant, nest, and cargo owned by one showcase arena client."""

    def __init__(self, arena: Arena, food_position: np.ndarray):
        self.food_position = food_position
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
            position = {
                "x": float(self.food_position[0]),
                "y": float(self.food_position[1]),
                "z": 0.075,
            }
        self._place(self.food, **position, yaw=0.0)
        return position


def run_showcase(config: SimConfig | None = None, *, capture_frames: bool = True) -> dict:
    """Run a successful ideal pickup-and-delivery scenario through real steps.

    Food is placed behind a seeded pillar. Scripted exploration and an ideal
    geometry-informed planner execute obstacle-free outbound and return routes.
    True scene geometry is explicitly available to this demonstration planner.
    """
    config = config or SimConfig()
    records, frames, overviews = [], [], []
    state, phase = "on_ground", "outbound"
    total, outbound_length, return_length = 0.0, 0.0, 0.0
    with Arena(config, noise=NoiseConfig()) as arena:
        arena.reset(_HOME)
        planner = _Planner(arena)
        food, pillar_index = _select_food(planner)
        outbound_route = _outbound_route(planner, food)
        return_route = planner.route(food, np.array([_HOME.x, _HOME.y]))
        scene = _ShowcaseScene(arena, food)

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
            if len(records) > 1400:
                raise RuntimeError("Showcase exceeded its fixed action budget")

        def turn_to(heading: float):
            while abs(delta := _angle(heading - arena.observe().compass_heading)) > _TURN:
                step(Motion(0.0, float(np.clip(delta, -_TURN, _TURN))))
            return delta

        record()
        home_rgb = arena.observe().rgb.copy()
        # Outbound coordinates belong to this explicit scripted engine showcase.
        for target in _movement_points(outbound_route):
            pose = arena.true_pose
            delta = target - np.array([pose.x, pose.y])
            remaining_turn = turn_to(atan2(delta[1], delta[0]))
            step(Motion(float(np.linalg.norm(delta)), remaining_turn))
        if hypot(arena.true_pose.x - food[0], arena.true_pose.y - food[1]) > 0.12:
            raise RuntimeError("Showcase did not reach its pickup radius")
        phase, state = "pickup", "carried"
        for _ in range(3):
            step(Motion())
        pickup_position = asdict(arena.true_pose)
        phase = "returning"
        # This engine showcase intentionally uses ideal scene geometry to
        # detour around the pillar. The research controllers are unchanged.
        for target in _movement_points(return_route):
            pose = arena.true_pose
            delta = target - np.array([pose.x, pose.y])
            remaining_turn = turn_to(atan2(delta[1], delta[0]))
            step(Motion(float(np.linalg.norm(delta)), remaining_turn))
        phase, state = "delivered", "delivered"
        for _ in range(4):
            step(Motion())
        endpoint = hypot(arena.true_pose.x - _HOME.x, arena.true_pose.y - _HOME.y)
        final_rgb = arena.observe().rgb.copy()
    summary = {
        "schema_version": 1,
        "method": "ideal_obstacle_navigation",
        "route_strategy": "ideal_scene_geometry_planner",
        "demonstration_mode": "ideal_foraging",
        "config": asdict(config),
        "noise": asdict(NoiseConfig()),
        "home_pose": asdict(_HOME),
        "food_initial_position": {"x": float(food[0]), "y": float(food[1]), "z": 0.075},
        "food_pillar_index": pillar_index,
        "food_pillar_bounds": list(planner.pillars[pillar_index]),
        "food_hidden_from_nest": True,
        "planner_clearance_m": planner.clearance,
        "outbound_waypoints": [p.tolist() for p in outbound_route],
        "return_waypoints": [p.tolist() for p in return_route],
        "pickup_position": pickup_position,
        "pickup_completed": True,
        "delivery_completed": True,
        "food_state": state,
        "success": endpoint <= 0.05,
        "success_radius": 0.05,
        "endpoint_error": endpoint,
        "stop_reason": "food_delivered",
        "steps": len(records) - 1,
        "max_steps": 1400,
        "path_length": total,
        "outbound_path_length": outbound_length,
        "return_path_length": return_length,
        "collisions": 0,
        "snapshots": 1,
        "memory_bytes": int(home_rgb.nbytes),
        "ideal_assumptions": [
            "scripted exploration waypoints",
            "ideal full-scene geometry and visibility-graph obstacle planning",
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
