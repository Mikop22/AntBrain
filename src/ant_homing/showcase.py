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

from .search import (
    FoodSensor,
    PillarSearchPolicy,
    food_candidates,
    inspection_sites,
)
from .search import (
    segment_intersects as _intersects,
)
from .simulation import Arena, Motion, NoiseConfig, Pose, SimConfig

_HOME = Pose(-0.9, -0.6, 1.7)
_CLEARANCE = 0.28
_STEP = 0.055
_TURN = 0.18
_DT = 0.18


def _angle(value: float) -> float:
    return float((value + pi) % (2 * pi) - pi)


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


def _select_food(planner: _Planner, seed: int = 0) -> tuple[np.ndarray, int]:
    """World-side random placement; this selected target never enters the policy."""
    candidates = food_candidates(planner, (_HOME.x, _HOME.y))
    pillars = sorted({pillar for pillar, _ in candidates})
    if not pillars:
        raise ValueError(
            "No accessible far-side food position has safe ant/cargo clearance in this scene"
        )
    rng = np.random.default_rng(seed)
    selected_pillar = int(rng.choice(pillars))
    points = [point for pillar, point in candidates if pillar == selected_pillar]
    return np.asarray(points[int(rng.integers(len(points)))]), selected_pillar


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


class _ActionBudget(Exception):
    pass


class _FoodFound(Exception):
    pass


def run_showcase(
    config: SimConfig | None = None,
    *,
    capture_frames: bool = True,
    food_seed: int = 0,
    search_seed: int = 0,
    max_steps: int = 2000,
) -> dict:
    """Search without target knowledge, then carry locally detected food home.

    Food placement and search have independent seeds. Geometry remains known
    for obstacle-safe routing; the search policy receives no selected food
    location. An ideal local range/occlusion sensor is the sole detection gate.
    """
    for name, value in (("food_seed", food_seed), ("search_seed", search_seed)):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    config = config or SimConfig()
    records, frames, overviews = [], [], []
    state, phase = "on_ground", "outbound"
    total, outbound_length, return_length = 0.0, 0.0, 0.0
    detected = None
    first_detection = None
    pickup_position = None
    pickup_completed = delivery_completed = False
    empty_inspections = []
    outbound_waypoints, return_route = [], []
    active_site = None
    stop_reason = "search_exhausted"
    with Arena(config, noise=NoiseConfig()) as arena:
        observation = arena.reset(_HOME)
        planner = _Planner(arena)
        food, pillar_index = _select_food(planner, food_seed)
        sensor = FoodSensor(food, planner.pillars, detection_range=0.6)
        policy = PillarSearchPolicy(inspection_sites(planner, (_HOME.x, _HOME.y)), seed=search_seed)
        scene = _ShowcaseScene(arena, food)

        def estimated_position():
            return np.array([_HOME.x + observation.odometry.x, _HOME.y + observation.odometry.y])

        def record(motion: Motion | None = None):
            nonlocal total, outbound_length, return_length, observation, detected, first_detection
            pose = arena.true_pose
            visible = sensor.observe((pose.x, pose.y)) if state == "on_ground" else None
            if detected is None and visible is not None:
                detected = visible
                first_detection = {
                    "step": len(records),
                    "pose": asdict(pose),
                    "position": list(visible.position),
                    "distance_m": visible.distance,
                }
            if records:
                old = records[-1]["true_pose"]
                distance = hypot(pose.x - old["x"], pose.y - old["y"])
                total += distance
                if phase == "outbound":
                    outbound_length += distance
                elif phase == "returning":
                    return_length += distance
            position = scene.update(pose, state, len(records))
            if capture_frames:
                observation = arena.observe()
            records.append(
                {
                    "step": len(records),
                    "time_s": len(records) * _DT,
                    "phase": phase,
                    "true_pose": asdict(pose),
                    "odometry": asdict(observation.odometry),
                    "collision": observation.collision,
                    "action": asdict(motion) if motion is not None else None,
                    "carrying": state == "carried",
                    "food_state": state,
                    "food_position": position,
                    "food_detected": detected is not None,
                    "food_visible": visible is not None,
                    "inspection_pillar": active_site.pillar_index if active_site else None,
                    "completed_inspection_count": len(empty_inspections),
                }
            )
            if capture_frames:
                frames.append(observation.rgb.copy())
                overviews.append(arena.render_overhead(size=520))

        def step(motion: Motion, *, searching: bool = False):
            nonlocal observation
            if len(records) - 1 >= max_steps:
                raise _ActionBudget
            observation = arena.step(motion)
            if observation.collision:
                raise ValueError("Showcase route blocked; select a different scene configuration")
            record(motion)
            if searching and detected is not None:
                raise _FoodFound

        def follow(route, *, searching=False):
            for target in _movement_points(route):
                delta = target - estimated_position()
                heading = atan2(delta[1], delta[0])
                while abs(turn := _angle(heading - observation.compass_heading)) > _TURN:
                    step(Motion(0.0, float(np.clip(turn, -_TURN, _TURN))), searching=searching)
                step(Motion(float(np.linalg.norm(delta)), turn), searching=searching)

        record()
        home_rgb = observation.rgb.copy()
        try:
            while detected is None:
                active_site = policy.next_site(estimated_position())
                if active_site is None:
                    break
                route = planner.route(estimated_position(), active_site.position)
                outbound_waypoints.extend(point.tolist() for point in route[1:])
                try:
                    follow(route, searching=True)
                except _FoodFound:
                    break
                policy.complete(active_site)
                empty_inspections.append(
                    {
                        "pillar_index": active_site.pillar_index,
                        "site_index": active_site.site_index,
                        "step": records[-1]["step"],
                    }
                )
            if detected is not None:
                # Target-directed movement is allowed only after the sensor
                # has released a nearby unobstructed food location.
                active_site = None
                approach = planner.route(estimated_position(), detected.position)
                outbound_waypoints.extend(point.tolist() for point in approach[1:])
                follow(approach)
                if hypot(arena.true_pose.x - food[0], arena.true_pose.y - food[1]) > 0.12:
                    raise RuntimeError("Showcase did not reach its pickup radius")
                phase, state = "pickup", "carried"
                pickup_completed = True
                pickup_position = asdict(arena.true_pose)
                for _ in range(3):
                    step(Motion())
                phase = "returning"
                return_route = planner.route(estimated_position(), np.array([_HOME.x, _HOME.y]))
                follow(return_route)
                phase, state = "delivered", "delivered"
                delivery_completed = True
                for _ in range(4):
                    step(Motion())
                stop_reason = "food_delivered"
        except _ActionBudget:
            # A state transition is real only once its corresponding frame was
            # recorded. Exhausting a budget during display holds must not
            # invent a pickup or erase a delivery that already happened.
            state = records[-1]["food_state"]
            pickup_completed = state in {"carried", "delivered"}
            delivery_completed = state == "delivered"
            if not pickup_completed:
                pickup_position = None
            stop_reason = "food_delivered" if delivery_completed else "step_limit"
        endpoint = hypot(arena.true_pose.x - _HOME.x, arena.true_pose.y - _HOME.y)
        final_rgb = arena.observe().rgb.copy()
    inspected_pillars = list(dict.fromkeys(row["pillar_index"] for row in empty_inspections))
    summary = {
        "schema_version": 2,
        "method": "food_blind_pillar_search",
        "route_strategy": "ideal_scene_geometry_planner",
        "search_strategy": "nearest_unvisited_pillar_sites",
        "demonstration_mode": "ideal_foraging",
        "config": asdict(config),
        "noise": asdict(NoiseConfig()),
        "home_pose": asdict(_HOME),
        "food_seed": food_seed,
        "search_seed": search_seed,
        "food_initial_position": {"x": float(food[0]), "y": float(food[1]), "z": 0.075},
        "food_pillar_index": pillar_index,
        "food_pillar_bounds": list(planner.pillars[pillar_index]),
        "food_hidden_from_nest": True,
        "food_known_before_detection": False,
        "detection_model": "ideal proximity and unobstructed line of sight",
        "detection_range_m": sensor.detection_range,
        "food_detected_step": first_detection["step"] if first_detection else None,
        "first_detection": first_detection,
        "inspected_pillars": inspected_pillars,
        "empty_pillars": [pillar for pillar in inspected_pillars if pillar != pillar_index],
        "empty_inspections": empty_inspections,
        "empty_inspection_count": len(empty_inspections),
        "inspection_catalogue": [asdict(site) for site in policy.sites],
        "planner_clearance_m": planner.clearance,
        "outbound_waypoints": outbound_waypoints,
        "return_waypoints": [p.tolist() for p in return_route],
        "pickup_position": pickup_position,
        "pickup_completed": pickup_completed,
        "delivery_completed": delivery_completed,
        "food_state": state,
        "success": delivery_completed and endpoint <= 0.05,
        "success_radius": 0.05,
        "endpoint_error": endpoint,
        "stop_reason": stop_reason,
        "steps": len(records) - 1,
        "max_steps": max_steps,
        "path_length": total,
        "outbound_path_length": outbound_length,
        "return_path_length": return_length,
        "collisions": 0,
        "snapshots": 1,
        "memory_bytes": int(home_rgb.nbytes),
        "ideal_assumptions": [
            "food-blind geometry-derived inspection policy with visited-site memory",
            "ideal local proximity and line-of-sight food sensor",
            "ideal full-scene geometry for obstacle routing only",
            "noiseless displacement integration",
            "perfect compass",
            "ideal actuation",
            "visual-only cargo attachment",
            "engine demonstration; no learned visual detection or biological validation",
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
