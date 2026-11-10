"""Food-blind pillar inspection and an explicitly ideal local food sensor.

The policy sees scene geometry and remembers inspection sites. Only the world
sensor owns the selected food location; it releases a target after range and
line-of-sight checks succeed. This is not learned image recognition.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, sin

import numpy as np


def segment_intersects(a, b, bounds) -> bool:
    """Test a closed floor-plane segment against an axis-aligned pillar."""
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


@dataclass(frozen=True)
class InspectionSite:
    pillar_index: int
    site_index: int
    position: tuple[float, float]


@dataclass(frozen=True)
class FoodDetection:
    position: tuple[float, float]
    distance: float


class FoodSensor:
    """World-side range/occlusion sensor; no target is returned before detection."""

    def __init__(self, food, pillars, *, detection_range: float = 0.6):
        if not np.isfinite(detection_range) or detection_range <= 0:
            raise ValueError("Food detection range must be finite and positive")
        self._food = tuple(float(v) for v in food)
        self._pillars = tuple(tuple(b) for b in pillars)
        self.detection_range = detection_range

    def observe(self, position) -> FoodDetection | None:
        distance = hypot(position[0] - self._food[0], position[1] - self._food[1])
        if distance > self.detection_range:
            return None
        if any(segment_intersects(position, self._food, box) for box in self._pillars):
            return None
        return FoodDetection(self._food, distance)


def food_candidates(planner, home) -> list[tuple[int, tuple[float, float]]]:
    """Find reachable far-side food pockets without consulting a search policy."""
    candidates = []
    for index, bounds in enumerate(planner.pillars):
        x0, x1, y0, y1 = bounds
        center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        angle = atan2(center[1] - home[1], center[0] - home[0])
        for offset in (0.0, -0.15, 0.15, -0.3, 0.3, -0.45, 0.45):
            direction = np.array([cos(angle + offset), sin(angle + offset)])
            exit_distance = min(
                ((x1 - x0) / 2 + planner.clearance) / max(abs(direction[0]), 1e-12),
                ((y1 - y0) / 2 + planner.clearance) / max(abs(direction[1]), 1e-12),
            )
            point = center + direction * (exit_distance + 0.045)
            if not planner.free(point) or not segment_intersects(home, point, bounds):
                continue
            try:
                planner.route(home, point)
            except ValueError:
                continue
            candidates.append((index, (float(point[0]), float(point[1]))))
    return candidates


def inspection_sites(planner, home) -> list[InspectionSite]:
    """Build the same reachable inspection catalogue for every food placement.

    Sites include near/far faces and corners around each expanded pillar. Far-
    side pockets are added for *all* pillars, not just the selected food pillar.
    """
    locations: list[tuple[int, tuple[float, float]]] = []
    for index, (x0, x1, y0, y1) in enumerate(planner.boxes):
        xmid, ymid = (x0 + x1) / 2, (y0 + y1) / 2
        for x, y in (
            (x0 - 0.045, ymid),
            (x1 + 0.045, ymid),
            (xmid, y0 - 0.045),
            (xmid, y1 + 0.045),
            (x0 - 0.045, y0 - 0.045),
            (x0 - 0.045, y1 + 0.045),
            (x1 + 0.045, y0 - 0.045),
            (x1 + 0.045, y1 + 0.045),
        ):
            if not planner.free((x, y)):
                continue
            try:
                planner.route(home, (x, y))
            except ValueError:
                continue
            locations.append((index, (float(x), float(y))))
    locations.extend(food_candidates(planner, home))
    sites, counters = [], {}
    for pillar, position in locations:
        if any(
            s.pillar_index == pillar
            and hypot(s.position[0] - position[0], s.position[1] - position[1]) < 0.10
            for s in sites
        ):
            continue
        site_index = counters.get(pillar, 0)
        sites.append(InspectionSite(pillar, site_index, position))
        counters[pillar] = site_index + 1
    return sites


class PillarSearchPolicy:
    """Inspect the nearest unvisited pillar, remembering each completed site.

    This interface deliberately accepts no food position, selected pillar, or
    sensor object. An external runner may interrupt its actions on detection.
    """

    def __init__(self, sites: list[InspectionSite], *, seed: int = 0):
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("Search seed must be a nonnegative integer")
        self.sites = tuple(sites)
        self.visited: set[tuple[int, int]] = set()
        order = np.random.default_rng(seed).permutation(len(sites))
        self._tie_rank = {int(i): rank for rank, i in enumerate(order)}
        self._active_pillar: int | None = None

    def next_site(self, position) -> InspectionSite | None:
        remaining = [
            (index, site)
            for index, site in enumerate(self.sites)
            if (site.pillar_index, site.site_index) not in self.visited
        ]
        same_pillar = [entry for entry in remaining if entry[1].pillar_index == self._active_pillar]
        pool = same_pillar or remaining
        if not pool:
            return None
        _, selected = min(
            pool,
            key=lambda entry: (
                hypot(position[0] - entry[1].position[0], position[1] - entry[1].position[1]),
                self._tie_rank[entry[0]],
            ),
        )
        self._active_pillar = selected.pillar_index
        return selected

    def complete(self, site: InspectionSite) -> None:
        if site not in self.sites:
            raise ValueError("Unknown inspection site")
        key = site.pillar_index, site.site_index
        if key in self.visited:
            raise ValueError("Inspection site was already completed")
        self.visited.add(key)
