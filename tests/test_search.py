"""Test perception and memory without needing a rendered world."""

import pytest

from ant_homing.search import FoodSensor, InspectionSite, PillarSearchPolicy


def test_sensor_releases_only_a_near_unobstructed_target():
    sensor = FoodSensor((0.5, 0.0), [(0.20, 0.30, -0.2, 0.2)], detection_range=0.6)
    assert sensor.observe((0.0, 0.0)) is None  # Nearby, but behind a pillar.
    assert sensor.observe((-1.0, 1.0)) is None  # Beyond sensor range.
    detection = sensor.observe((0.5, 0.3))
    assert detection.position == (0.5, 0.0)
    assert detection.distance == pytest.approx(0.3)


def test_policy_visits_each_site_once_and_exhausts_memory():
    sites = [
        InspectionSite(0, 0, (1, 0)),
        InspectionSite(0, 1, (2, 0)),
        InspectionSite(1, 0, (3, 0)),
    ]
    policy = PillarSearchPolicy(sites)
    pose = (0, 0)
    visited = []
    while (site := policy.next_site(pose)) is not None:
        visited.append(site)
        policy.complete(site)
        pose = site.position
    assert visited == sites
    assert len(policy.visited) == 3
    with pytest.raises(ValueError, match="already completed"):
        policy.complete(sites[0])


def test_sensor_and_policy_reject_invalid_configuration():
    with pytest.raises(ValueError, match="range"):
        FoodSensor((0, 0), [], detection_range=-1)
    with pytest.raises(ValueError, match="seed"):
        PillarSearchPolicy([], seed=-1)
