"""Exercise the ideal foraging scenario without weakening research semantics."""

from math import hypot

import numpy as np
import pytest

from ant_homing.showcase import run_showcase
from ant_homing.simulation import SimConfig


@pytest.fixture(scope="module")
def episode():
    return run_showcase(capture_frames=False)


def test_foraging_has_long_search_and_direct_noiseless_return(episode):
    summary = episode["summary"]
    assert summary["demonstration_mode"] == "ideal_foraging"
    assert summary["method"] == "ideal_path_integration"
    assert summary["success"] and summary["collisions"] == 0
    assert summary["endpoint_error"] < 1e-8
    assert summary["outbound_path_length"] > 7.0
    home, pickup = summary["home_pose"], summary["pickup_position"]
    direct = hypot(pickup["x"] - home["x"], pickup["y"] - home["y"])
    assert summary["return_path_length"] == pytest.approx(direct)
    assert summary["path_length"] == pytest.approx(
        summary["outbound_path_length"] + summary["return_path_length"]
    )
    assert "noiseless displacement integration" in summary["ideal_assumptions"]


def test_object_lifecycle_and_visible_attachment(episode):
    records = episode["trajectory"]
    phases = list(dict.fromkeys(record["phase"] for record in records))
    assert phases == ["outbound", "pickup", "returning", "delivered"]
    for record in records:
        assert record["carrying"] == (record["food_state"] == "carried")
        if record["phase"] in {"pickup", "returning"}:
            assert record["carrying"]
            ant, food = record["true_pose"], record["food_position"]
            assert hypot(ant["x"] - food["x"], ant["y"] - food["y"]) == pytest.approx(0.19)
    delivered = records[-1]
    assert delivered["food_state"] == "delivered" and not delivered["carrying"]
    home, food = episode["summary"]["home_pose"], delivered["food_position"]
    assert hypot(home["x"] - food["x"], home["y"] - food["y"]) < 0.2


def test_no_reverse_probing_and_bounded_turning(episode):
    records = episode["trajectory"]
    for record in records[1:]:
        assert record["action"]["forward"] >= 0
        assert abs(record["action"]["turn"]) <= 0.18 + 1e-10
        assert record["action"]["forward"] <= 0.055 + 1e-8
    assert len(records) < 501
    assert records[-1]["time_s"] > 30


def test_actual_rendered_views_and_overviews_stay_synchronized():
    episode = run_showcase(SimConfig(camera_size=8), capture_frames=True)
    assert len(episode["frames"]) == len(episode["overview_frames"]) == len(episode["trajectory"])
    assert episode["frames"][0].shape == (8, 32, 3)
    assert episode["overview_frames"][0].shape == (520, 520, 3)
    assert not np.array_equal(episode["overview_frames"][0], episode["overview_frames"][-1])
    assert all(not row["collision"] for row in episode["trajectory"])


def test_minimum_arena_and_reproducibility(episode):
    minimum = run_showcase(SimConfig(arena_size=4, seed=3), capture_frames=False)
    assert minimum["summary"]["success"] and minimum["summary"]["collisions"] == 0
    replay = run_showcase(capture_frames=False)
    assert replay["trajectory"] == episode["trajectory"]
