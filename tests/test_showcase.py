"""Exercise the ideal foraging scenario without weakening research semantics."""

from itertools import pairwise
from math import hypot

import numpy as np
import pytest

from ant_homing.showcase import _intersects, _Planner, _select_food, run_showcase
from ant_homing.simulation import Arena, SimConfig


@pytest.fixture(scope="module")
def episode():
    return run_showcase(capture_frames=False)


def test_foraging_has_long_search_and_obstacle_aware_return(episode):
    summary = episode["summary"]
    assert summary["demonstration_mode"] == "ideal_foraging"
    assert summary["method"] == "food_blind_pillar_search"
    assert summary["success"] and summary["collisions"] == 0
    assert summary["endpoint_error"] < 1e-8
    assert summary["outbound_path_length"] > 10.0
    assert len(summary["empty_pillars"]) >= 2
    assert summary["food_known_before_detection"] is False
    home, pickup = summary["home_pose"], summary["pickup_position"]
    direct = hypot(pickup["x"] - home["x"], pickup["y"] - home["y"])
    assert summary["return_path_length"] > direct + 0.1
    assert summary["route_strategy"] == "ideal_scene_geometry_planner"
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
    assert len(records) <= episode["summary"]["max_steps"] + 1
    assert records[-1]["time_s"] > 30


def test_actual_rendered_views_and_overviews_stay_synchronized():
    episode = run_showcase(SimConfig(camera_size=8), capture_frames=True, max_steps=8)
    assert len(episode["frames"]) == len(episode["overview_frames"]) == len(episode["trajectory"])
    assert episode["frames"][0].shape == (8, 32, 3)
    assert episode["overview_frames"][0].shape == (520, 520, 3)
    assert not np.array_equal(episode["overview_frames"][0], episode["overview_frames"][-1])
    assert episode["summary"]["stop_reason"] == "step_limit"
    assert not episode["summary"]["pickup_completed"]
    assert all(not row["collision"] for row in episode["trajectory"])


def test_food_is_hidden_behind_a_real_pillar_and_routes_clear_cargo(episode):
    summary = episode["summary"]
    home = summary["home_pose"]
    food = summary["food_initial_position"]
    assert _intersects(
        (home["x"], home["y"]), (food["x"], food["y"]), summary["food_pillar_bounds"]
    )
    with Arena(SimConfig(**summary["config"])) as arena:
        planner = _Planner(arena)
        assert planner.free((food["x"], food["y"]))
        for previous, current in pairwise(episode["trajectory"]):
            a, b = previous["true_pose"], current["true_pose"]
            assert planner.visible((a["x"], a["y"]), (b["x"], b["y"]))
            cargo = current["food_position"]
            assert all(
                not (
                    x0 - 0.075 <= cargo["x"] <= x1 + 0.075
                    and y0 - 0.075 <= cargo["y"] <= y1 + 0.075
                )
                for x0, x1, y0, y1 in planner.pillars
            )


def test_tight_arena_fails_clearly_and_reproducibility(episode):
    with (
        Arena(SimConfig(arena_size=4, seed=0)) as arena,
        pytest.raises(ValueError, match="No accessible far-side food position"),
    ):
        _select_food(_Planner(arena))
    replay = run_showcase(capture_frames=False)
    assert replay["trajectory"] == episode["trajectory"]


def test_food_seed_changes_location_without_changing_predetection_actions(episode):
    step = episode["summary"]["food_detected_step"]
    alternate = run_showcase(food_seed=1, capture_frames=False, max_steps=step + 2)
    assert (
        alternate["summary"]["food_initial_position"] != episode["summary"]["food_initial_position"]
    )
    assert alternate["summary"]["food_detected_step"] is None
    for left, right in zip(
        episode["trajectory"][:step], alternate["trajectory"][:step], strict=True
    ):
        assert left["action"] == right["action"]
        assert left["true_pose"] == right["true_pose"]
        assert not left["food_detected"] and not right["food_detected"]


def test_inspection_memory_and_first_detection_are_honest(episode):
    from ant_homing.search import FoodSensor

    summary = episode["summary"]
    inspections = summary["empty_inspections"]
    keys = [(row["pillar_index"], row["site_index"]) for row in inspections]
    assert len(keys) == len(set(keys))
    food = summary["food_initial_position"]
    with Arena(SimConfig(**summary["config"])) as arena:
        sensor = FoodSensor((food["x"], food["y"]), _Planner(arena).pillars)
        first_step = summary["food_detected_step"]
        for record in episode["trajectory"][: first_step + 1]:
            pose = record["true_pose"]
            observation = sensor.observe((pose["x"], pose["y"]))
            assert (observation is not None) == (record["step"] == first_step)
    assert summary["first_detection"]["distance_m"] <= summary["detection_range_m"]


def test_budget_at_exact_pickup_and_delivery_boundaries(episode):
    first_pickup = next(row["step"] for row in episode["trajectory"] if row["phase"] == "pickup")
    before_pickup = run_showcase(capture_frames=False, max_steps=first_pickup - 1)
    assert not before_pickup["summary"]["pickup_completed"]
    assert before_pickup["summary"]["food_state"] == "on_ground"
    assert before_pickup["trajectory"][-1]["food_state"] == "on_ground"
    first_delivery = next(
        row["step"] for row in episode["trajectory"] if row["phase"] == "delivered"
    )
    after_delivery = run_showcase(capture_frames=False, max_steps=first_delivery)
    assert after_delivery["summary"]["delivery_completed"]
    assert after_delivery["summary"]["success"]
    assert (
        after_delivery["summary"]["food_state"]
        == after_delivery["trajectory"][-1]["food_state"]
        == "delivered"
    )
