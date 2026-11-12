"""Verify frame synchronization and evaluator-camera projection."""

import numpy as np
import pytest
from PIL import Image

from ant_homing.navigation import run_episode
from ant_homing.replay import _display_indices, project_overhead, write_replay
from ant_homing.simulation import SimConfig


def test_overhead_projection_matches_camera_axes():
    assert project_overhead(0, 0, 6, 520) == (260, 260)
    assert project_overhead(1, 0, 6, 520)[0] > 260
    assert project_overhead(0, 1, 6, 520)[1] < 260
    edge = np.tan(np.radians(32)) * 6 * 0.95
    assert project_overhead(edge, 0, 6, 520)[0] == pytest.approx(520)


def test_replay_contains_every_action_and_standalone_controls(tmp_path):
    result = run_episode(SimConfig(), max_steps=3, method="normalized", capture_frames=True)
    write_replay(result, tmp_path)
    with Image.open(tmp_path / "replay.gif") as animation:
        assert animation.n_frames == len(result["trajectory"])
        assert animation.size == (1120, 820)
    html = (tmp_path / "replay.html").read_text()
    assert 'id="seek"' in html and 'id="play"' in html
    assert "FRAME_DATA" not in html and "data:image/png;base64," in html
    assert "Recorded actions, not a live controller" in html
    result["frames"].pop()
    with pytest.raises(ValueError, match="one camera frame"):
        write_replay(result, tmp_path)


def test_subsampling_retains_short_pickup_and_delivery_events():
    records = [{"phase": "outbound", "carrying": False} for _ in range(300)]
    records[17] = {"phase": "pickup", "carrying": True}
    for index in range(18, 299):
        records[index] = {"phase": "returning", "carrying": True}
    records[-1] = {"phase": "delivered", "carrying": False}
    displayed = _display_indices(records, 150)
    assert {0, 17, 18, 299}.issubset(displayed)
    assert len(displayed) < len(records)


def test_ideal_replay_uses_dynamic_overview_and_explicit_showcase_labels(tmp_path):
    result = run_episode(SimConfig(), max_steps=3, method="normalized", capture_frames=True)
    result["summary"]["demonstration_mode"] = "ideal_foraging"
    phases = ["outbound", "pickup", "returning", "delivered"]
    result["overview_frames"] = []
    for index, record in enumerate(result["trajectory"]):
        record.update(
            phase=phases[index], carrying=index in {1, 2}, food_position={"x": 0.5, "y": 0.3}
        )
        result["overview_frames"].append(
            np.full((520, 520, 3), [10 + index, 20, 30], dtype=np.uint8)
        )
    write_replay(result, tmp_path)
    html = (tmp_path / "replay.html").read_text()
    assert "Food-unknown pillar search" in html and "not a trained neural policy" in html
    assert 'value="100" selected' in html
    with Image.open(tmp_path / "replay.gif") as animation:
        assert animation.size == (1120, 656)
        assert animation.info["duration"] == 100
        top = np.asarray(animation.convert("RGB"))[:16]
        assert np.all(top == top[0, 0])
        assert animation.convert("RGB").getpixel((25, 156)) == (10, 20, 30)
        animation.seek(1)
        assert animation.info["duration"] == 1300
        animation.seek(3)
        assert animation.info["duration"] == 2200
    result["overview_frames"].pop()
    with pytest.raises(ValueError, match="one overview frame"):
        write_replay(result, tmp_path)
