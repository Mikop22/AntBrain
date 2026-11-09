"""Verify frame synchronization and evaluator-camera projection."""

import numpy as np
import pytest
from PIL import Image

from ant_homing.navigation import run_episode
from ant_homing.replay import project_overhead, write_replay
from ant_homing.simulation import SimConfig


def test_overhead_projection_matches_camera_axes():
    assert project_overhead(0, 0, 6, 520) == (260, 260)
    assert project_overhead(1, 0, 6, 520)[0] > 260
    assert project_overhead(0, 1, 6, 520)[1] < 260
    # A 32-degree half-FOV corresponds to the edge of the perspective view.
    edge = np.tan(np.radians(32)) * 6 * 0.95
    assert project_overhead(edge, 0, 6, 520)[0] == pytest.approx(520)


def test_replay_contains_every_action_and_standalone_controls(tmp_path):
    result = run_episode(SimConfig(), max_steps=3, method="raw", capture_frames=True)
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
