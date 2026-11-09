"""Exercise installed entry points and artifact generation, not only argument parsing."""

import json

import numpy as np
import pytest

from ant_homing.artifacts import new_output, trajectory_image
from ant_homing.cli import main


def test_render_cli_produces_observations_and_provenance(tmp_path):
    output = tmp_path / "arena"
    assert main(["render", "--output", str(output), "--seed", "2"]) == 0
    assert (output / "arena.png").stat().st_size > 100
    assert (output / "panorama.png").stat().st_size > 100
    assert json.loads((output / "config.json").read_text())["seed"] == 2
    assert json.loads((output / "provenance.json").read_text())["packages"]["pybullet"]
    assert main(["render", "--output", str(output)]) == 2


def test_demo_cli_saves_complete_episode(tmp_path):
    output = tmp_path / "demo"
    assert main(["demo", "--method", "odometry", "--output", str(output)]) == 0
    summary = json.loads((output / "summary.json").read_text())
    trajectory = json.loads((output / "trajectory.json").read_text())
    assert summary["success"]
    assert summary["endpoint_error"] < 0.05
    assert summary["steps"] == len(trajectory) - 1
    assert (output / "observations.gif").stat().st_size > 100
    assert "trajectory.json" in (output / "index.html").read_text()


def test_output_preserves_existing_files(tmp_path):
    (tmp_path / "important.txt").write_text("keep")
    with pytest.raises(ValueError, match="not empty"):
        new_output(tmp_path)
    assert (tmp_path / "important.txt").read_text() == "keep"


def test_trajectory_image_handles_an_empty_trial():
    image = trajectory_image([], 6.0)
    assert np.asarray(image).shape == (712, 640, 3)


@pytest.mark.parametrize("settings", [{"unknown": 1}, [], None, {"arena_size": "bad"}])
def test_invalid_config_is_a_concise_error(tmp_path, settings, capsys):
    config = tmp_path / "bad.json"
    config.write_text(json.dumps(settings))
    assert main(["render", "--config", str(config), "--output", str(tmp_path / "out")]) == 2
    assert "Error:" in capsys.readouterr().err


def test_custom_route_and_independent_noise_seed_are_recorded(tmp_path):
    route = tmp_path / "route.json"
    route.write_text('[{"forward": 0.31, "turn": 0.2}]')
    output = tmp_path / "custom"
    assert (
        main(
            [
                "demo",
                "--method",
                "odometry",
                "--noise-seed",
                "23",
                "--route",
                str(route),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["noise"]["seed"] == 23
    assert summary["outbound_route"] == [{"forward": 0.31, "turn": 0.2}]
