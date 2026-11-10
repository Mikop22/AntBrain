"""Portable visualizations and provenance for simulation runs."""

from __future__ import annotations

import html
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def provenance() -> dict:
    """Record runtime versions without reading environment variables or secrets."""
    versions = {}
    for name in ("ant-homing-sim", "numpy", "pybullet", "pillow", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": versions}


def new_output(path: str | Path) -> Path:
    """Reject populated run directories instead of silently overwriting evidence."""
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}. Choose a new run directory.")
    return output


def trajectory_image(records: list[dict], arena_size: float, size: int = 640) -> Image.Image:
    """Plot evaluator truth and measured odometry in the same metric coordinate frame."""
    image = Image.new("RGB", (size, size + 72), "#f4f6f8")
    draw = ImageDraw.Draw(image)
    margin = 44
    scale = (size - 2 * margin) / arena_size

    def project(pose: dict) -> tuple[float, float]:
        return (size / 2 + pose["x"] * scale, size / 2 - pose["y"] * scale)

    for tick in np.linspace(-arena_size / 2, arena_size / 2, 7):
        x, _ = project({"x": float(tick), "y": 0})
        _, y = project({"x": 0, "y": float(tick)})
        draw.line((x, margin, x, size - margin), fill="#dfe4eb")
        draw.line((margin, y, size - margin, y), fill="#dfe4eb")
    draw.rectangle((margin, margin, size - margin, size - margin), outline="#9aa8b8", width=2)
    for key, color in (("odometry", "#d58a16"), ("true_pose", "#1766ad")):
        if key == "odometry" and records:
            # Dead reckoning is relative to reset; evaluator poses are world coordinates.
            origin = records[0]
            points = [
                project(
                    {
                        "x": record[key]["x"] - origin[key]["x"] + origin["true_pose"]["x"],
                        "y": record[key]["y"] - origin[key]["y"] + origin["true_pose"]["y"],
                    }
                )
                for record in records
            ]
        else:
            points = [project(record[key]) for record in records]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
    if records:
        for record, color in ((records[0], "#15916c"), (records[-1], "#c63849")):
            x, y = project(record["true_pose"])
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
    draw.text(
        (margin, size + 4), "Blue: true trajectory   Amber: measured odometry", fill="#24344a"
    )
    draw.text(
        (margin, size + 24), "Green: home   Red: final position   Units: metres", fill="#24344a"
    )
    draw.text(
        (margin, size + 44), "Evaluator visualization; never a navigation input", fill="#596979"
    )
    return image


def write_episode(result: dict, output: Path) -> None:
    """Write machine-readable results and a self-contained local report."""
    summary = result["summary"]
    ideal = summary.get("demonstration_mode") == "ideal_foraging"
    write_json(output / "summary.json", summary)
    write_json(output / "trajectory.json", result["trajectory"])
    write_json(output / "provenance.json", provenance())
    for key, filename in (("home_rgb", "home.png"), ("final_rgb", "final.png")):
        Image.fromarray(result[key]).resize((768, 192)).save(output / filename)
    arena_size = summary["config"].get("arena_size", 6.0)
    trajectory_image(result["trajectory"], arena_size).save(output / "trajectory.png")
    frames = result.get("frames", [])
    if frames:
        # Cap replay size independently of the complete trajectory/action log.
        stride = max(1, int(np.ceil(len(frames) / 180)))
        replay = [Image.fromarray(frame).resize((512, 128)) for frame in frames[::stride]]
        replay[0].save(
            output / "observations.gif",
            save_all=True,
            append_images=replay[1:],
            duration=100,
            loop=0,
        )
    animation = '<img src="observations.gif" alt="Agent camera replay">' if frames else ""
    replay_link = (
        '<p><a href="replay.html">Open the annotated replay with pause and step controls</a></p>'
        if frames
        else ""
    )
    text = html.escape(json.dumps(summary, indent=2))
    description = (
        "Ideal engine showcase: scripted foraging, hidden food pickup and obstacle-aware scene-geometry planning. "
        "This is a demonstration of the intended behaviour, not a trained navigation result."
        if ideal
        else "This is a simulation run, not evidence of research superiority. True state is used only for evaluation."
    )
    report = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Ant Homing — Episode</title>
<style>body{{font:16px system-ui;max-width:1040px;margin:40px auto;padding:0 20px;color:#203047;background:#f7f9fc}}
img{{max-width:100%;border:1px solid #d6dfea;border-radius:8px}}pre{{overflow:auto;padding:20px;background:#eaf0f7}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}@media(max-width:650px){{.grid{{grid-template-columns:1fr}}}}</style>
<h1>Ant Homing Simulation</h1><p>Method: <b>{html.escape(summary["method"])}</b> ·
Endpoint error: <b>{summary["endpoint_error"]:.3f} m</b> · Stop: {html.escape(summary["stop_reason"])}</p>
<p>{description}</p>
{replay_link}
<div class="grid"><section><h2>Trajectory</h2><img src="trajectory.png" alt="Trajectory"></section>
<section><h2>Stored home view</h2><img src="home.png" alt="Stored panorama"><h2>Final view</h2>
<img src="final.png" alt="Final panorama"><h2>Agent observations</h2>{animation}</section></div>
<h2>Configuration and metrics</h2><pre>{text}</pre>
<p>Full records: <a href="summary.json">summary</a>, <a href="trajectory.json">trajectory</a>,
<a href="provenance.json">runtime versions</a>.</p></html>"""
    (output / "index.html").write_text(report, encoding="utf-8")
