"""Annotated episode replays with synchronized evaluator and agent views."""

from __future__ import annotations

import base64
import io
import json
from itertools import pairwise
from math import cos, hypot, radians, sin, tan
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .simulation import Arena, SimConfig


def project_overhead(x: float, y: float, arena_size: float, size: int) -> tuple[float, float]:
    """Project floor coordinates using Arena.render_overhead's perspective camera."""
    scale = size / (2 * tan(radians(32)) * arena_size * 0.95)
    return size / 2 + x * scale, size / 2 - y * scale


def _encode(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def write_replay(result: dict, output: Path) -> None:
    """Create a scrub-able standalone replay and a shareable annotated GIF.

    The overview and distance readouts are evaluator overlays, never observations
    supplied to the controller. Every recorded camera frame is paired with the
    same action record; only very long replays are subsampled for display.
    """
    frames, records, summary = result["frames"], result["trajectory"], result["summary"]
    if not frames or len(frames) != len(records):
        raise ValueError("Replay requires one camera frame per trajectory record")
    config = SimConfig(**summary["config"])
    with Arena(config) as arena:
        background = Image.fromarray(arena.render_overhead(size=520))
    font = ImageFont.load_default(size=17)
    heading = ImageFont.load_default(size=25)
    small = ImageFont.load_default(size=14)
    home = records[0]["true_pose"]
    trail = [record["true_pose"] for record in records]
    stride = max(1, int(np.ceil(len(frames) / 180)))
    indices = sorted(set(range(0, len(frames), stride)) | {len(frames) - 1})
    composites = []
    travelled = 0.0
    distances = [0.0]
    for previous, current in pairwise(trail):
        travelled += hypot(current["x"] - previous["x"], current["y"] - previous["y"])
        distances.append(travelled)
    # Stable bounds across frames prevent apparent motion from auto-scaling.
    xs, ys = [p["x"] for p in trail], [p["y"] for p in trail]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 0.6) + 0.3
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2

    def zoom(p):
        return 828 + (p["x"] - cx) * 195 / span, 652 - (p["y"] - cy) * 195 / span

    for index in indices:
        record, pose = records[index], trail[index]
        action = record["action"]
        phase = "Home view stored"
        if index:
            if record["phase"] == "outbound":
                phase = "Outbound: leaving home"
            elif action and action["forward"] < 0:
                phase = "Homing: reversing a probe"
            else:
                phase = "Homing: probing / advancing"
        if index == len(frames) - 1:
            phase = "Finished: " + summary["stop_reason"].replace("_", " ")
        image = Image.new("RGB", (1120, 820), "#f4f7fb")
        draw = ImageDraw.Draw(image)
        draw.text((24, 18), "AntBrain | Recorded simulation replay", font=heading, fill="#142b49")
        draw.text(
            (24, 58),
            f"{phase}    |    action {record['step']} / {summary['steps']}",
            font=font,
            fill="#1766ad",
        )
        draw.text(
            (24, 91),
            f"Method: {summary['method']}   |   Travel including probes: {distances[index]:.2f} m",
            font=font,
            fill="#34475e",
        )
        image.paste(background, (24, 155))
        draw.text((24, 128), "3D arena - evaluator overview", font=font, fill="#142b49")

        def world_point(p):
            x, y = project_overhead(p["x"], p["y"], config.arena_size, 520)
            return x + 24, y + 155

        points = [world_point(p) for p in trail[: index + 1]]
        if len(points) > 1:
            draw.line(points, fill="#3e9bff", width=3)
        hx, hy = world_point(home)
        px, py = world_point(pose)
        draw.ellipse((hx - 10, hy - 10, hx + 10, hy + 10), outline="#1bd7a0", width=3)
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill="#ffad32", outline="white", width=2)
        draw.line(
            (px, py, px + 20 * cos(pose["yaw"]), py - 20 * sin(pose["yaw"])),
            fill="#ffad32",
            width=3,
        )
        draw.text(
            (24, 692), "Orange: agent + heading   Green ring: home", font=font, fill="#34475e"
        )
        draw.text(
            (24, 721),
            "Blue: travelled path, including exploratory probes",
            font=font,
            fill="#34475e",
        )
        draw.text(
            (24, 754),
            "Overview/zoom use true positions for display only.",
            font=small,
            fill="#596979",
        )
        draw.text(
            (24, 777),
            "Camera strip stays compass-aligned as the agent turns.",
            font=small,
            fill="#596979",
        )
        for title, y, rgb in (
            ("Current agent view", 128, frames[index]),
            ("Stored home view", 315, result["home_rgb"]),
        ):
            draw.text((580, y), title, font=font, fill="#142b49")
            for panel, label in enumerate(("East", "South", "West", "North")):
                draw.text((580 + panel * 124, y + 26), label, font=small, fill="#596979")
            image.paste(
                Image.fromarray(rgb).resize((496, 124), Image.Resampling.NEAREST), (580, y + 48)
            )
        draw.text((580, 510), "Route close-up (metres relative to home)", font=font, fill="#142b49")
        draw.rounded_rectangle((580, 540, 1076, 770), radius=10, fill="white", outline="#cdd8e6")
        zoom_points = [zoom(p) for p in trail[: index + 1]]
        if len(zoom_points) > 1:
            draw.line(zoom_points, fill="#1766ad", width=3)
        for p, color in ((home, "#15916c"), (pose, "#e58b05")):
            x, y = zoom(p)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        error = hypot(pose["x"] - home["x"], pose["y"] - home["y"])
        draw.text((592, 550), f"Distance to home: {error:.2f} m", font=small, fill="#34475e")
        draw.text(
            (592, 745),
            "N: up / E: right | Final success tolerance: " + f"{summary['success_radius']:.2f} m",
            font=small,
            fill="#596979",
        )
        composites.append(image)
    composites[0].save(output / "replay.png")
    composites[0].save(
        output / "replay.gif",
        save_all=True,
        append_images=composites[1:],
        duration=[350] * (len(composites) - 1) + [1800],
        loop=0,
    )
    data = json.dumps([_encode(frame) for frame in composites])
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>AntBrain replay</title>
<style>body{font:16px system-ui;background:#f4f7fb;max-width:1120px;margin:24px auto;padding:0 12px}img{width:100%}nav{display:flex;gap:14px;align-items:center;flex-wrap:wrap}input{flex:1;min-width:140px}button,select{font:inherit;padding:8px}p{color:#42566e}</style>
<nav><button id="play">Play</button><button id="back">Previous</button><button id="next">Next</button><input id="seek" aria-label="Replay position" type="range" min="0" value="0"><select id="speed" aria-label="Playback speed"><option value="700">Slow</option><option value="350" selected>Normal</option><option value="150">Fast</option></select></nav>
<img id="frame" alt="Synchronized simulation and agent views">
<p>Recorded actions, not a live controller. Pause or drag the slider to inspect a probe. The overhead views are evaluator displays; the agent receives camera images and permitted sensors only.</p>
<script>const frames=FRAME_DATA;let i=0,timer=null;const image=document.getElementById('frame'),seek=document.getElementById('seek'),play=document.getElementById('play');seek.max=frames.length-1;
function show(n){i=Math.max(0,Math.min(frames.length-1,n));image.src='data:image/png;base64,'+frames[i];seek.value=i}function pause(){clearInterval(timer);timer=null;play.textContent='Play'}
play.onclick=()=>{if(timer){pause();return}if(i===frames.length-1)show(0);play.textContent='Pause';timer=setInterval(()=>{if(i===frames.length-1){pause();return}show(i+1)},Number(document.getElementById('speed').value))};
seek.oninput=()=>{pause();show(Number(seek.value))};document.getElementById('back').onclick=()=>{pause();show(i-1)};document.getElementById('next').onclick=()=>{pause();show(i+1)};document.getElementById('speed').onchange=pause;show(0);</script></html>"""
    (output / "replay.html").write_text(page.replace("FRAME_DATA", data), encoding="utf-8")
