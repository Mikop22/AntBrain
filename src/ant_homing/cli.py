"""Command-line entry points for rendering, foraging showcase, and navigation experiments."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from PIL import Image

from .artifacts import new_output, provenance, write_episode, write_json


def _settings(path: str, name: str) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")  # noqa: TRY004 -- external input value
    return value


def _config(args):
    from .simulation import SimConfig

    values = _settings(args.config, "Simulation configuration") if args.config else {}
    if args.seed is not None:
        values["seed"] = args.seed
    try:
        return SimConfig(**values)
    except TypeError as error:
        raise ValueError(f"Invalid simulation configuration: {error}") from error


def _noise(args):
    from .simulation import NoiseConfig

    values = _settings(args.noise, "Noise configuration") if args.noise else {}
    if args.noise_seed is not None:
        values["seed"] = args.noise_seed
    try:
        return NoiseConfig(**values)
    except TypeError as error:
        raise ValueError(f"Invalid noise configuration: {error}") from error


def _episode(args, *, method=None, config=None):
    from .navigation import run_episode
    from .simulation import Motion

    route = None
    if args.route:
        motions = json.loads(Path(args.route).read_text())
        if not isinstance(motions, list) or not motions:
            raise ValueError("Route must be a nonempty JSON array of motion objects")
        try:
            route = [Motion(**motion) for motion in motions]
        except TypeError as error:
            raise ValueError(f"Invalid route motion: {error}") from error

    return run_episode(
        config if config is not None else _config(args),
        method=method or args.method,
        noise=_noise(args),
        max_steps=args.max_steps,
        step_distance=args.step_distance,
        success_radius=args.success_radius,
        checkpoint=args.checkpoint,
        capture_frames=args.command == "demo",
        outbound=route,
    )


def _render(args):
    from .simulation import Arena

    output = new_output(args.output)
    config = _config(args)
    with Arena(config) as arena:
        Image.fromarray(arena.observe().rgb).resize((1024, 256)).save(output / "panorama.png")
        Image.fromarray(arena.render_overhead()).save(output / "arena.png")
    write_json(output / "config.json", asdict(config))
    write_json(output / "provenance.json", provenance())
    return {"output": str(output), "config": asdict(config)}


def _demo(args):
    from .replay import write_replay

    output = new_output(args.output)
    result = _episode(args)
    write_episode(result, output)
    write_replay(result, output)
    return {"output": str(output), **result["summary"]}


def _showcase(args):
    from .replay import write_replay
    from .showcase import run_showcase

    output = new_output(args.output)
    food_seed = (
        args.food_seed
        if args.food_seed is not None
        else int(np.random.SeedSequence().generate_state(1)[0])
    )
    result = run_showcase(
        _config(args),
        capture_frames=True,
        food_seed=food_seed,
        search_seed=args.search_seed,
        max_steps=args.max_steps,
    )
    write_episode(result, output)
    write_replay(result, output)
    return {"output": str(output), **result["summary"]}


def _benchmark(args):
    output = new_output(args.output)
    config = _config(args)
    seeds = [int(x) for x in args.seeds.split(",")]
    methods = args.methods.split(",")
    if len(set(seeds)) != len(seeds) or len(set(methods)) != len(methods):
        raise ValueError("Benchmark seeds and methods must not contain duplicates")
    summaries = []
    for seed in seeds:
        for method in methods:
            result = _episode(args, method=method, config=replace(config, seed=seed))
            run_output = output / f"{method}-seed-{seed}"
            run_output.mkdir()
            write_episode(result, run_output)
            summaries.append(result["summary"])
    aggregate = {}
    for method in methods:
        rows = [row for row in summaries if row["method"] == method]
        errors = [row["endpoint_error"] for row in rows]
        aggregate[method] = {
            "episodes": len(rows),
            "successes": sum(row["success"] for row in rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "mean_endpoint_error_m": float(np.mean(errors)),
            "median_endpoint_error_m": float(np.median(errors)),
            "mean_path_length_m": float(np.mean([row["path_length"] for row in rows])),
        }
    report = {
        "description": "Paired environment seeds, one prescribed route; pilot metrics only.",
        "seeds": seeds,
        "methods": methods,
        "aggregate": aggregate,
        "episodes": summaries,
        "provenance": provenance(),
    }
    write_json(output / "benchmark.json", report)
    return {"output": str(output), "aggregate": aggregate}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="3D visual-memory navigation research sandbox")
    commands = root.add_subparsers(dest="command", required=True)
    showcase = commands.add_parser(
        "showcase", help="Ideal food pickup and direct homing demonstration"
    )
    showcase.set_defaults(function=_showcase)
    showcase.add_argument("--config", help="JSON object with SimConfig fields")
    showcase.add_argument("--seed", type=int, default=None)
    showcase.add_argument("--food-seed", type=int, help="Reproduce food placement; omitted=random")
    showcase.add_argument(
        "--search-seed", type=int, default=0, help="Search order independent of food"
    )
    showcase.add_argument("--max-steps", type=int, default=2000, help="Total action budget")
    showcase.add_argument("--output", required=True, help="New or empty artifact directory")
    for name, function in (("render", _render), ("demo", _demo), ("benchmark", _benchmark)):
        command = commands.add_parser(name)
        command.set_defaults(function=function)
        command.add_argument("--config", help="JSON object with SimConfig fields")
        command.add_argument("--seed", type=int, default=None)
        command.add_argument("--output", required=True, help="New or empty artifact directory")
        if name == "render":
            continue
        command.add_argument("--noise", help="JSON object with NoiseConfig fields")
        command.add_argument(
            "--noise-seed", type=int, help="Noise replicate independent of scene seed"
        )
        command.add_argument("--route", help="JSON array of turn-then-translate motion objects")
        command.add_argument("--max-steps", type=int, default=160)
        command.add_argument("--step-distance", type=float, default=0.15)
        command.add_argument("--success-radius", type=float, default=0.25)
        command.add_argument(
            "--checkpoint",
            help="InfoNCE checkpoint path (defaults to models/level1/infonce-seed-0.pt)",
        )
        if name == "demo":
            command.add_argument(
                "--method",
                choices=["odometry", "normalized", "infonce", "cnn", "hybrid"],
                default="normalized",
            )
        else:
            command.add_argument("--seeds", default="0,1,2")
            command.add_argument("--methods", default="odometry,normalized,infonce,hybrid")
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        report = arguments.function(arguments)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0
