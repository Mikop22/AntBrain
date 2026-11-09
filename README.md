# AntBrain: Ant Homing Simulation

A Python research sandbox for learned visual memory and local homing. An agent moves on a flat floor in a procedurally generated 3D arena and observes low-resolution RGB panoramas. Rendering and neural-network training run on CPU; no GPU or external scene assets are required.

![Visual homing camera replay](docs/assets/homing-demo.gif)

[Arena overview](docs/assets/arena.png) · [Evaluator trajectory](docs/assets/trajectory.png)

## Quick start

Requires Linux x86-64 (validated platform), [uv](https://docs.astral.sh/uv/getting-started/installation/), and network access for the first installation. Python 3.11 and dependencies are pinned through `.python-version` and `uv.lock`.

```bash
git clone https://github.com/Mikop22/AntBrain.git
cd AntBrain
bash scripts/bootstrap.sh
.venv/bin/python -m pytest
bash scripts/smoke.sh artifacts/first-run
```

The smoke workflow renders an arena, collects data, trains a small CNN, evaluates held-out retrieval, runs visual and neural homing episodes, and benchmarks classical controllers. Choose a new output path on each run. Short smoke runs validate the software, not a research hypothesis.

For an existing cloud checkout, start in `/workspace/AntBrain` and run the setup and validation commands there.

Open `artifacts/first-run/visual-demo/index.html` locally to inspect the trajectory, stored/current panoramas, animation, and metrics. `artifacts/first-run/arena/arena.png` shows the 3D environment. Full outputs are JSON/NPZ/PyTorch files suitable for analysis.

For the clearest demonstration, open `visual-demo/replay.html`: a standalone recorded replay with play/pause, individual action stepping, and a scrubber. It combines the 3D arena, agent/heading marker, travelled path, route close-up, and labelled current/home camera views. `replay.gif` is the shareable animated version. Overhead positions are evaluation overlays, not controller inputs.

## Commands

The installed `ant-homing` console command and `python -m ant_homing` are equivalent.

```bash
# Inspect a deterministic scene and its agent-level panorama.
.venv/bin/python -m ant_homing render --seed 5 --output artifacts/scene-5

# Compare image-based local homing with measured-odometry return.
.venv/bin/python -m ant_homing demo --method raw --output artifacts/raw-1
.venv/bin/python -m ant_homing demo --method odometry \
  --noise configs/noise-mild.json --output artifacts/odom-1

# Generate independent train/validation/test scenes and capture sessions.
.venv/bin/python -m ant_homing collect --locations 24 --seed 0 --output artifacts/data
.venv/bin/python -m ant_homing train --data artifacts/data \
  --epochs 8 --seed 0 --output artifacts/model
.venv/bin/python -m ant_homing evaluate --data artifacts/data \
  --checkpoint artifacts/model/model.pt --output artifacts/retrieval.json

# Test the learned image score in the same physical-probing controller.
.venv/bin/python -m ant_homing demo --method cnn \
  --checkpoint artifacts/model/model.pt --output artifacts/cnn-1

# Paired pilot episodes across scene seeds; results include unsuccessful runs.
.venv/bin/python -m ant_homing benchmark --seeds 0,1,2 \
  --methods odometry,raw,normalized,hybrid \
  --noise configs/noise-mild.json --output artifacts/pilot
```

Use `--help` on any subcommand. `render`, `demo`, and `benchmark` accept `--config configs/arena.json`. `collect --scenes configs/scenes-smoke.json` selects an explicit split-to-scene-seed mapping. The Python APIs allow custom outbound motion sequences and dataset parameters.

For a non-grid route, add `--route configs/route-example.json` to `demo` or `benchmark`. Use `--noise-seed 17` for an independent noise repetition without changing scene geometry. The default outbound route is deliberately simple and commensurate with the probe step size; success on it does not establish a broad catchment area.

## Components

| Module | Responsibility |
| --- | --- |
| `simulation.py` | Seeded arena, compass-aligned cameras, kinematic motion, collisions, and noisy odometry |
| `dataset.py` | Gallery/query capture, scene-disjoint splits, hashes, and dataset validation |
| `learning.py` | Compact spatial CNN, within-scene contrastive training, validation checkpoint selection, and retrieval metrics |
| `navigation.py` | Sensor-only controllers and evaluator-owned trajectories/metrics |
| `artifacts.py` | JSON records, runtime provenance, trajectory plots, GIFs, and local HTML reports |
| `replay.py` | Synchronized overhead/camera replay, annotated GIF, and standalone playback controls |
| `cli.py` | Collection, training, evaluation, rendering, and pilot experiment commands |

The CNN learns an image embedding from matched gallery/query views using within-scene negatives. It receives images only. Recognition is evaluated against other stored locations in the same held-out scene. Every query has a gallery counterpart; unfamiliar-place rejection is not implemented.

Navigation methods are `odometry`, `raw`, `normalized`, `cnn`, and `hybrid`. `hybrid` currently combines odometry with **raw-pixel** visual correction. Visual homing performs actual cardinal-direction probes and reverse movements; all actions and travelled distance are counted. It can stop at a misleading local minimum, which is recorded as an unsuccessful endpoint when outside the goal tolerance.

## Research boundaries

- The world is 3D, but motion is planar and kinematic. This is not a wheel-slip or ant-biomechanics model.
- A perfect compass aligns the panorama to world directions. Heading-estimation uncertainty is not yet a visual-learning task.
- Odometry noise changes measured motion estimates; actual actuation remains deterministic. It is not physical wheel slip.
- Dataset appearance changes are post-render exposure/noise perturbations, not realistic relighting.
- Ground-truth coordinates are used for labels, sampling, visualization, and evaluation, never navigation decisions.
- Retrieval accuracy does not demonstrate navigation success. A small installation run does not establish neural superiority.

See [research protocol](docs/RESEARCH.md) for objectives and fair comparisons, [architecture](docs/ARCHITECTURE.md) for interfaces and data formats, [validation](docs/VALIDATION.md) for executed checks, and [sources](docs/SOURCES.md) for tooling references.

## Development

```bash
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/python -m pytest
```

Use the existing checkout. Each cloud task is already isolated; no Git worktree is needed. Generated runs live under ignored `artifacts/`; caches and the virtual environment are local setup files. The Python package remains named `ant-homing-sim`.
