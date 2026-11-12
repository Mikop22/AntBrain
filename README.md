# AntBrain

A reproducible 3D simulation sandbox for learned visual memory and local homing.

![Ant foraging around pillars, finding food and returning home](docs/assets/homing-demo.gif)

[Arena overview](docs/assets/arena.png) · [Trajectory example](docs/assets/trajectory.png) · [Architecture](docs/ARCHITECTURE.md) · [Research roadmap](docs/RESEARCH.md) · [Level 1 visual models](docs/level1/README.md) · [Foraging showcase](docs/SHOWCASE.md)

## Quick start

Requires Linux x86-64, Python 3.11, and [uv](https://docs.astral.sh/uv/).

```bash
bash scripts/bootstrap.sh
.venv/bin/python -m pytest
bash scripts/smoke.sh artifacts/smoke
```

## Final Level 1 view models

Saved checkpoints and calibrated familiarity thresholds are tracked in [`models/level1/`](models/level1/manifest.json):

| Model | Module | Descriptor | Recall@1 | Familiarity AUROC | Unfamiliar FPR |
| --- | --- | --- | ---: | ---: | ---: |
| **InfoNCE CNN** (seeds 0–2) | `ant_homing.level1.infonce` | 64D (`256 B`) | 96.21% | **0.989** | **6.25%** |
| **Normalized Pixels** | `ant_homing.level1.pixels` | 12,288D (`49 KiB`) | **98.87%** | 0.866 | 48.61% |

## Repository layout

- `src/ant_homing/` — 3D PyBullet arena (`simulation.py`), homing controllers (`navigation.py`), foraging showcase (`showcase.py`, `search.py`), replay renderer (`replay.py`), and Level 1 visual models (`level1/`).
- `models/level1/` — Frozen InfoNCE checkpoints (`infonce-seed-{0,1,2}.pt`) and calibrated threshold manifest (`manifest.json`).
- `configs/` — Arena, route, and odometry-noise configurations.
- `scripts/` — Environment bootstrap (`bootstrap.sh`) and end-to-end smoke check (`smoke.sh`).
- `tests/` — Automated test suite.
