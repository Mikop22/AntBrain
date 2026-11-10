# AntBrain: Ant Homing Simulation

A Python research sandbox for learned visual memory and local homing. An agent moves on a flat floor in a procedurally generated 3D arena and observes low-resolution RGB panoramas. Rendering and neural-network training run on CPU; no GPU or external scene assets are required.

![Ant foraging around pillars, finding food and returning home](docs/assets/homing-demo.gif)

The preview shows **food-unknown search**: the ant inspects pillars, remembers empty checks, and discovers food only when nearby and unobstructed before carrying it home. Food placement varies independently of search. The showcase still uses ideal scene geometry for obstacle routing; it is not a neural-network result. The [probing baseline replay](docs/assets/probing-baseline.gif) remains available for comparison.

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


