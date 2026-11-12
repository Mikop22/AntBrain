# Architecture

## Simulator (`ant_homing.simulation`)

`Arena(SimConfig(...), noise=NoiseConfig(...))` manages an isolated headless PyBullet (`DIRECT`) client.

```python
from ant_homing.simulation import Arena, Motion, NoiseConfig, Pose, SimConfig

with Arena(SimConfig(seed=3), noise=NoiseConfig(seed=17, distance_std=0.01)) as arena:
    arena.reset(Pose(0, 0, 0))
    obs = arena.step(Motion(forward=0.1, turn=0.2))
```

- **Kinematics & collisions:** `Motion` turns first, then translates along the new heading. Swept collision checks against the agent's square footprint (`agent_radius = 0.12 m`) reject colliding translations while preserving the turn.
- **Panorama (`CAMERA_CONVENTION`):** Four $90^\circ$ perspective views ($32 \times 32$ each) facing **East, South, West, North**, concatenated into a $32 \times 128 \times 3$ compass-aligned RGB strip independent of body yaw.
- **Information boundary:** `Observation` exposes only `rgb`, relative `odometry`, `compass_heading`, and `collision`. `Arena.true_pose` and `render_overhead()` are restricted to evaluation and replay rendering.
- **Odometry noise:** `NoiseConfig` corrupts dead-reckoning increments (`distance_std`, `distance_bias`, `heading_std`, `heading_bias`) without altering physical movement.

## Visual memory (`ant_homing.level1`)

- `pixels.py`: Z-score normalized RGB descriptors (`encode`) and image distance (`normalized_image_distance`).
- `infonce.py`: 42,728-parameter spatial CNN producing 64D unit descriptors (`build_model`, `encode`, `load_checkpoint`, `load_image_scorer`). Frozen checkpoints are stored in `models/level1/`.
- `data.py` & `evaluation.py`: Scene-disjoint dataset generator and retrieval/familiarity metrics.

## Navigation controllers (`ant_homing.navigation`)

`run_episode(config, method, ...)` executes an outbound route followed by homing within `max_steps`:

- `odometry`: Drive directly toward the dead-reckoned origin.
- `normalized`: Physically probe 4 cardinal directions ($+0.15\text{ m}$ forward and $-0.15\text{ m}$ return) and descend `normalized_image_distance`.
- `infonce` (or `cnn`): Physically probe 4 cardinal directions and descend the InfoNCE descriptor distance (defaults to `models/level1/infonce-seed-0.pt`).
- `hybrid`: Return to within $0.35\text{ m}$ of the dead-reckoned origin via odometry, then switch to visual homing.

## CLI (`ant_homing.cli`)

```bash
.venv/bin/python -m ant_homing render --seed 0 --output artifacts/arena
.venv/bin/python -m ant_homing demo --method infonce --output artifacts/demo
.venv/bin/python -m ant_homing benchmark --seeds 0,1,2 --noise configs/noise-mild.json --output artifacts/bench
.venv/bin/python -m ant_homing showcase --food-seed 0 --search-seed 0 --output artifacts/foraging
```
