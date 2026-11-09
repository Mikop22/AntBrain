# Architecture and experiment interfaces

## Simulator

`Arena(SimConfig(...), noise=NoiseConfig(...))` owns one independent PyBullet DIRECT client. Always use a context manager or call `close()`.

```python
from ant_homing.simulation import Arena, Motion, NoiseConfig, Pose, SimConfig

with Arena(SimConfig(seed=3), noise=NoiseConfig(seed=17, distance_std=0.01)) as arena:
    arena.reset(Pose(0, 0, 0))
    observation = arena.step(Motion(forward=0.1, turn=0.2))
    image = observation.rgb
```

`Motion` first turns, then translates along the resulting heading. Units are metres and radians. Positive yaw is counterclockwise about the vertical axis. The origin is the arena centre. Swept collision checks reject the entire translation when the conservative square footprint would intersect a landmark or wall; the requested turn still occurs. There is no partial advance to a contact surface.

The agent has no visible body. Its camera is rendered directly at the agent pose. `agent_radius` sets the conservative planar collision footprint, not a simulated articulated robot.

| `SimConfig` field | Default | Meaning |
| --- | --- | --- |
| `seed` | 0 | Procedural scene seed |
| `arena_size` | 6.0 | Square arena side length in metres; minimum 4 |
| `camera_size` | 32 | Width/height of each of four square RGB views |
| `camera_height` | 0.30 | Camera height above the floor in metres |
| `agent_radius` | 0.12 | Footprint half-width in metres |
| `landmark_count` | 14 | Number of procedural landmarks around the open centre |

The panorama concatenates 90-degree perspective views facing **east, south, west, north**. This is a four-view perspective strip rather than an equirectangular image. It is aligned to the ideal compass, independent of body yaw. `CAMERA_CONVENTION` versions that ordering; dataset and checkpoint loaders reject incompatible conventions.

`Observation` contains only `rgb`, relative-position `odometry`, `compass_heading`, and `collision`. Odometry displacement uses world-aligned axes and starts at `(0, 0)` on reset; its yaw begins at the reset heading. `Arena.true_pose`, `sample_free_pose`, and `render_overhead` are evaluator/data-collector APIs.

## Noise and repeatability

Actual motion is deterministic. Noise corrupts the measured dead-reckoning increments, not the physical motion or compass.

| `NoiseConfig` field | Unit/meaning |
| --- | --- |
| `seed` | Independent nonnegative measurement-noise seed; default 0 |
| `distance_std` | Gaussian standard deviation in metres per accepted nonzero translation |
| `distance_bias` | Fractional distance scale bias; 0.1 means +10% |
| `heading_std` | Gaussian standard deviation in radians per nonzero action |
| `heading_bias` | Additive heading bias in radians per nonzero action |

`reset()` restarts the noise stream. Identical scene, noise seed, and actions replay deterministically in the validated runtime. Change the noise seed for independent noisy repetitions in a fixed scene. Different methods consume different numbers of noise samples because their actions differ; paired seeds are not identical perturbations at identical physical times.

Use `--noise-seed 17` with `demo` or `benchmark` to override the JSON noise seed. For custom routes, pass `--route configs/route-example.json`; each entry supplies `forward` and `turn`. A route may fail by collision, and that failure is retained in the results.

## Data and learning

`generate_dataset()` saves paired reference/query captures and metadata. Each scene's gallery locations are at least 0.25 m apart. Query offsets are sampled within 0.06 m per axis by default, subject to collision-free placement. Paired sessions use independent sensor exposure/noise perturbations. Every query has a gallery counterpart, so this is closed-set retrieval, not unfamiliar-place rejection.

`images.npz` contains uint8 `gallery` and `query` images, exact pose labels, scene/split/location IDs, and exposure gains. `metadata.json` records the image convention, scene configurations, conditions, and NPZ SHA-256. Loading validates hashes, shapes, finite labels, and declared split membership.

The encoder has three convolutional layers, spatial pooling, and a 64-dimensional output. Symmetric InfoNCE uses matched captures as positives and other locations **within the same scene** as negatives. Training does not use test images. Checkpoints are selected by validation top-1 retrieval accuracy, with the earliest epoch winning ties. Checkpoints record training/validation scene IDs, data hash, image convention, and seed.

Evaluation searches only the gallery of the query's scene. It reports top-1 location identity accuracy, fraction retrieved within 0.25 m, location error, and sample/scene counts. This task assumes gallery scene membership is known. It does not measure global place recognition across unrelated environments.

## Navigation and artifacts

`run_episode(config, method, outbound=..., noise=...)` runs a prescribed outbound route followed by homing. Controllers receive the restricted observation/action interface. The recorder alone accesses true pose for metrics. The action budget includes outbound travel, turns, visual probes, and reversals.

- `odometry`: drive toward the measured origin.
- `raw` / `normalized` / `cnn`: physically probe four cardinal directions and descend the selected image-distance score.
- `hybrid`: approach the origin using odometry, then apply raw-pixel visual correction.

Probes assume ideal actuation can return the agent to its previous position. This assumption must be revisited before introducing actuation noise. Visual match thresholds use the selected scorer's units; default zero requires exact match and otherwise the controller stops on insufficient improvement or the action limit. Threshold selection is a research parameter, not an evaluator-goal signal.

`goal_within_radius` records final geometry. `success` also requires a completed outbound route, a started homing phase, and a nonfailure controller stop. Timeout/collision failures remain failures even if the endpoint happens to be near home. A local-minimum stop within tolerance can count as success without the controller knowing true coordinates.

Episode output contains:

- `summary.json`: configuration, noise seed, full outbound route, completion flags, stop reason, error, path/action costs, snapshot bytes, and CNN checkpoint hash/size when used.
- `trajectory.json`: every action, phase, measured odometry, evaluator pose, and collision flag.
- `provenance.json`: runtime and dependency versions.
- `home.png`, `final.png`, `trajectory.png`, optional `observations.gif`, and local `index.html`.

Snapshot memory counts stored RGB only. CNN weights and controller state are excluded; checkpoint file size is reported separately. No memory-efficiency claim should combine these quantities without specifying what is counted.

## Extension points

Start with different scene seeds, routes, sensor conditions, and independent noise repetitions. Then add unfamiliar-place rejection, varied camera headings, multiple snapshots, learned visual correction inside the hybrid controller, or adaptive memory. Keep the information boundary and baseline/controller comparability intact as features are added.
