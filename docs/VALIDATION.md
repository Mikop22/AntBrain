# Validation record

Validated on 9 November 2026 in the cloud workspace, using Python 3.11.17, PyBullet 3.2.7, PyTorch 2.6.0+cpu, and NumPy 2.4.6. This record establishes software functionality, not experimental superiority or biological fidelity.

## Automated checks

| Check | Result |
| --- | --- |
| Full pytest suite | **45 passed**, no failures or skips |
| Simulation tests | 11 passed: actual rendering, deterministic views, compass ordering, collision sweeps, parallel clients, independent noise seeds, and lifecycle |
| Learning tests | 12 passed: actual optimization/weight updates, checkpoint reload, deterministic training, held-out scenes, provenance validation, and stale camera-convention rejection |
| Navigation tests | 13 passed: sensor-only controllers, physical probe costs, bounded trials, blocked-route failure, route recording, and checkpoint hashing |
| CLI/artifact tests | 9 passed: actual render/demo outputs, safe output handling, malformed configuration diagnostics, and custom route/noise seed |
| Ruff check and format check | Passed for all 12 Python source/test files |
| Bootstrap repeatability | Rerun successfully from `/tmp`, using the script's own project-directory resolution |
| Wheel distribution | Built successfully; installed separately and rendered a scene through the installed wheel |

After migration into the AntBrain repository, a fresh repository-local virtual environment was installed with the frozen lockfile. Imports were verified against the new checkout, all 45 tests and Ruff checks passed again, and the complete smoke workflow regenerated its outputs successfully from the repository root.

## End-to-end execution

The final `scripts/smoke.sh artifacts/validated` run completed successfully after the panorama convention was finalized. It generated:

- A 3D arena overview and 128 × 32 agent panorama.
- A dataset with 32 gallery/query pairs: two training scenes, one validation scene, and one test scene, each with eight locations.
- Two actual CNN optimization epochs, a validation-selected checkpoint, and held-out retrieval evaluation.
- Raw-pixel and CNN-scored local homing demonstrations, each completing the default route and returning within tolerance.
- Eight baseline benchmark episodes: four methods in two scenes with noisy odometry, including complete path/action accounting.

On this small smoke dataset, all three retrieval methods correctly matched all eight test queries. This is a basic functionality check with a small and easy sample; it provides no evidence that learning improves recognition. The training loss decreased from approximately 1.943 to 1.107, confirming a working optimization path.

An additional non-grid route (`configs/route-example.json`) was evaluated with normalized-image homing. It finished approximately **0.060 m** from home after **73 actions**, with **11.55 m** of total travel including probes. It stopped because no sampled direction improved familiarity, rather than because the controller received true target coordinates.

The default visual demonstration used 65 actions and 10.5 m of travel. Its successful return should not hide this substantial probing cost. The default route is commensurate with the probe lattice; evaluate off-grid starts and multiple routes before drawing catchment-area conclusions.

The original standalone validation generated `artifacts/validated/`. Generated datasets, models, and run logs are intentionally excluded from Git. Reproduce them with `bash scripts/smoke.sh artifacts/validated`, using a new output path if that directory already exists. Selected validated preview images are included in `docs/assets/`.

## Independent review

The simulation, learning, and navigation implementers each spawned a separate reviewer. They exchanged interface and correctness findings directly. A fourth reviewer checked CLI integration, installation, artifacts, and documentation.

Review findings addressed before final validation included: false success for a blocked outbound route; incomplete dataset/checkpoint provenance; coupled scene/noise seeds; discontinuous panorama ordering; stale-camera checkpoint reuse; and unhelpful malformed-configuration errors. Checkpoint hashes and exact outbound routes are now retained with episode results.

## Remaining research work

The platform is ready for controlled pilot studies. Larger datasets, multiple training seeds, harder sensor conditions, unfamiliar-place rejection, broader local-homing trials, and a stronger compass-aided odometry baseline remain research tasks. Adaptive snapshot allocation and multi-snapshot route navigation are not implemented. No GUI, hardware transfer, or new-task snapshot restoration is claimed by this validation.
