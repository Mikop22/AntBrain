# Validation record

Validated on 9–10 November 2026 in the cloud workspace, using Python 3.11.17, PyBullet 3.2.7, PyTorch 2.6.0+cpu, and NumPy 2.4.6. This record establishes software functionality, not experimental superiority or biological fidelity.

## Automated checks

| Check | Result |
| --- | --- |
| Full pytest suite | **114 passed**, no failures or skips |
| Level 1 tests | 52 passed: data contracts, pixel matching, encoder interfaces, loss/miner scalar references and finite gradients, spatial batch guards, retrieval/familiarity metrics, review gating, and frozen provenance |
| Simulation tests | 11 passed: actual rendering, deterministic views, compass ordering, collision sweeps, parallel clients, independent noise seeds, and lifecycle |
| Learning tests | 12 passed: actual optimization/weight updates, checkpoint reload, deterministic training, held-out scenes, provenance validation, and stale camera-convention rejection |
| Navigation tests | 13 passed: sensor-only controllers, physical probe costs, bounded trials, blocked-route failure, route recording, and checkpoint hashing |
| CLI/artifact tests | 10 passed: actual render/demo outputs, safe output handling, malformed configuration diagnostics, custom route/noise seed, and displaced-home odometry plotting |
| Replay tests | 4 passed: perspective projection, action/frame synchronization, lifecycle-event retention, and ideal showcase playback |
| Showcase tests | 9 passed: food-unknown search, cargo lifecycle/clearance, bounded movement, synchronized renders, hidden-food geometry, pre-detection invariance, inspection memory, and exact-budget state handling |
| Search tests | 3 passed: range/occlusion sensing, visited-site memory, and invalid configuration |
| Ruff check and format check | Passed for all 31 Python source/test/report files |
| Bootstrap repeatability | Rerun successfully from `/tmp`, using the script's own project-directory resolution |
| Wheel distribution | Built successfully; installed separately and rendered a scene through the installed wheel |

After migration into the AntBrain repository, a fresh repository-local virtual environment was installed with the frozen lockfile. Imports were verified against the new checkout, all 45 tests and Ruff checks passed again, and the complete smoke workflow regenerated its outputs successfully from the repository root.

The annotated-replay update adds synchronized evaluator/camera views and standalone play/pause, step, speed, and scrub controls. The updated suite has 47 passing tests. Chromium/Playwright verified HTML initialization, action stepping, scrubbing, and play/pause with no JavaScript errors; a frame was visually inspected for alignment and readable labels. The replay is recorded playback, not a live simulator GUI.

The ideal-foraging update adds a separate `showcase` command. The full suite now has 55 passing tests. Its actual rendered episode travelled 7.713 m outbound, picked up food, and returned 1.978 m directly home with zero collisions. Pickup, carrying, and delivery frames and browser controls were checked. An independent reviewer also exercised 15 episodes across arena sizes 4/6/8 m and five scene seeds; all completed without collision. This is an ideal engine demonstration, not evidence that a learned navigation policy has achieved these results.

The subsequent pillar scenario replaces that simple route: the default run travels **19.143 m outbound** to food behind an actual pillar and **3.958 m home** around the obstruction, with zero collisions. Tests check the nest-to-food sightline against the pillar and maintain clearance for the carried object. The latest full suite has **56 passing tests**. The compact 1120 × 656 GIF omits the title/status banner and explanatory footer; camera/scene alignment and browser pickup/delivery scrubbing, stepping, and play/pause were verified. Metadata and planning assumptions remain in the HTML/JSON and documentation.

The current food-unknown search replaces target-directed outbound planning. With scene/search/food seeds all 0, the ant completes 36 empty inspection sites across five empty pillars before detecting food at pillar 12, then picks it up and delivers it without collision. Outbound travel is 21.989 m and return travel is 3.772 m. Food seed 1 produces a different target and later discovery under the same search policy. An independent reviewer confirmed identical actions, poses, odometry, and selected inspection pillars for all 662 records before the earlier first detection; range and occlusion checks passed independently. Exact action-budget boundaries preserve the recorded pickup/delivery state. The latest full suite has **62 passing tests**. Browser search/detection/delivery scrubbing and playback were verified with no JavaScript errors; the compact layout remains unchanged.

The current preview is reproducible with `showcase --food-seed 0 --search-seed 0`. Omitting `--food-seed` generates and records a fresh seed. A one-action CLI budget was also checked: it returned a recorded `step_limit` failure with food still on the ground and no false pickup or delivery.

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

## Level 1 controlled pilot

Three independent builders and three independent reviewers prepared a scene-disjoint visual-memory comparison. All pre-training approvals and source/data hashes were frozen before optimization. Two neural approaches each completed 20 epochs at seeds 0, 1, and 2, with identical per-seed batches, update counts, and image exposure. Validation selected checkpoints and calibrated thresholds; all six checkpoints were frozen before held-out evaluation.

On 12 held-out scenes, normalized pixels achieved 98.87% spatial Recall@1, InfoNCE 96.21%, and the ConvAP adaptation 94.84%. InfoNCE had the strongest familiarity result: 6.25% unfamiliar false acceptance at 95.37% familiar acceptance, versus 48.61% at 94.70% for normalized pixels. A subsequent independent read-only audit verified chronology, frozen artifacts, saved-row metrics, aggregation, and paired scene confidence intervals. See the [assessment](level1/ASSESSMENT.md) and [complete results](level1/results/RESULTS.md).

## Remaining research work

The visual-memory pilot now includes multiple training seeds, synthetic sensor conditions, and unfamiliar-place rejection. Generalization to different scene families, heading uncertainty, physical lighting changes, broader local-homing trials, and a stronger compass-aided odometry baseline remain research tasks. Adaptive snapshot allocation and multi-snapshot route navigation are not implemented. No live simulator GUI, hardware transfer, or new-task snapshot restoration is claimed by this validation.
