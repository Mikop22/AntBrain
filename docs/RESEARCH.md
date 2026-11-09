# Research protocol

## Scope and completion levels

The simulator is infrastructure for an empirical ML project, not a validated biological model.

1. **Visual memory:** train an image embedding and compare location retrieval against raw and normalized pixel distances.
2. **Local homing:** evaluate a controller that samples real visited views around a stored location.
3. **Combined navigation:** use measured odometry to return near home, then apply visual correction.
4. **Future work:** multiple memories, snapshot spacing, adaptive memory allocation, and more realistic sensing.

Complete each level's evaluation before expanding scope. A failed navigation controller does not invalidate a well-designed visual-memory experiment, but retrieval performance must not be reported as homing performance.

## Information boundary

Controllers receive RGB images, measured odometry, collision feedback, and a perfect compass assumption. Ground-truth poses, target distance, and scene geometry are reserved for data generation and evaluation. A top-down rendering is an evaluator visualization, never a policy input. Visual probes are actual movements; their distance and action cost belong in the trial's totals.

The arena uses planar kinematics in a rendered 3D world. It does not model ant biomechanics, tyre slip dynamics, or physically accurate illumination. Brightness changes are sensor-level appearance perturbations. The compass-aligned panorama deliberately removes heading-estimation difficulty from the first study.

## Fair comparisons

- Use the same routes, environments, action budgets, and stopping rules for compared visual representations.
- The current odometry baseline deliberately does not fuse the ideal compass into its accumulated position estimate. Include a compass-aided integration baseline before attributing improvements specifically to learned visual memory rather than correcting an avoidable sensor-estimation error.
- Separate training, validation, and test scenes. Keep paired capture sessions identifiable. Never split adjacent frames randomly across sets.
- Current retrieval is closed-set within each held-out scene, with a gallery counterpart for every query and known scene membership. It does not evaluate unfamiliar-place rejection or cross-scene search.
- Select thresholds and checkpoints using training/validation data, then freeze them before final testing.
- Report which generalization setting is tested. Familiar-location retrieval in an unseen scene is not unrestricted navigation in an unknown world.
- Report endpoint error, success within a predeclared tolerance and action limit, path length including probes, collisions, controller stop reason, and memory bytes.
- Count independent scenes/routes as experimental units. Multiple frames from one route do not establish generalization.
- Use multiple training and environment seeds; report uncertainty. A short installation smoke run provides no evidence that a neural method is superior.
- Vary the independent noise seed to repeat a noisy trial in the same scene. Resetting with the same scene/noise seeds intentionally replays the same realization.

## Suggested first study

**Question:** Does a small learned image representation improve recognition of stored locations under a controlled appearance change?

Compare raw RGB, per-image normalized RGB, and a small trained CNN on the same held-out query/gallery pairs. Include brightness-only and unchanged-appearance controls. Inspect examples of mistaken retrievals. Next, hold the navigation controller fixed and test whether any recognition improvement translates into local homing success.

Before collecting final results, set a spatial success tolerance, total action budget, number of independent scenes/routes, appearance conditions, and model-selection rule. Record the complete configuration and seeds with each run. Keep pilot runs separate from final results.
