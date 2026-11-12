# Research roadmap & protocol

## Stages

1. **Level 1 — Visual memory (Completed):** Compare learned embeddings against normalized pixel distances on scene-disjoint place retrieval and unfamiliar-place rejection. Final retained models (`InfoNCE CNN` and `Normalized Pixels`) are saved in [`models/level1/`](../models/level1/manifest.json).
2. **Level 2 — Local homing (Next):** Characterize the spatial familiarity catchment basin ($0.0\text{–}1.0\text{ m}$ displacement sweep) and compare `InfoNCE CNN` vs. `Normalized Pixels` inside closed-loop homing controllers.
3. **Level 3 — Combined navigation:** Combine noisy dead-reckoning odometry with visual familiarity handoff and correction.

## Evaluation rules

- **Information boundary:** Controllers receive only RGB images, noisy odometry, compass heading, and collision flags. Ground-truth pose and scene geometry never enter action selection.
- **Physical probing cost:** Visual probes are executed movements; their path length and step count are included in episode totals.
- **Disjoint splits:** Training, validation, and test scenes are strictly disjoint. Thresholds and checkpoints are frozen on validation data before test evaluation.
- **Scene-level uncertainty:** Bootstrap confidence intervals resample scenes/routes as experimental units rather than correlated frames.
