# Pre-training review: ConvAP and shared evaluation

**Status: PASS.** Independent reviewer: `review_level1_final` (third reviewer, ConvAP/evaluation role). Reviewed at `2026-11-10T21:30:39Z`.

No training, optimizer steps, or canonical dataset model scoring were performed. Verification used source inspection, synthetic forward/loss/backward tests, and the five `tests/test_level1_*.py` modules only. No blocking findings remain.

- The project-specific ConvAP adaptation uses one shared CNN on four separate camera panels, preserves E/S/W/N order and 2-by-2 spatial pooling, and produces normalized 128-dimensional descriptors. It has 91,328 parameters; it is not the original ResNet Conv-AP model.
- Multi-Similarity uses cosine similarity, strict positive/negative mining with epsilon 0.1, alpha 1, beta 50, and base 0. The formula, inactive-anchor mean reduction, and differentiable-zero guard when both directed pair counts are at most one agree with the public `pytorch-metric-learning` v2.3.0 source. The sparse-pair regression and finite-gradient/no-weight-update tests pass.
- Every approach receives the same fixed RGB captures and each query searches only its scene's gallery. Spatial Recall@1/5 uses 0.25 m; exact place identity is reported separately. Familiar queries and novel queries follow the validated dataset's 0.20 m and at-least-0.60 m contracts.
- Thresholds use the validation familiar-distance 95th percentile only. AUROC awards half credit to ties. Paired uncertainty resamples scenes, with its scope explicitly conditional on the chosen training seeds. Warm-up and single-observation latency use training images.
- The training gate requires all three approvals, mandatory source hashes, and both data and metadata hashes. Evaluation verifies the copied manifest's exact hash, frozen checkpoints, and checkpoint module, seed, split, image shape, camera convention, dataset and review provenance. Test scoring is reached only after the training report is frozen.
- Architecture, descriptor size, and objective change together. Findings must be described as pipeline/approach comparisons, without attributing an observed difference solely to pooling or loss.

Evidence: **52 passed** via `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_level1_convap.py tests/test_level1_experiment.py tests/test_level1_infonce.py tests/test_level1_pixels.py tests/test_level1_data.py -q`. Ruff lint and format checks passed for the three reviewed implementation files and two corresponding test files. Pinned source inspection covered `MultiSimilarityLoss`, `MultiSimilarityMiner`, `GenericPairLoss`, and `MeanReducer` at the reference URL recorded in `LOSS_REFERENCE`.

| Reviewed artifact | SHA-256 |
| --- | --- |
| `src/ant_homing/level1/convap.py` | `1500c66b790986cc0421717b90457acd4952cc26c25f86b4619753efb7c99e0e` |
| `src/ant_homing/level1/evaluation.py` | `f40dfb5b582f121f5282c0ebd0b676e05f288264a20f3d4e496576b696bb01be` |
| `src/ant_homing/level1/experiment.py` | `90d836f095ce53c12c453d91df9a2a8c29b736d8b0b33dad2f334e26b7238d35` |
| `tests/test_level1_convap.py` | `3c61db7a3728b9a6dbb6ccf9a01d8bfbe4e44668b751da6947fac14b7047f5dd` |
| `tests/test_level1_experiment.py` | `669ae37451af6aec68dd87ba6960c72a1bb1b6b7e4a8ea7db680b5bdd5dd992f` |
| `artifacts/level1/data/images.npz` | `33fbd02f0cf8c3f354f38bdd4313b7cbe3b9d12a90fd08cb5823595f2f761ff1` |
| `artifacts/level1/data/metadata.json` | `1710562a24b327948a3e6b7c957ff94f1c6ddad62c7cd2520bd679d5ebbac9e2` |
