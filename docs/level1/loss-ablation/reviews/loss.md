# Pre-training review: matched-encoder loss interface

**Status: PASS.** Independent reviewer: `review_ablation_loss`. Reviewed at `2026-11-11T01:42:47.512744+00:00`.

No optimizer steps, fitted-model scoring, or fresh-test image/score inspection were performed. Verification used source inspection and synthetic forward/loss/backward tests only. No blocking findings remain.

- `infonce_ms.build_model` and `infonce_ms.encode` directly re-export the original InfoNCE functions. Both arms therefore use the same 42,728-parameter CNN, 64-dimensional unit descriptors, image validation and normalization. Seeded initial-state fingerprints match at seeds 0, 1, and 2.
- The wrapper consumes one complete four-camera observation per visit. Place-major flattening and `repeat_interleave(4)` labels preserve place identity across all four visits; camera panels are not treated as positive observations.
- The only new training computation is the call to the existing reviewed Multi-Similarity loss with unchanged alpha 1, beta 50, base 0 and epsilon 0.1. Its scalar reference, strict miner inequalities, inactive-anchor reduction and sparse-pair differentiable-zero checks pass. The wrapper matches the direct reviewed loss and produces finite parameter gradients without changing model state.
- Both objectives consume all four visits once through the same encoder. InfoNCE averages two paired symmetric objectives; MS uses all repeated-view positive pairs and mined negatives. The protocol correctly identifies this as a loss-recipe intervention, including pair handling/mining, rather than an isolated change to a scalar formula.
- Explicit re-export declarations and the review-fixture dictionary comprehension resolved initial Ruff findings. No mathematical or encoder-interface changes were required.

Evidence: **32 passed** via `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_loss_ablation.py tests/test_level1_convap.py tests/test_level1_infonce.py -q` (2.19 s). Ruff lint and format checks passed for the wrapper, reused objective modules and ablation tests.

| Reviewed artifact | SHA-256 |
| --- | --- |
| `src/ant_homing/level1/infonce_ms.py` | `bc070677974c4384b7f6caeab7bcbee716e7c9e1de3ea535bb5605b2e4d0e652` |
| `src/ant_homing/level1/infonce.py` | `7406a0aeaaed3e5f153f64876ca91917befa1d067bb198ff7fe980874d2a7710` |
| `src/ant_homing/level1/convap.py` | `1500c66b790986cc0421717b90457acd4952cc26c25f86b4619753efb7c99e0e` |
| `tests/test_loss_ablation.py` | `408dd4c7673806a52342719ff80d77bf30d3421df0aa65f0feb153f1ed04328e` |
| `docs/level1/loss-ablation/PROTOCOL.md` | `8cb811d400fd04a1155366247778a7730b9912acca92c3485273007b923a147d` |
