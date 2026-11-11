# Pre-training review: shared training and data fairness

**Status: PASS.** Independent reviewer: `review_ablation_training`. Reviewed at `2026-11-11T01:42:39.285978+00:00`.

No optimizer steps, canonical model scoring, or fresh-test image inspection were performed. No blocking findings remain.

- Both objectives call the unchanged `experiment.train_one`, which resets the Torch seed before constructing the shared encoder, uses Adam at 0.001 for 20 epochs, and builds batches and visit permutations exclusively from the common seed/epoch/batch rules. Model updates cannot influence that input stream. The paired initialization hashes, parameter/descriptor counts, and per-epoch update/exposure checks provide additional guards.
- Independently loaded and validated both archives, then checked all 15 training/validation arrays and their dtypes against the pilot. Every array is identical. Training scenes 100–123 and validation scenes 200–207 are unchanged. Fresh test scenes 400–411 contain the expected 576 references, 2,304 familiar and 288 unfamiliar queries. Only training/validation contents were compared.
- Selection uses highest validation spatial Recall@1 with the earliest tie, and calibration uses only familiar validation scores. Different selected epochs and thresholds are expected consequences of the changed objective under the same selection rule.
- The gate requires three PASS roles, binds report hashes, mandatory source/protocol/test hashes, data and metadata hashes, and the fixed budget. It runs before the shared trainer. Six completed runs and frozen thresholds precede test scoring; the final report digest marker now detects accidental post-freeze threshold changes.
- The protocol accurately describes a loss-recipe comparison: Multi-Similarity includes its miner and all-repeat positive handling, while InfoNCE retains paired symmetric supervision. No isolated formula/mining superiority or best-tuned-loss claim is warranted.

Evidence: **15 passed** with `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_loss_ablation.py tests/test_level1_experiment.py -q`. Ruff lint passed for the two new implementation modules and the ablation tests. Review chronology must remain: all three PASS reports and the new manifest before the first training invocation.

| Reviewed artifact | SHA-256 |
| --- | --- |
| `src/ant_homing/level1/loss_ablation.py` | `fd5da6f8e86034a6eba6d8bd3ebb4aec73857acfa23f9bb2ffe5788fc910c81d` |
| `src/ant_homing/level1/experiment.py` | `90d836f095ce53c12c453d91df9a2a8c29b736d8b0b33dad2f334e26b7238d35` |
| `src/ant_homing/level1/infonce_ms.py` | `bc070677974c4384b7f6caeab7bcbee716e7c9e1de3ea535bb5605b2e4d0e652` |
| `tests/test_loss_ablation.py` | `408dd4c7673806a52342719ff80d77bf30d3421df0aa65f0feb153f1ed04328e` |
| `docs/level1/loss-ablation/PROTOCOL.md` | `8cb811d400fd04a1155366247778a7730b9912acca92c3485273007b923a147d` |
| `artifacts/level1-loss-ablation/data/images.npz` | `09654a6df1675bed6c701e044881644df750adbb5de460d56b6686b467111f8e` |
| `artifacts/level1-loss-ablation/data/metadata.json` | `88ac941badc2926e2956f6cb37bf80898abe167a19e0619c82c00c349b0e8b59` |
| `artifacts/level1-loss-ablation/data/identity-check.json` | `46598adbd888faeeff2d3b92367e7f9df3959ea8b1e9e4748b3275f5fb4930eb` |
