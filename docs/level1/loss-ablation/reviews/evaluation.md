# Pre-training review: frozen evaluation and provenance

**Status: PASS.** Independent reviewer: `/root/review_ablation_evaluation`. Reviewed at `2026-11-11T01:42:54.586341+00:00`.

No optimizer steps, canonical-model scoring, or fresh-test image inspection occurred during this review. Inspection covered the ablation runner, unchanged shared trainer/evaluator, no-update tests and frozen protocol. No blocking findings remain.

- Checkpoint selection is validation spatial Recall@1 with earliest-tie selection; familiarity thresholds use only familiar validation distances. Fresh-test seeds 400–411 are fixed, with 576 references, 2,304 familiar queries and 288 unfamiliar queries. Original pilot source fingerprints remain unchanged.
- The runner writes the frozen report only after all six runs finish and paired initialization/budget checks pass. A SHA-256 marker binds the complete final report, including selected thresholds. Evaluation rejects a missing marker or any changed report before model scoring. This fixes the initially identified threshold-binding gap.
- All six checkpoint hashes and module, seed, selected epoch, dataset, review, scene split, image shape and camera convention are verified before any test scoring. A synthetic bad-sixth-checkpoint regression confirms that even the first arm is not scored when a later checkpoint has invalid provenance.
- Review gating binds required source files, protocol, dataset, metadata, all three independent reports and the fixed budget. Evaluation rechecks the exact copied manifest hash and frozen dataset/metadata identity.
- Each arm uses the same test images and scene-local gallery, RGB-only feature inputs, 0.25 m spatial tolerance, familiar-query retrieval metrics and unknown-query rejection metrics. AUROC has correct tied-score handling; acceptance includes equality at the threshold.
- Confidence intervals bootstrap twelve complete paired scenes after averaging training seeds. Differences are MS minus InfoNCE. The AUROC interval describes the mean per-scene AUROC difference and is explicitly distinguished from pooled AUROC. Frames and repeated predictions are not independent bootstrap units.
- The protocol appropriately limits causal interpretation to the two loss recipes on a fixed encoder and shared budget. Mining and positive-pair handling change with the recipe; no separate mathematical-loss-only or best-tuned-performance claim is justified.

Verification: **15 passed** via `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_loss_ablation.py tests/test_level1_experiment.py -q`. Synthetic independent checks also matched brute-force tied AUROC, inclusive threshold acceptance and constant paired-scene intervals. Ruff lint and format checks passed for the runner and ablation tests. Tests perform forward/backward checks and synthetic checkpoint serialization but no optimizer updates or canonical-model scoring.

| Reviewed artifact | SHA-256 |
| --- | --- |
| `src/ant_homing/level1/loss_ablation.py` | `fd5da6f8e86034a6eba6d8bd3ebb4aec73857acfa23f9bb2ffe5788fc910c81d` |
| `src/ant_homing/level1/evaluation.py` | `f40dfb5b582f121f5282c0ebd0b676e05f288264a20f3d4e496576b696bb01be` |
| `src/ant_homing/level1/experiment.py` | `90d836f095ce53c12c453d91df9a2a8c29b736d8b0b33dad2f334e26b7238d35` |
| `tests/test_loss_ablation.py` | `408dd4c7673806a52342719ff80d77bf30d3421df0aa65f0feb153f1ed04328e` |
| `docs/level1/loss-ablation/PROTOCOL.md` | `8cb811d400fd04a1155366247778a7730b9912acca92c3485273007b923a147d` |
