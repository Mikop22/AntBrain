# Pre-training review: InfoNCE and shared training

**Status: PASS.** Independent reviewer: `review_level1_infonce_training`.

Reviewed `level1/infonce.py`, the shared training and review-gate paths in `level1/experiment.py`, and their tests. Dataset generation/validation and Conv-AP/loss evaluation have separate reviewers.

- The 42,728-parameter, 64-dimensional encoder matches the original architecture. Its image-only interface validates native inputs and normalizes descriptors.
- The symmetric InfoNCE formula and four-observation pair averaging agree with independent analytic tests. Gradients are finite; forward/backward checks do not update weights.
- Both neural approaches receive identical seeded within-scene place batches and per-place visit permutations. All training places are covered. Canonical separation is at least 0.9 m, and actual cross-view negative separation is checked above 0.5 m.
- Model selection uses validation spatial Recall@1 only, with the earliest checkpoint winning a tie. Familiarity thresholds use validation examples. Test scoring occurs after training is frozen.
- The training entry point verifies all three approvals, mandatory source fingerprints, dataset hash, and metadata hash before reaching an optimizer update. Evaluation rechecks frozen provenance and checkpoint hashes.
- Native four-camera shape and four observations per place are checked before training. Runtime/loss configuration is recorded, and warm-up explicitly selects training images.

Findings resolved before approval: metadata was initially unhashed; the source list could omit required files; warm-up relied on array ordering; fixed nuisance-condition pairs were replaced with shared seeded visit permutations.

Independent checks: **18 tests passed** (13 InfoNCE and 5 shared experiment checks), plus Ruff and formatting checks. Additional manual checks verified deterministic batching, training-scene filtering, full place coverage, and spatial separation.

No training or optimizer steps were performed during this review. No blocking findings remain. The comparison changes architecture and objective together, so resulting differences must be described as approach-level differences rather than attributed solely to the loss or pooling method.
