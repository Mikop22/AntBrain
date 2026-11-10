# Pre-training review: pixel baselines and shared data

**Status: PASS.** Independent reviewer: `review_level1_pixels_data`.

No model training, optimizer steps, retrieval scoring, or test-set model metrics were executed during this review. Only source inspection, data validation, training-image inspection, and non-training unit tests were used. Approval covers the files and dataset identified below; optimizer, checkpoint selection, and shared evaluation require the other reviews.

## Checks and findings

- Raw RGB descriptors and per-image global brightness/contrast normalization match the existing baseline ranking semantics. Encoding is deterministic, nonmutating, finite for constant images, and independent of other images in the batch. No fitted split statistics, spatial labels, or scene IDs enter encoding.
- Native observations are four compass-aligned RGB views, stored as a 32 × 128 strip. Separate geometry, observation, and query RNG streams produce independent captures. Train, validation, and test scenes are disjoint.
- Familiar queries remain within 0.20 m of their reference; unfamiliar queries are at least 0.60 m from every reference. Four declared conditions are represented: clean, exposure, occlusion, and combined. Perturbations are synthetic sensor changes, not physical relighting or biological sensing.
- Requested schema/provenance fixes are present: exact four-camera shape, integer IDs, finite and consistent spatial thresholds, scene-config membership, declared sample counts, and familiar-query/condition consistency. Hash corruption and misleading metadata regressions pass.
- Nearby reference places remain useful evaluation distractors. Training must use the shared canonical-spacing and actual cross-view distance guards specified in the protocol; the data alone does not guarantee every differently labelled view is a safe negative. The common-trainer reviewer checks those guards separately.
- Memory accounting must distinguish the 12,288-byte uint8 source image from the 49,152-byte float32 pixel-descriptor cache. Constant images become indistinguishable after normalization.

## Evidence

`pytest tests/test_level1_pixels.py tests/test_level1_data.py -q`: **25 passed**. Ruff checks and formatting pass for all four reviewed files. A training-only condition montage was visually inspected; no held-out model performance was inspected or used to revise the dataset.

The canonical collection validates as 44 scenes: 24 training, 8 validation, and 12 test scenes. It contains 2,112 gallery observations, four repeat observations per place, and 9,504 queries. The held-out test set has 576 references, 2,304 familiar queries, and 288 unfamiliar queries. These are data counts, not model results.

| Artifact | SHA-256 |
| --- | --- |
| `images.npz` | `33fbd02f0cf8c3f354f38bdd4313b7cbe3b9d12a90fd08cb5823595f2f761ff1` |
| `metadata.json` | `1710562a24b327948a3e6b7c957ff94f1c6ddad62c7cd2520bd679d5ebbac9e2` |
| `src/ant_homing/level1/pixels.py` | `5ebb0589145d8f24b09f281b2bd5739e37997169c01c12c6c35f8abfbe6b681a` |
| `src/ant_homing/level1/data.py` | `f1bc446416b264426f9012669b97541f1a163d274a50b7794eb762503ad180fb` |
| `tests/test_level1_pixels.py` | `e8c1c07cf7980d32a21256f49139dbce14ceaa8e09bdec54993bbdb6f36b8ab3` |
| `tests/test_level1_data.py` | `eb2f228ad790a79fed68b9e6a5558c83b508d9cb419b3e96432bbe5b025d47aa` |

Both the array archive and metadata should be frozen by the training gate. Changes to reviewed implementation or dataset contracts require a focused re-review before optimization.
