# Matched-encoder loss comparison

This protocol is frozen before optimization and fresh-test model scoring. The experiment compares loss recipes while holding the encoder and training inputs fixed. It does not compare different architectures or claim an optimal tuning of either objective.

## Fixed design

- Both arms use the original 42,728-parameter spatial CNN, shared construction and encoding functions, 64D unit descriptors, and identical 32-by-128 RGB observations.
- Arm 1: original paired symmetric InfoNCE, temperature 0.1, averaged over visit pairs (0,1) and (2,3).
- Arm 2: the already reviewed Multi-Similarity implementation, alpha 1, beta 50, base 0, epsilon 0.1, pinned pytorch-metric-learning v2.3.0 mean-all-anchor semantics. All four visits provide positive pairs. Its mining and positive-pair handling are part of the loss recipe being changed, not separately isolated effects.
- Train from scratch at seeds 0, 1, 2. Initial model-state hashes must match within every seed. No pretrained weights or pilot warm starts.
- Both arms use the existing unchanged trainer: Adam at 0.001, 20 epochs, one CPU Torch thread, identical within-scene batches, spatial negative guards, visit permutations, and image exposure. No loss-specific hyperparameter tuning.
- Select the earliest checkpoint achieving maximum validation spatial Recall@1. Calibrate each selected checkpoint's familiarity threshold on validation familiar-query distances at the 95th percentile. The selection rule is common; selected epochs and thresholds may differ as consequences of training.

## Data and leakage controls

The reviewed generator creates a separate archive with the original training and validation scene seeds, and fresh test seeds 400–411. Every training/validation array must be identical to the pilot; this is checked before the review gate is frozen. Image sizes, place spacing, jitter, exposure/occlusion conditions and counts stay the same. Each arm receives the same archive.

The data contain 24 training, eight validation, and 12 test scenes, 48 reference places per scene, four visits per training place, and four nuisance conditions. Fresh test scenes contain 576 references, 2,304 familiar queries and 288 unfamiliar queries. Familiar queries are within 0.20 m of their reference; unfamiliar queries are at least 0.60 m from every reference. The intervening distance interval remains untested.

Three independent reviews cover the loss/encoder interface, shared training/data fairness, and frozen evaluation/provenance. The new gate binds their reports, all required source files, this protocol, data, and metadata hashes. No optimizer steps occur before these approvals. All six checkpoints and validation thresholds must be frozen before any fresh-test scoring.

## Outcomes and interpretation

Primary outcome: unfamiliar false acceptance at the frozen validation-calibrated threshold, alongside actual familiar acceptance. Also report familiarity AUROC, spatial Recall@1/5 at 0.25 m, exact place accuracy, spatial-error quantiles, correct-and-accepted rate, condition results, seed variation and validation learning curves.

Paired scene bootstrap intervals resample the 12 complete test scenes, conditional on the three training seeds. Differences are MS minus InfoNCE. For AUROC, scene intervals describe mean per-scene AUROC differences; pooled AUROC is reported separately. Frames and repeated predictions from different seeds are not treated as independent samples.

A result supports an effect of these loss recipes on this fixed encoder and common optimization budget. It does not isolate mining from the loss formula, establish best achievable performance, explain every difference from the earlier camera-aware model, or demonstrate homing. Test outcomes will not be used to tune this experiment.
