# Level 1 comparison protocol

This study compares complete visual-memory approaches under a shared simulation/data budget. It does not evaluate homing or establish biological fidelity.

## Approaches

1. Raw RGB and per-image brightness/contrast-normalized matching. Neither baseline is fitted.
2. Original 42,728-parameter spatial CNN, 64D unit descriptors, symmetric paired InfoNCE (temperature 0.1).
3. Camera-aware 91,328-parameter Conv-AP-inspired CNN, 128D unit descriptors, mined Multi-Similarity loss (alpha 1, beta 50, base 0, epsilon 0.1). Loss/miner semantics are pinned to pytorch-metric-learning v2.3.0, including mean-all-anchor reduction and its sparse-pair guard.

One observation contains four compass-aligned 32 × 32 cameras in E,S,W,N order. Camera faces are not positive examples. Positives are independently captured repeat observations of the same place. This is supervised metric learning using simulator poses to define training groups; inference receives RGB only.

## Frozen data

The archive contains 44 scene seeds, disjoint between training (24), validation (8), and test (12). Each scene has 48 reference places and four repeat observations per place. There are 2,112 reference places and 9,504 queries in total. The test split contains 576 references, 2,304 familiar queries, and 288 unfamiliar queries.

Familiar queries are within 0.20 m of their reference. Unfamiliar queries are at least 0.60 m from every reference. Four conditions are used: clean, exposure, partial camera occlusion, and combined. These conditions use the same distributions in training and evaluation; combined is not an out-of-distribution test. Perturbations are post-render sensor effects rather than physical relighting. Nearby reference pairs (0.52–0.68 m) provide difficult evaluation distractors.

The NPZ and metadata are hashed separately. Training cannot begin until three independent review approvals and required source hashes match the review manifest. Review reports are retained in `reviews/`.

## Shared training budget

- CPU, one PyTorch thread per run; no pretrained weights.
- Seeds 0, 1, 2 for each neural approach; 20 epochs; Adam learning rate 0.001.
- Same seeded within-scene batches, up to 16 places × four observations.
- Canonical inter-place spacing at least 0.9 m within a batch, with actual cross-view negative spacing explicitly checked above 0.5 m.
- All training places are covered each epoch; a separated support place is reused if needed for a final singleton. Both approaches receive identical batches and image exposures.
- Repeat-view order is independently permuted per place using the same seed/epoch/batch rule for both approaches. InfoNCE averages pairs (0,1) and (2,3); Multi-Similarity uses all valid positive/negative pairs.
- Select the checkpoint with highest validation spatial Recall@1; earliest epoch wins ties. Freeze all checkpoints and validation-calibrated thresholds before scoring test queries.

Architecture, loss, and descriptor size change together. This compares approaches, not the isolated causal effect of a loss or pooling method. The common budget is not a per-model hyperparameter search.

## Evaluation

All approaches receive identical reference/query images and search within the query's scene gallery. Scene membership is assumed known. Report spatial Recall@1/5 at 0.25 m, exact reference identity accuracy, median/90th-percentile position error, and familiarity AUROC. Queries with no valid remembered place evaluate rejection, not retrieval success.

For each approach/run, select the familiarity threshold at the 95th percentile of familiar validation-query distances. Report actual test familiar acceptance and unfamiliar false acceptance at that frozen threshold; test acceptance is not assumed to remain 95%.

Report condition-level and scene-level results. Neural results are averaged across three training seeds. Paired confidence intervals resample complete test scenes, conditional on those training seeds; individual correlated frames are not bootstrap units.

CPU measurements separate batched feature-extraction throughput from sequential single-observation latency. Memory reporting separates model-state bytes, per-reference float32 descriptor-cache bytes, and original uint8 RGB storage. The pixel cache uses 49,152 bytes per reference; the source image itself uses 12,288 bytes. No total-memory claim should omit the neural model or silently substitute source images for cached features.

## Limits

Procedural arenas, fixed wall appearance, ideal compass alignment, small scene counts, and synthetic sensor perturbations limit generalization claims. Recognition performance is not homing performance. A subsequent study should test heading uncertainty, different scene families, real relighting, and navigation with noisy odometry.
