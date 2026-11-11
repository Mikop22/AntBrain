# Level 1 assessment

Use **InfoNCE as the learned visual-memory candidate** for the next navigation experiment, and retain normalized pixels as a required baseline. The present evidence does not justify choosing the larger ConvAP adaptation.

A subsequent [matched-encoder experiment](loss-ablation/ASSESSMENT.md) supports this choice: on fresh scenes, InfoNCE has 8.22% unfamiliar false acceptance versus 13.43% for Multi-Similarity on the same encoder, with essentially equal familiar acceptance. Its retrieval difference is uncertain. The pilot results below remain the original architecture-and-loss comparison.

## What the results establish

Normalized pixels have the highest familiar-place Recall@1: **98.87%**, compared with **96.21%** for InfoNCE and **94.84%** for ConvAP + Multi-Similarity. The paired scene intervals exclude zero for both learned models versus normalized pixels, and for ConvAP versus InfoNCE. Learning has not improved top-match retrieval on this benchmark.

InfoNCE is better at recognizing when no remembered place is appropriate. At thresholds calibrated on validation data, normalized pixels falsely accept **48.61%** of unfamiliar queries, versus **6.25%** for InfoNCE and **15.05%** for ConvAP. Familiar-query acceptance remains approximately 95% for all three. InfoNCE familiarity AUROC is **0.989**, versus 0.866 and 0.972 respectively. These are averages over three neural training seeds, not a guarantee for an individual episode.

The learned representations also reduce cache size: InfoNCE stores 256 bytes per reference, ConvAP 512 bytes, and the implemented float32 pixel baseline 49,152 bytes. The source RGB image is only 12,288 bytes; storing images and recomputing features is another possible baseline implementation. Model state adds 166.9 KiB for InfoNCE and 356.8 KiB for ConvAP, excluding runtime overhead. CPU encoding takes approximately 0.70 ms and 1.40 ms per observation respectively, excluding rendering and gallery search.

These findings support a practical choice for a controller that must detect unfamiliar views. They do not establish improved homing. Rejection is imperfect, and an accepted match can still be incorrect; the [results](results/RESULTS.md) include correct-and-accepted rates separately.

## How to interpret the comparison

The strong pixel baseline is plausible in this task: views are compass-aligned, query offsets are small, and brightness/contrast normalization directly addresses one of the simulated perturbations. This is an interpretation of the task design, not a measured causal explanation. Generalization beyond the procedural scene family remains untested.

ConvAP is a small project-specific adaptation trained from scratch, not the published pretrained architecture. Architecture, loss, and descriptor size changed together. Its lower scores under a shared 20-epoch budget do not show that Multi-Similarity learning or published Conv-AP methods are generally inferior. No hyperparameters were changed after observing these test results.

## Next research experiment

1. Freeze this benchmark as a completed pilot. Use validation scenes for development and reserve fresh scene seeds and scene families for the next final test.
2. Compare InfoNCE and normalized pixels in the same local-homing controller, using the saved validation thresholds. Include an odometry baseline and account for every sensing probe and movement.
3. Test heading error, larger start displacements, larger arenas, repeated-looking landmarks, physical lighting changes, and unfamiliar trajectories. Report return success, path length, collisions, rejected observations, and false confident matches over multiple scenes and seeds.
4. If studying why a model works, run controlled architecture/loss ablations with matched descriptor size and a validation-based tuning budget. The current comparison cannot isolate those causes.

## Verification

Three independent pre-training reviews passed before optimization. All six runs completed 20 epochs; training batches, update counts, and image exposure matched between models within each seed. Checkpoints were selected using validation Recall@1 and frozen before held-out scoring. A subsequent independent read-only audit recomputed saved-query metrics and checked hashes, chronology, checkpoint selection, aggregation, and scene confidence intervals. The full software suite passed **114 tests**.

See the [protocol](PROTOCOL.md), [results](results/RESULTS.md), and [pre-training reviews](reviews/) for the assumptions and evidence.
