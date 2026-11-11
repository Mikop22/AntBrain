# Same encoder, different loss

This experiment compares the InfoNCE and Multi-Similarity recipes on exactly the same 64D encoder. Training and validation arrays match the first pilot; fresh test layouts prevent reusing its inspected test set. The [protocol](PROTOCOL.md), [three independent reviews](reviews/), and [review manifest](review-manifest.json) were frozen before training.

**Completed result:** InfoNCE has lower unfamiliar false acceptance (8.22% versus 13.43%) at essentially equal familiar acceptance. Read the [assessment](ASSESSMENT.md) and [full results](results/RESULTS.md).

## Reproduce

Run from the repository root after `bash scripts/bootstrap.sh`. The original pilot data must exist at `artifacts/level1/data`; its generation command is in the [Level 1 guide](../README.md).

```bash
.venv/bin/python scripts/prepare_loss_ablation_data.py \
  --pilot artifacts/level1/data --output artifacts/level1-loss-ablation/data

.venv/bin/python -m ant_homing.level1.loss_ablation train \
  --data artifacts/level1-loss-ablation/data \
  --output artifacts/level1-loss-ablation/training \
  --reviews docs/level1/loss-ablation/review-manifest.json

.venv/bin/python -m ant_homing.level1.loss_ablation evaluate \
  --data artifacts/level1-loss-ablation/data \
  --run-dir artifacts/level1-loss-ablation/training \
  --output artifacts/level1-loss-ablation/evaluation

.venv/bin/python scripts/report_loss_ablation.py \
  --data artifacts/level1-loss-ablation/data \
  --run-dir artifacts/level1-loss-ablation/training \
  --evaluation artifacts/level1-loss-ablation/evaluation \
  --output docs/level1/loss-ablation/results
```

Use new output directories for reruns; existing completed runs are protected against overwrite. NumPy suffices for reports; optional matplotlib adds validation plots. The review gate rejects changes to reviewed sources, protocol, data or metadata. The trained report has a digest marker so threshold edits after freezing are rejected. All six checkpoints pass provenance checks before any test scoring starts.

The original pilot and this experiment have different test images. Compare losses within this experiment; raw percentages from the two studies are not a paired comparison.

Data, weights, full per-query results, and a compact run bundle remain generated artifacts under `artifacts/level1-loss-ablation/`. Tracked results retain summaries, plots, metadata and exact artifact fingerprints.
