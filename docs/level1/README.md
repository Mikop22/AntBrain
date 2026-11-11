# Level 1 visual-memory experiments

Three independent builders implemented pixel matching, the original InfoNCE CNN, and a camera-aware Conv-AP-inspired CNN with Multi-Similarity learning. Three independent reviewers approved the implementations and shared data/training/evaluation code **before optimization**. See the [protocol](PROTOCOL.md), [reviews](reviews/), and [frozen review manifest](review-manifest.json).

The completed pilot favors **InfoNCE for learned familiarity and compact memory**, while normalized pixels remain strongest at familiar-place retrieval. Read the [assessment](ASSESSMENT.md) and [held-out results](results/RESULTS.md).

The [familiarity analysis](FAMILIARITY.md) examines why unfamiliar-place false acceptance differs, using the frozen query scores without retraining.

The [matched-encoder experiment](loss-ablation/README.md) holds the encoder, initialization, training inputs and budget fixed while comparing the two loss recipes on fresh test scenes.

![Training-only appearance examples](data-preview.png)

Rows show training places; columns show the reference and independent clean, exposure, occlusion, and combined visits. Each cell contains all four camera directions. These examples are training data, not held-out results.

## Reproduce the study

From the repository root, install the ML/development extras with `bash scripts/bootstrap.sh`. Generate the canonical data only if the output directory does not already exist:

```bash
.venv/bin/python - <<'PY'
from ant_homing.level1.data import generate_dataset
generate_dataset("artifacts/level1/data")
PY

.venv/bin/python -m ant_homing.level1.experiment train \
  --data artifacts/level1/data --output artifacts/level1/training \
  --reviews docs/level1/review-manifest.json --epochs 20 --seeds 0,1,2

.venv/bin/python -m ant_homing.level1.experiment evaluate \
  --data artifacts/level1/data --run-dir artifacts/level1/training \
  --output artifacts/level1/evaluation

.venv/bin/python scripts/report_level1.py \
  --data artifacts/level1/data --training artifacts/level1/training \
  --evaluation artifacts/level1/evaluation --output docs/level1/results
```

Use new output directories for new runs. The training gate rejects missing reviews or changed source/data/metadata. The saved manifest identifies the reviewed collection and implementation; investigate a hash mismatch rather than silently replacing the hashes. Changes require focused review before a new training gate is created.

Report tables require NumPy; plots additionally use optional matplotlib. The report still writes complete tables and JSON if matplotlib is unavailable. Plotting dependencies do not affect training or scored predictions.

## Outputs

- `artifacts/level1/data/`: arrays, validated metadata, and a training-only appearance preview.
- `artifacts/level1/training/`: six validation-selected neural checkpoints, histories, calibrated thresholds, copied reviews, and the final frozen manifest.
- `artifacts/level1/evaluation/comparison.json`: per-query, condition, scene, familiarity, timing, and memory results for every approach/run.
- `docs/level1/results/`: concise tracked results, summary JSON, and plots when available.

Data and model files are local generated artifacts and are not committed to Git. Tracked [artifact fingerprints](results/artifacts.json) identify the exact archive, checkpoints, histories, and complete evaluation; [data metadata](results/data-metadata.json) records the scene configurations and split counts.

`artifacts/level1/run-bundle.zip` preserves the six trained checkpoints, histories, frozen review manifest, complete per-query evaluation, and report. It excludes the image archive; generate that with the reviewed generator above and verify its fingerprint before using the saved models.

## Use a trained encoder

Level 1 checkpoints use the Level 1 module interface. For example, encode a current simulator observation with the seed-0 InfoNCE checkpoint:

```python
import torch
from ant_homing.level1 import infonce
from ant_homing.level1.experiment import neural_features
from ant_homing.simulation import Arena, SimConfig

saved = torch.load(
    "artifacts/level1/training/infonce-seed-0/model.pt",
    map_location="cpu", weights_only=True,
)
model = infonce.build_model()
model.load_state_dict(saved["state_dict"])
with Arena(SimConfig(arena_size=8.0, landmark_count=24)) as arena:
    descriptor = neural_features(infonce, model, arena.observe().rgb[None])
assert descriptor.shape == (1, 64)
```

Reference and query observations must use the same camera convention. Familiarity compares the nearest **squared** descriptor distance with that run's saved threshold in `trained.json`. A descriptor or familiarity score alone does not choose a movement; controller evaluation is the next experiment.

## Implementation references

- [Conv-AP / GSV-Cities](https://github.com/amaralibey/gsv-cities): spatial aggregation. Our small shared four-camera encoder is a project-specific adaptation, not the published ResNet architecture.
- [BoQ](https://github.com/amaralibey/Bag-of-Queries) and [SALAD](https://github.com/serizba/salad): supervised visual-place metric-learning recipes.
- [pytorch-metric-learning v2.3.0](https://github.com/KevinMusgrave/pytorch-metric-learning/tree/v2.3.0): the explicitly pinned loss/miner semantics checked against independent scalar tests.
- [CliqueMining](https://github.com/serizba/cliquemining): motivation for informative negatives and geographically precise place recognition. This experiment uses a shared spatial guard and within-scene batches, not the full CliqueMining algorithm.

These experiments evaluate recognition and familiarity. They do not demonstrate successful ant homing or a biologically faithful memory mechanism.
