# Why unfamiliar-place acceptance differs

InfoNCE's frozen distance scores separate familiar and unfamiliar observations more strongly than the other approaches in this pilot. The difference persists after equalizing acceptance and on clean observations. This is evidence about the resulting representations; it does not isolate which training choice caused the difference.

## What the decision means

For each query, every method finds the closest remembered image in the same scene. It accepts that match only if its squared descriptor distance is below its own validation-calibrated threshold. The threshold targets 95% familiar acceptance. It is not a predicted probability.

A correct ranking does not guarantee trustworthy absolute distance. Normalized pixels can rank the correct familiar reference first while also giving low distances to unfamiliar locations. This explains how 98.87% familiar-place retrieval can coexist with 48.61% unfamiliar false acceptance.

Here an unfamiliar query is an unremembered position within a remembered arena, at least 0.60 m from every reference. It is not an entirely new arena or object. Familiar queries are within 0.20 m of their assigned reference. The intervening distances are not evaluated by this benchmark.

## Measured checks

| Approach | Published familiar acceptance | Published unfamiliar false acceptance | Diagnostic unfamiliar acceptance at common 95% test familiar acceptance |
| --- | ---: | ---: | ---: |
| Raw pixels | 95.75% | 89.58% | 86.81% |
| Normalized pixels | 94.70% | 48.61% | 48.96% |
| InfoNCE | 95.37% | 6.25% | 5.44% |
| ConvAP + MS | 94.94% | 15.05% | 15.28% |

The last column is a **post-hoc ROC diagnostic using test labels**, not a new calibrated setting or independently validated performance claim. No saved threshold was changed. The gap remains, so small differences in actual test familiar acceptance do not explain it.

Different numerical distance units also do not explain the gap: AUROC depends on score ordering. On clean observations alone, mean familiarity AUROC is 0.907 for normalized pixels, 0.996 for InfoNCE, and 0.990 for ConvAP. Thus synthetic corruption is not the entire explanation. Conditions use different query positions, so their differences do not isolate the causal effect of corruption.

Across training seeds, published unfamiliar acceptance ranges from 4.17–7.99% for InfoNCE and 13.54–16.67% for ConvAP. This is not an isolated favorable InfoNCE run, although three seeds and twelve test scenes remain a small study.

The [score distributions](results/familiarity-diagnostics/score-distributions.png) divide every score by its own frozen threshold, placing each acceptance boundary at 1. The unfamiliar curve is much lower at that boundary for InfoNCE. Ratios are not probabilities, and the plot does not treat scores across seeds as independent samples.

## Mechanisms supported by implementation

Raw pixels compare RGB values directly. Exposure and contrast changes can make familiar images distant, which forces a permissive cutoff when retaining most familiar views. Global brightness/contrast normalization reduces that sensitivity but leaves direct pixel correspondence and remaining sensor effects. Shared walls, floor, and distant landmarks can make different positions look similar; how much each contributes would require masking or scene interventions.

Both neural models learn from repeated visits to the same place and contrasting locations within the same scene. Their objectives can learn which visual changes to tolerate and which distinguish locations. They do not explicitly learn an unfamiliar class: rejection emerges from learned distances and subsequent validation calibration.

InfoNCE compares a positive repeat visit against the other places using a temperature-scaled softmax. The ConvAP adaptation uses mined Multi-Similarity pairs and a different encoder and descriptor size. The resulting InfoNCE scores have stronger separation here, but attributing that to the softmax, mining, pooling, model size, or optimization budget would exceed the experiment's evidence.

## Next causal checks

The first check below is now complete: the [matched-encoder study](loss-ablation/ASSESSMENT.md) finds lower unfamiliar false acceptance for InfoNCE (8.22%) than Multi-Similarity (13.43%) on fresh scenes, with essentially equal familiar acceptance. This supports a loss-recipe effect on the fixed encoder under the common budget; mining and positive-pair handling remain part of that intervention.

1. Keep the encoder, descriptor size, data, batches, and tuning budget fixed while comparing InfoNCE and Multi-Similarity. Use validation for tuning and reserve fresh test scenes. This isolates the loss more closely than the existing pipeline comparison.
2. Capture a distance sweep around remembered places with matched sensor conditions. Measure score and acceptance versus physical displacement, particularly through the currently untested 0.20–0.60 m interval. This tests whether low acceptance is a useful spatial familiarity signal.
3. Mask shared backgrounds or change arena appearance in a controlled capture experiment. This tests the hypothesis that pixel similarity relies excessively on common scene content.

No retraining or new model scoring was performed for this analysis. All published scores, checkpoints, and thresholds remain unchanged. See the [generated diagnostics](results/familiarity-diagnostics/DIAGNOSTICS.md) and [JSON](results/familiarity-diagnostics/diagnostic.json).

Reproduce with a new output directory:

```bash
python scripts/diagnose_level1_familiarity.py \
  --comparison artifacts/level1/evaluation/comparison.json \
  --output artifacts/level1/familiarity-diagnostics
```

NumPy is required; matplotlib is optional for the score plot. The script verifies the published acceptance rates against the saved query scores before deriving diagnostics. Generated Markdown links assume the tracked `docs/level1/results/familiarity-diagnostics/` location.
