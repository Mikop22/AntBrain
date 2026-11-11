# Matched-encoder assessment

Retain **InfoNCE** as the selected Level 1 visual-memory recipe. On fresh test layouts, it rejects unfamiliar observations more reliably than the tested Multi-Similarity recipe on the same encoder, at similar familiar acceptance.

## Result

| Outcome | InfoNCE | Multi-Similarity |
| --- | ---: | ---: |
| Unfamiliar false acceptance | **8.22%** | 13.43% |
| Familiar acceptance | 94.78% | 94.81% |
| Familiarity AUROC | **0.983** | 0.974 |
| Spatial Recall@1 | 96.77% | 96.40% |
| Correct and accepted, familiar queries | 92.56% | 92.71% |

Multi-Similarity adds **5.21 percentage points** of unfamiliar false acceptance. The 95% paired scene interval is **+2.66 to +7.87 points**, conditional on these three training seeds. Familiar acceptance differs by only 0.03 points, with an interval spanning zero. The rejection advantage is therefore not explained by InfoNCE rejecting substantially more familiar views at the chosen operating points. Lower unfamiliar acceptance occurs in every matched seed pair.

The retrieval difference is small: MS minus InfoNCE is -0.38 points, with a scene interval of -0.81 to +0.07. This experiment does not establish a clear retrieval advantage. MS has a slightly higher mean correct-and-accepted rate, so InfoNCE does not lead every metric. The decision follows the predeclared primary unfamiliar-rejection outcome, supported by familiarity AUROC.

## What this resolves

Both arms share the encoder construction and encoding functions, 42,728 parameters, 64D descriptors, initial weights within each seed, training/validation arrays, batches, visit ordering, optimizer, selection rule and 20-epoch budget. This removes architecture and descriptor-size differences from this comparison.

The result supports an effect of the **loss recipe** on this encoder's familiarity scores under the common budget. Multi-Similarity includes its miner and all-repeat positive handling; InfoNCE retains paired symmetric supervision. We have not isolated the formula from those pair/mining choices or performed per-loss tuning. The experiment does not prove that every part of the original camera-aware model's performance gap came from its loss, or that InfoNCE is universally superior.

## Why the numbers differ from the pilot

Test seeds 400–411 are fresh, while the original pilot used 300–311. All training/validation arrays and dtypes are identical. The three retrained InfoNCE runs reproduce every original epoch's loss, exposure and validation metrics exactly; their selected model-state tensors also match the pilot exactly. The new 8.22% false-acceptance result versus the pilot's 6.25% reflects different test samples, not changed InfoNCE weights. Do not compare the two studies as though they used paired test scenes.

## Verification and next use

Three independent reviews passed before any optimizer steps. The final report digest binds checkpoints' selected thresholds; all six checkpoint provenances are checked before fresh-test scoring. Initial states and per-epoch update/exposure counts match within seed. The full suite passed **124 tests**, and lint/format checks passed. A read-only post-evaluation audit independently checked saved-query metrics, aggregation, chronology and artifact hashes.

The next practical experiment remains a displacement sweep followed by controlled local homing, retaining normalized pixels and odometry baselines. Familiar and unfamiliar distance bands, ideal compass alignment, procedural scenes and known scene membership still limit interpretation. This study measures visual memory, not successful homing.

See [full results](results/RESULTS.md), [protocol](PROTOCOL.md), and [reproduction instructions](README.md). The generated `artifacts/level1-loss-ablation/run-bundle.zip` preserves the six checkpoints and full evaluation without the image archive.
