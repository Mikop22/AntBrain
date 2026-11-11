#!/usr/bin/env python3
"""Summarize the frozen matched-encoder experiment; optional standalone plots."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

LABELS = {"infonce": "InfoNCE", "infonce_ms": "Multi-Similarity"}
METRICS = (
    "recall_at_1",
    "recall_at_5",
    "place_identity_accuracy",
    "familiarity_auroc",
    "familiar_acceptance",
    "unfamiliar_false_acceptance",
    "correct_and_accepted",
    "median_error_m",
    "p90_error_m",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads((args.evaluation / "comparison.json").read_text())
    training = json.loads((args.run_dir / "trained.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    families = {}
    for method, label in LABELS.items():
        runs = [r for r in comparison["results"] if r["method"] == method]
        trained = [r for r in training["runs"] if r["method"] == method]
        families[method] = {
            "label": label,
            "means": {k: float(np.mean([r["metrics"][k] for r in runs])) for k in METRICS},
            "seed_sd": {k: float(np.std([r["metrics"][k] for r in runs], ddof=1)) for k in METRICS},
            "selected_epochs": [r["selected_epoch"] for r in trained],
            "conditions": {
                c: {k: float(np.mean([r["conditions"][c][k] for r in runs])) for k in METRICS}
                for c in runs[0]["conditions"]
            },
        }
    summary = {
        "families": families,
        "paired_ms_minus_infonce": comparison["paired_ms_minus_infonce"],
        "dataset_sha256": comparison["dataset_sha256"],
        "review_manifest_sha256": comparison["review_manifest_sha256"],
        "training_started_at_utc": training["started_at_utc"],
        "frozen_at_utc": training["frozen_at_utc"],
        "evaluated_at_utc": comparison["evaluated_at_utc"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Matched-encoder loss results",
        "",
        "Both arms use the same 42,728-parameter CNN and 64D normalized descriptors (256 bytes/reference), identical initialization within seed, training/validation data, batches, optimizer and 20-epoch budget. Only the loss recipe changes. All three independent reviews passed before optimization; six checkpoints and validation thresholds were frozen before fresh-test scoring.",
        "",
        "| Loss | Recall@1 | Recall@5 | Familiarity AUROC | Unfamiliar false acceptance | Familiar acceptance | Correct and accepted |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for f in families.values():
        m, sd = f["means"], f["seed_sd"]
        lines.append(
            f"| {f['label']} | {100 * m['recall_at_1']:.2f}% ± {100 * sd['recall_at_1']:.2f} | {100 * m['recall_at_5']:.2f}% | {m['familiarity_auroc']:.3f} | {100 * m['unfamiliar_false_acceptance']:.2f}% ± {100 * sd['unfamiliar_false_acceptance']:.2f} | {100 * m['familiar_acceptance']:.2f}% | {100 * m['correct_and_accepted']:.2f}% |"
        )
    lines += [
        "",
        "Means are over seeds 0, 1, 2; ± is sample SD in percentage points, not a confidence interval. Thresholds target 95% familiar acceptance on validation; actual test acceptance is measured. Retrieval uses 0.25 m tolerance and excludes unfamiliar queries. Correct-and-accepted is the fraction of familiar queries both correctly localized and accepted.",
        "",
        "## Localization",
        "",
        "| Loss | Exact place accuracy | Median error | 90th-percentile error |",
        "| --- | ---: | ---: | ---: |",
    ]
    for f in families.values():
        m = f["means"]
        lines.append(
            f"| {f['label']} | {100 * m['place_identity_accuracy']:.2f}% | {m['median_error_m']:.3f} m | {m['p90_error_m']:.3f} m |"
        )
    lines += [
        "",
        "Error quantiles are computed per run and then averaged; they are not pooled across repeated seed predictions.",
        "",
        "## Individual runs",
        "",
        "| Loss | Seed | Selected epoch | Recall@1 | AUROC | Unfamiliar false acceptance | Familiar acceptance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in comparison["results"]:
        t = next(
            t for t in training["runs"] if (t["method"], t["seed"]) == (r["method"], r["seed"])
        )
        m = r["metrics"]
        lines.append(
            f"| {LABELS[r['method']]} | {r['seed']} | {t['selected_epoch']} | {100 * m['recall_at_1']:.2f}% | {m['familiarity_auroc']:.3f} | {100 * m['unfamiliar_false_acceptance']:.2f}% | {100 * m['familiar_acceptance']:.2f}% |"
        )
    lines += [
        "",
        "## Paired scene differences",
        "",
        "| Metric | MS minus InfoNCE | 95% paired scene interval |",
        "| --- | ---: | ---: |",
    ]
    for key, value in comparison["paired_ms_minus_infonce"].items():
        lo, hi = value["ci95"]
        scale = 1 if key == "familiarity_auroc" else 100
        unit = "" if scale == 1 else " pp"
        label = "Mean scene AUROC" if key == "familiarity_auroc" else key.replace("_", " ")
        lines.append(
            f"| {label} | {scale * value['mean_difference']:+.4f}{unit} | [{scale * lo:+.4f}, {scale * hi:+.4f}]{unit} |"
        )
    lines += [
        "",
        comparison["interval_scope"]
        + ". AUROC in the first table is pooled across queries within each run; the paired AUROC interval above concerns scene means. These are different estimands.",
        "",
        "## Conditions",
        "",
        "| Condition | InfoNCE false acceptance | MS false acceptance | InfoNCE Recall@1 | MS Recall@1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for c in families["infonce"]["conditions"]:
        a, b = [families[m]["conditions"][c] for m in LABELS]
        lines.append(
            f"| {c} | {100 * a['unfamiliar_false_acceptance']:.2f}% | {100 * b['unfamiliar_false_acceptance']:.2f}% | {100 * a['recall_at_1']:.2f}% | {100 * b['recall_at_1']:.2f}% |"
        )
    lines += [
        "",
        "## Scope",
        "",
        "Training and validation arrays match the pilot exactly. Test seeds 400–411 are fresh: 12 scenes, 576 references, 2,304 familiar queries, 288 unfamiliar queries. Both arms see the same archive. Ideal compass alignment, known scene membership, one procedural scene family and synthetic sensor conditions limit generalization. Homing is not evaluated.",
        "",
        "This compares the original paired InfoNCE recipe with mined Multi-Similarity on the same encoder. Positive-pair handling and mining belong to the changed recipe. There is no per-loss tuning; results describe this common optimization budget, not the best attainable performance of either loss. No test-based retuning was performed.",
        "",
        "See [protocol](../PROTOCOL.md), [pre-training reviews](../reviews/), and [assessment](../ASSESSMENT.md). Full query results and weights are generated artifacts; tracked fingerprints identify them.",
    ]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        pass
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for method, color in zip(LABELS, ("#347dc1", "#169377"), strict=True):
            runs = [r for r in training["runs"] if r["method"] == method]
            for ax, metric, title in zip(
                axes,
                ("recall_at_1", "familiarity_auroc"),
                ("Validation Recall@1", "Validation familiarity AUROC"),
                strict=True,
            ):
                curves = np.array([[e["validation"][metric] for e in r["history"]] for r in runs])
                epochs = np.arange(1, 21)
                ax.plot(epochs, curves.mean(0), label=LABELS[method], color=color)
                ax.fill_between(epochs, curves.min(0), curves.max(0), color=color, alpha=0.15)
                ax.set(xlabel="Epoch", ylabel=title, title=title + "; shaded seed range")
                ax.legend()
        fig.savefig(args.output / "validation.png", dpi=180)
        plt.close(fig)
        lines += ["", "![Validation curves](validation.png)"]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    shutil.copyfile(args.data / "metadata.json", args.output / "data-metadata.json")
    artifacts = [args.data / n for n in ("images.npz", "metadata.json", "identity-check.json")]
    artifacts += [p for p in args.run_dir.rglob("*") if p.is_file()]
    artifacts += [args.evaluation / "comparison.json"]
    root = Path(__file__).resolve().parents[1]
    manifest = {
        str(p.resolve().relative_to(root)): {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }
        for p in artifacts
    }
    (args.output / "artifacts.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
