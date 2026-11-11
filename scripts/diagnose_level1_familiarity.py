#!/usr/bin/env python3
"""Explain frozen familiarity scores without fitting or changing thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

LABELS = {
    "raw": "Raw pixels",
    "normalized": "Normalized pixels",
    "infonce": "InfoNCE",
    "convap": "ConvAP + MS",
}


def diagnostic(run: dict) -> dict:
    rows = run["rows"]
    threshold = run["metrics"]["threshold"]
    known = np.array([r["distance"] for r in rows if r["familiar"]])
    novel = np.array([r["distance"] for r in rows if not r["familiar"]])
    known_acceptance = float(np.mean(known <= threshold))
    novel_acceptance = float(np.mean(novel <= threshold))
    # Recompute published operating points before deriving new diagnostics.
    assert np.isclose(known_acceptance, run["metrics"]["familiar_acceptance"])
    assert np.isclose(novel_acceptance, run["metrics"]["unfamiliar_false_acceptance"])
    # This is a descriptive test ROC point, never a deployable calibration.
    roc_threshold = np.quantile(known, 0.95, method="higher")
    return {
        "seed": run["seed"],
        "frozen_threshold": threshold,
        "familiar_acceptance": known_acceptance,
        "unfamiliar_false_acceptance": novel_acceptance,
        "test_roc_fpr_at_95pct_tpr": float(np.mean(novel <= roc_threshold)),
        "test_roc_achieved_tpr": float(np.mean(known <= roc_threshold)),
        "familiarity_auroc": run["metrics"]["familiarity_auroc"],
        "clean_auroc": run["conditions"]["clean"]["familiarity_auroc"],
        "median_novel_distance_over_threshold": float(np.median(novel / threshold)),
        "conditions": {
            c: {
                k: metrics[k]
                for k in ("familiar_acceptance", "unfamiliar_false_acceptance", "familiarity_auroc")
            }
            for c, metrics in run["conditions"].items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    groups = {}
    metrics = (
        "familiar_acceptance",
        "unfamiliar_false_acceptance",
        "test_roc_fpr_at_95pct_tpr",
        "test_roc_achieved_tpr",
        "familiarity_auroc",
        "clean_auroc",
        "median_novel_distance_over_threshold",
    )
    for method in LABELS:
        runs = [diagnostic(r) for r in comparison["results"] if r["method"] == method]
        groups[method] = {
            "runs": runs,
            "mean": {k: float(np.mean([r[k] for r in runs])) for k in metrics},
            "conditions": {
                c: {
                    k: float(np.mean([r["conditions"][c][k] for r in runs]))
                    for k in runs[0]["conditions"][c]
                }
                for c in runs[0]["conditions"]
            },
        }
    summary = {
        "comparison_sha256": hashlib.sha256(args.comparison.read_bytes()).hexdigest(),
        "scope": "Post-hoc description of saved test scores; no fitting, model scoring, or deployed threshold changes.",
        "roc_warning": "Test-derived ROC thresholds are diagnostic only. They are not validation-calibrated operating results or deployable settings.",
        "aggregation": "Equal mean over training seeds; baseline has one run. Distance ratios and medians are per-run before averaging.",
        "families": groups,
    }
    (args.output / "diagnostic.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Familiarity score diagnostics",
        "",
        "Post-hoc analysis of the completed pilot's saved test scores. No models were fitted or scored again, and no deployed thresholds changed. Neural numbers average three training seeds.",
        "",
        "| Method | Familiar accepted | Unfamiliar falsely accepted | Diagnostic FPR at 95% test TPR | AUROC | Clean-only AUROC | Median unfamiliar score / threshold |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, label in LABELS.items():
        m = groups[method]["mean"]
        lines.append(
            f"| {label} | {100 * m['familiar_acceptance']:.2f}% | {100 * m['unfamiliar_false_acceptance']:.2f}% | {100 * m['test_roc_fpr_at_95pct_tpr']:.2f}% | {m['familiarity_auroc']:.3f} | {m['clean_auroc']:.3f} | {m['median_novel_distance_over_threshold']:.2f} |"
        )
    lines += [
        "",
        "The diagnostic ROC column uses test labels to describe a common operating point (at least 95% familiar acceptance, subject to sample discreteness). It is not a new calibrated threshold, an independent experiment, or a deployment recommendation. Published operating points remain unchanged. The gap persists when acceptance is equalized.",
        "",
        "Each method has its own distance units. Dividing scores by that run's frozen threshold puts its acceptance boundary at 1; ratios are not calibrated probabilities. AUROC measures separation across thresholds, so different numerical distance scales alone cannot explain its gap.",
        "",
        "| Condition | Raw false acceptance | Normalized | InfoNCE | ConvAP + MS |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for c in groups["raw"]["conditions"]:
        values = [
            f"{100 * groups[m]['conditions'][c]['unfamiliar_false_acceptance']:.2f}%"
            for m in LABELS
        ]
        lines.append(f"| {c} | " + " | ".join(values) + " |")
    lines += [
        "",
        "Each condition contains 576 familiar and 72 unfamiliar test queries. Conditions use different query positions; condition differences do not isolate the causal effect of a perturbation. The clean-only AUROC gap shows that synthetic corruption is not the entire explanation.",
        "",
        "See the [interpretation](../../FAMILIARITY.md).",
    ]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        pass
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for ax, (method, label) in zip(axes.flat, LABELS.items(), strict=True):
            runs = [r for r in comparison["results"] if r["method"] == method]
            for familiar, name, color in (
                (True, "Familiar", "#347dc1"),
                (False, "Unfamiliar", "#c35b46"),
            ):
                scores = np.sort(
                    [
                        q["distance"] / r["metrics"]["threshold"]
                        for r in runs
                        for q in r["rows"]
                        if q["familiar"] == familiar
                    ]
                )
                ax.step(
                    scores,
                    np.arange(1, len(scores) + 1) / len(scores),
                    where="post",
                    label=name,
                    color=color,
                )
            ax.axvline(1, color="#555555", linestyle="--", linewidth=1)
            ax.set(
                title=label,
                xlim=(0, 3),
                ylim=(0, 1.03),
                xlabel="Distance / frozen threshold",
                ylabel="Fraction at or below score",
            )
            ax.legend(loc="lower right", fontsize=8)
        fig.suptitle("Acceptance boundary = 1; unfamiliar curve should stay low there")
        fig.savefig(args.output / "score-distributions.png", dpi=170)
        plt.close(fig)
        lines += [
            "",
            "![Score distributions](score-distributions.png)",
            "",
            "Curves are empirical cumulative distributions. Neural curves equally mix runs after each run's threshold normalization; repeated scores across runs are not independent samples. The displayed score range is 0–3; larger scores are outside the panel.",
        ]
    (args.output / "DIAGNOSTICS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({m: g["mean"] for m, g in groups.items()}, indent=2))


if __name__ == "__main__":
    main()
