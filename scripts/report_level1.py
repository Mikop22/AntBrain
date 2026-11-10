#!/usr/bin/env python3
"""Summarize frozen Level 1 outputs; plotting with matplotlib is optional."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ant_homing.level1.evaluation import paired_scene_interval

LABELS = {
    "raw": "Raw pixels",
    "normalized": "Normalized pixels",
    "infonce": "InfoNCE CNN",
    "convap": "Camera ConvAP + MS",
}
COLORS = ["#a9b3bf", "#5e6c7e", "#347dc1", "#169377"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads((args.evaluation / "comparison.json").read_text())
    training = json.loads((args.training / "trained.json").read_text())
    data = json.loads((args.data / "metadata.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    families = {}
    metrics = [
        "recall_at_1",
        "recall_at_5",
        "place_identity_accuracy",
        "familiarity_auroc",
        "unfamiliar_false_acceptance",
        "familiar_acceptance",
        "correct_and_accepted",
        "median_error_m",
        "p90_error_m",
    ]
    for method, label in LABELS.items():
        runs = [r for r in comparison["results"] if r["method"] == method]
        families[method] = {
            "label": label,
            "runs": len(runs),
            "metrics": {key: float(np.mean([r["metrics"][key] for r in runs])) for key in metrics},
            "recall_seed_sd": float(np.std([r["metrics"]["recall_at_1"] for r in runs], ddof=1))
            if len(runs) > 1
            else None,
            "latency_ms": float(np.mean([r["observation_latency"]["median_ms"] for r in runs])),
            "parameter_count": runs[0]["parameter_count"],
            "model_state_bytes": runs[0]["model_state_bytes"],
            "descriptor_bytes": comparison["cache_bytes_per_reference"][method],
            "conditions": {
                condition: {
                    key: float(np.mean([r["conditions"][condition][key] for r in runs]))
                    for key in metrics
                }
                for condition in runs[0]["conditions"]
            },
            "scenes": {
                int(scene): float(np.mean([r["scenes"][scene]["recall_at_1"] for r in runs]))
                for scene in runs[0]["scenes"]
            },
            "selected_epochs": [
                r["selected_epoch"] for r in training["runs"] if r["method"] == method
            ],
        }
    paired = dict(comparison["paired_recall_vs_normalized"])
    paired["convap_vs_infonce"] = paired_scene_interval(
        families["convap"]["scenes"], families["infonce"]["scenes"]
    )
    summary = {
        "families": families,
        "paired": paired,
        "counts": data["counts"],
        "dataset_sha256": comparison["dataset_sha256"],
        "metadata_sha256": training["metadata_sha256"],
        "review_manifest_sha256": training["review_manifest_sha256"],
        "reviewed_training_start_utc": training["started_at_utc"],
        "frozen_at_utc": training["frozen_at_utc"],
        "epochs": training["epochs"],
        "seeds": training["seeds"],
        "provenance": comparison["provenance"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Level 1 held-out comparison",
        "",
        "Three approach groups were built independently and reviewed by three independent agents before optimization. Raw and normalized pixels are two variants of the first group.",
        "",
        "## Results",
        "",
        "| Approach | Recall@1 | Recall@5 | Familiarity AUROC | Novel false acceptance | Known acceptance | CPU encoding | Model weights | Cache/reference |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family in families.values():
        m = family["metrics"]
        sd = family["recall_seed_sd"]
        recall = f"{100 * m['recall_at_1']:.2f}%" + (f" ± {100 * sd:.2f}" if sd is not None else "")
        lines.append(
            f"| {family['label']} | {recall} | {100 * m['recall_at_5']:.2f}% | {m['familiarity_auroc']:.3f} | {100 * m['unfamiliar_false_acceptance']:.2f}% | {100 * m['familiar_acceptance']:.2f}% | {family['latency_ms']:.3f} ms | {family['model_state_bytes'] / 1024:.1f} KiB | {family['descriptor_bytes']:,} B |"
        )
    lines += [
        "",
        f"Neural entries average seeds {training['seeds']}; ± is sample standard deviation across training runs. CPU encoding is the mean of per-run median sequential-observation encoding latency after warm-up; rendering and gallery search are excluded. Weight bytes count the in-memory state, not checkpoint-file overhead. Pixel cache entries use float32; their original uint8 RGB image requires 12,288 bytes instead. Cached-feature memory and source-image storage are different implementation choices. These counts exclude Python/PyTorch runtime overhead and are not process RSS.",
        "",
        "Novel false acceptance and known acceptance use thresholds calibrated to target 95% familiar acceptance on validation data; test acceptance is measured rather than assumed. Spatial recall uses a 0.25 m tolerance, and unknown queries are excluded from retrieval recall.",
        "",
        "## Localization",
        "",
        "| Approach | Exact place accuracy | Correct and accepted | Median spatial error | 90th-percentile error |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for family in families.values():
        m = family["metrics"]
        lines.append(
            f"| {family['label']} | {100 * m['place_identity_accuracy']:.2f}% | {100 * m['correct_and_accepted']:.2f}% | {m['median_error_m']:.3f} m | {m['p90_error_m']:.3f} m |"
        )
    lines += [
        "",
        "Correct and accepted is the fraction of all familiar test queries whose top match is within 0.25 m and passes the frozen familiarity threshold. Acceptance alone does not guarantee a correct match. Neural error quantiles are averaged across runs, rather than pooled across repeated predictions.",
        "",
        "## By condition",
        "",
        "| Condition | Raw | Normalized | InfoNCE | ConvAP + MS |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for condition in families["raw"]["conditions"]:
        values = [
            f"{100 * families[method]['conditions'][condition]['recall_at_1']:.2f}%"
            for method in LABELS
        ]
        lines.append(f"| {condition} | " + " | ".join(values) + " |")
    lines += [
        "",
        "Conditions use the same declared nuisance distributions in training and testing; combined is not an out-of-distribution claim.",
        "",
        "## Paired uncertainty",
        "",
        "| Comparison | Recall@1 difference | 95% paired scene interval |",
        "| --- | ---: | ---: |",
    ]
    names = {
        "infonce": "InfoNCE − normalized",
        "convap": "ConvAP − normalized",
        "convap_vs_infonce": "ConvAP − InfoNCE",
    }
    for key, value in paired.items():
        lo, hi = value["ci95"]
        lines.append(
            f"| {names[key]} | {100 * value['mean_difference']:+.2f} pp | [{100 * lo:+.2f}, {100 * hi:+.2f}] pp |"
        )
    lines += [
        "",
        f"Intervals resample {len(comparison['test_scene_ids'])} complete paired test scenes, conditional on these training seeds. Correlated individual frames are not bootstrap units. This is a single procedural data distribution, not evidence of real-world superiority.",
        "",
        "## Protocol and reproducibility",
        "",
        f"- Disjoint scenes: {data['counts']['train']['scenes']} training, {data['counts']['val']['scenes']} validation, {data['counts']['test']['scenes']} test. Test set: {data['counts']['test']['gallery']} references, {data['counts']['test']['familiar_queries']} familiar and {data['counts']['test']['unfamiliar_queries']} unfamiliar queries.",
        f"- Both neural approaches: {training['epochs']} epochs, Adam learning rate {training['learning_rate']}, identical batches and image exposure, validation-only checkpoint selection. Four views of each place are independently captured; four camera directions form one observation.",
        "- Review approvals bind implementation, array and metadata hashes. The training start follows the three approvals. Test scoring follows frozen checkpoints and validation thresholds.",
        "- The comparison changes architecture, loss and descriptor size together. It does not isolate a causal advantage of Multi-Similarity or ConvAP pooling, and uses a common budget rather than per-model tuning.",
        "- Ideal compass, fixed wall appearance, small procedural scenes, synthetic exposure/occlusion and known scene membership limit generalization. Retrieval and familiarity are not homing outcomes.",
        "",
        f"Dataset SHA256: `{comparison['dataset_sha256']}`.",
        "",
        "See [assessment and recommendation](../ASSESSMENT.md), [protocol](../PROTOCOL.md), [pre-training reviews](../reviews/), and [review manifest](../review-manifest.json). Full per-query results and trained checkpoints remain under `artifacts/level1/`.",
    ]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        lines += [
            "",
            "Plot generation requires optional matplotlib; tabulated results above are complete.",
        ]
    else:
        methods = list(LABELS)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        values = [100 * families[m]["metrics"]["recall_at_1"] for m in methods]
        errors = [100 * (families[m]["recall_seed_sd"] or 0) for m in methods]
        axes[0].bar(range(4), values, yerr=errors, capsize=4, color=COLORS)
        axes[0].set_ylim(0, 105)
        axes[0].set_ylabel("Spatial Recall@1 (%)")
        axes[0].set_xticks(range(4), ["Raw", "Normalized", "InfoNCE", "ConvAP+MS"])
        axes[0].set_title("Held-out retrieval; bars ± training-seed SD")
        for i, value in enumerate(values):
            axes[0].text(i, value + errors[i] + 1.5, f"{value:.1f}", ha="center", fontsize=9)
        conditions = list(families["raw"]["conditions"])
        x = np.arange(len(conditions))
        for i, method in enumerate(methods):
            axes[1].bar(
                x + (i - 1.5) * 0.2,
                [100 * families[method]["conditions"][c]["recall_at_1"] for c in conditions],
                width=0.2,
                color=COLORS[i],
                label=LABELS[method],
            )
        axes[1].set_ylim(0, 105)
        axes[1].set_xticks(x, conditions)
        axes[1].set_title("Appearance conditions")
        axes[1].legend(fontsize=8, loc="lower left")
        fig.savefig(args.output / "retrieval.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        x = np.arange(len(methods))
        for offset, metric, label, color in (
            (-0.2, "familiar_acceptance", "Familiar accepted (higher is better)", "#347dc1"),
            (
                0.2,
                "unfamiliar_false_acceptance",
                "Unfamiliar falsely accepted (lower is better)",
                "#c35b46",
            ),
        ):
            values = [100 * families[m]["metrics"][metric] for m in methods]
            ax.bar(x + offset, values, width=0.4, label=label, color=color)
            for i, value in enumerate(values):
                ax.text(i + offset, value + 1, f"{value:.1f}", ha="center", fontsize=9)
        ax.set(
            ylim=(0, 115),
            ylabel="Test queries (%)",
            title="Acceptance at validation-calibrated thresholds",
        )
        ax.set_xticks(x, ["Raw", "Normalized", "InfoNCE", "ConvAP+MS"])
        ax.legend(fontsize=8, loc="upper center", ncol=2)
        fig.savefig(args.output / "familiarity.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for method, color in (("infonce", COLORS[2]), ("convap", COLORS[3])):
            runs = [r for r in training["runs"] if r["method"] == method]
            curves = np.array(
                [[100 * e["validation"]["recall_at_1"] for e in r["history"]] for r in runs]
            )
            epochs = np.arange(1, curves.shape[1] + 1)
            ax.plot(epochs, curves.mean(0), label=LABELS[method], color=color)
            ax.fill_between(epochs, curves.min(0), curves.max(0), color=color, alpha=0.15)
        ax.set(
            xlabel="Epoch",
            ylabel="Validation Recall@1 (%)",
            title="Validation learning curves; shaded seed range",
        )
        ax.legend()
        fig.savefig(args.output / "training.png", dpi=180)
        plt.close(fig)
        lines += [
            "",
            "![Held-out retrieval](retrieval.png)",
            "",
            "![Familiarity and false acceptance](familiarity.png)",
            "",
            "![Validation curves](training.png)",
        ]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({m: families[m]["metrics"] for m in families}, indent=2))


if __name__ == "__main__":
    main()
