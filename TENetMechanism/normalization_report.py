"""Integrated A-E normalization results with fixed discovery-ranked class pairs."""
import csv

import torch
from common import save_json


CONDITIONS = [
    ("A", "graph_raw", "Raw G"),
    ("B", "graph_normalized", "Normalized (G-mu)/s"),
    ("C", "graph_scale_restored", "Scale restored: s*B"),
    ("D", "graph_mean_restored", "Mean added: B+mu (partial intervention)"),
    ("E", "graph_restored", "Fully restored: s*B+mu"),
]


def write_normalization_report(output, metrics, predictions, labels, discovery, confirmation, names, checks):
    from analyze import pair_changes
    if not checks:
        raise RuntimeError("No raw-matrix reconstruction checks were collected")
    before, after = predictions["graph_raw"], predictions["graph_normalized"]
    ranking = pair_changes(labels, before, after, discovery, len(names))
    pairs = sorted(ranking, key=lambda key: (-ranking[key]["delta_pp"], key))
    summaries, pair_rows = [], []
    for letter, stage, description in CONDITIONS:
        prediction = predictions[stage]
        summaries.append({"condition": letter, "stage": stage, "description": description,
            "accuracy": metrics[stage]["accuracy"],
            "delta_vs_B_pp": metrics[stage]["accuracy"] - metrics["graph_normalized"]["accuracy"],
            "discovery_accuracy": float((prediction[discovery] == labels[discovery]).float().mean() * 100),
            "confirmation_accuracy": float((prediction[confirmation] == labels[confirmation]).float().mean() * 100)})
    for rank, (i, j) in enumerate(pairs, 1):
        for partition, ids in [("discovery", discovery), ("confirmation", confirmation)]:
            selected_labels = labels[ids]
            support = int(((selected_labels == i) | (selected_labels == j)).sum())
            row = {"discovery_rank": rank, "class_a": names[i], "class_b": names[j],
                   "partition": partition, "support": support}
            for letter, stage, _ in CONDITIONS:
                prediction = predictions[stage][ids]
                errors = int((((selected_labels == i) & (prediction == j)) |
                              ((selected_labels == j) & (prediction == i))).sum())
                row[letter + "_errors"] = errors
                row[letter + "_confusion_percent"] = errors * 100 / support
            pair_rows.append(row)
    reconstruction = {"rtol": 1e-5, "atol": 1e-5, "batches_checked": len(checks),
        "all_passed": all(row["passed"] for row in checks),
        "max_abs_error": max(row["max_abs_error"] for row in checks),
        "A_E_prediction_agreement_percent": float((before == predictions["graph_restored"]).float().mean() * 100),
        "batch_checks": checks}
    for filename, rows in [("normalization_summary.csv", summaries), ("normalization_pairs.csv", pair_rows)]:
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    save_json(output / "normalization_report.json", {"conditions": summaries, "reconstruction": reconstruction,
        "class_pairs": pair_rows, "ranking_rule": "A-to-B confusion increase on discovery ONLY"})
    lines = ["# Normalization mechanism study", "",
        "All alternatives are constructed per local graph/channel BEFORE descriptor pooling and hashing.",
        "Network weights are frozen. Each condition fits a separate, equally sized probe on train with the same seed/budget.", "",
        "| Condition | Representation | Accuracy (%) | Delta vs B (pp) | Discovery (%) | Confirmation (%) |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in summaries:
        lines.append(f"| {row['condition']} | {row['description']} | {row['accuracy']:.2f} | "
                     f"{row['delta_vs_B_pp']:+.2f} | {row['discovery_accuracy']:.2f} | {row['confirmation_accuracy']:.2f} |")
    lines += ["", f"Reconstruction checks passed: {reconstruction['all_passed']}; max absolute error: {reconstruction['max_abs_error']:.6g}.",
              f"A/E prediction agreement: {reconstruction['A_E_prediction_agreement_percent']:.2f}%.",
              "A/E matrix equality is tested with tolerances, not bitwise equality; large probe discrepancies require investigation.",
              "", "## Top pairs (fixed A-to-B discovery ranking)", "",
              "| Pair | Partition | Support | A errors | B errors | C errors | D errors | E errors |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in pair_rows:
        if row["discovery_rank"] <= 10:
            lines.append(f"| {row['class_a']} / {row['class_b']} | {row['partition']} | {row['support']} | " +
                         " | ".join(str(row[letter + "_errors"]) for letter in "ABCDE") + " |")
    lines += ["", "C tests scale restoration; D adds mean without restoring scale and is not a reconstruction.",
              "E is an implementation control, not independent evidence that a learned repair helps the network.",
              "Probe improvements measure accessibility and may depend on the descriptor; repeat seeds, grids and dimensions.",
              "Confirmation is exploratory because the network may already have used val for checkpoint selection."]
    (output / "normalization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nNormalization study (accuracy / delta vs B):", flush=True)
    for row in summaries:
        print(f"  {row['condition']} {row['stage']}: {row['accuracy']:.2f}% / {row['delta_vs_B_pp']:+.2f} pp", flush=True)
