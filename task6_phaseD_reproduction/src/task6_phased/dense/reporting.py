from __future__ import annotations

import csv
import json

from task6_phased.common.config import TASKS, dense_root, dense_run_path
from task6_phased.common.io import read_json


def dense_results(config, run_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    evaluation = dense_run_path(config, "evaluation", run_id)
    selection = read_json(evaluation / "selection.json")
    rows = []
    for state in selection["candidates"]:
        data = read_json(evaluation / f"{state['name']}.json")
        row = {
            "step": state["step"], "macro": data["summary"]["macro"],
            "worst_domain": data["summary"]["worst_domain"],
            "per_domain_exposure": state["step"] * int(config["dense"]["domain_batch_size"]),
            "role": "best" if state["step"] == selection["state"]["step"] else (
                "final" if state["step"] == config["dense"]["total_optimizer_steps"] else "trajectory"
            ),
        }
        for task in TASKS:
            metrics = data[task]["metrics"]
            row[f"{task}_native"] = metrics["native"]
            row[f"{task}_greedy_exact"] = metrics["greedy_exact"]
            row[f"{task}_greedy_invalid_rate"] = metrics["greedy_invalid_rate"]
        rows.append(row)
    root = dense_root(config) / "results" / run_id
    data_dir, table_dir = root / "data", root / "tables"
    figure_dir, diagnostic_dir = root / "figures/main", root / "figures/diagnostics"
    for path in (data_dir, table_dir, figure_dir, diagnostic_dir):
        path.mkdir(parents=True, exist_ok=True)
    with (data_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best_step = selection["state"]["step"]
    lines = ["# Dense-MT trajectory", "", f"Selected macro-best: step {best_step}", "",
             "| step | role | exposure/domain | SST-2 | MNLI | QNLI | QQP | macro | worst | greedy exact (4 domains) | invalid rate (4 domains) |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in rows:
        greedy = "/".join(f"{100*row[f'{task}_greedy_exact']:.2f}" for task in TASKS)
        invalid = "/".join(f"{100*row[f'{task}_greedy_invalid_rate']:.2f}" for task in TASKS)
        lines.append("| " + " | ".join([
            str(row["step"]), row["role"], str(row["per_domain_exposure"]),
            *(f"{100*row[f'{task}_native']:.4f}" for task in TASKS),
            f"{100*row['macro']:.4f}", f"{100*row['worst_domain']:.4f}", greedy, invalid,
        ]) + " |")
    (table_dir / "dense_mt_trajectory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, axis = plt.subplots(figsize=(8.5, 5))
    for task in TASKS:
        axis.plot([r["step"] for r in rows], [100*r[f"{task}_native"] for r in rows], marker="o", label=task.upper())
    axis.plot([r["step"] for r in rows], [100*r["macro"] for r in rows], color="black", linewidth=2, label="Macro")
    axis.axvline(best_step, color="grey", linestyle=":", label="Macro-best")
    axis.set(xlabel="Optimizer step", ylabel="Native score ×100", title="Dense-MT four-domain validation trajectory")
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "dense_mt_domain_and_macro.pdf")
    fig.savefig(figure_dir / "dense_mt_domain_and_macro.png", dpi=config["metrics"]["png_dpi"])
    plt.close(fig)

    telemetry_path = dense_run_path(config, "train", run_id) / "telemetry.jsonl"
    telemetry = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not telemetry:
        raise ValueError("Dense training telemetry is empty")
    telemetry_fields = [
        "step", "loss", "lr", "grad_norm",
        *(f"{task}_loss" for task in TASKS),
        *(f"{task}_exposure" for task in TASKS),
        *(f"{task}_cycles" for task in TASKS),
    ]
    telemetry_rows = []
    for item in telemetry:
        telemetry_rows.append({
            "step": item["step"], "loss": item["loss"], "lr": item["lr"],
            "grad_norm": item["grad_norm"],
            **{f"{task}_loss": item["domain_losses"][task] for task in TASKS},
            **{f"{task}_exposure": item["domain_exposure"][task] for task in TASKS},
            **{f"{task}_cycles": item["cycles"][task] for task in TASKS},
        })
    with (data_dir / "training_telemetry.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=telemetry_fields)
        writer.writeheader()
        writer.writerows(telemetry_rows)
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    steps = [row["step"] for row in telemetry_rows]
    axes[0].plot(steps, [row["loss"] for row in telemetry_rows], color="black", linewidth=1, label="equal-domain loss")
    for task in TASKS:
        axes[0].plot(steps, [row[f"{task}_loss"] for row in telemetry_rows], linewidth=.8, alpha=.7, label=task.upper())
    axes[0].set(ylabel="Loss", title="Dense-MT optimization diagnostics")
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].plot(steps, [row["lr"] for row in telemetry_rows], color="#4c78a8", label="LR")
    axes[1].set(ylabel="Learning rate")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for task in TASKS:
        axes[2].plot(steps, [row[f"{task}_exposure"] for row in telemetry_rows], label=task.upper())
    axes[2].set(xlabel="Optimizer step", ylabel="Samples seen per domain")
    axes[2].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(diagnostic_dir / "dense_mt_loss_lr_exposure.pdf")
    fig.savefig(diagnostic_dir / "dense_mt_loss_lr_exposure.png", dpi=config["metrics"]["png_dpi"])
    plt.close(fig)
    return root
