from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import json

from task6.common.config import protocol_id, root_for
from task6.common.io import read_json

COLORS = {
    "dense-ft": "#111111",
    "R2": "#777777", "R2-soft": "#8c564b", "R4o": "#d62728",
    "R4d": "#9467bd", "R4o-hard": "#e377c2", "G1": "#1f77b4",
    "G2": "#2ca02c", "G4": "#17becf",
}
STYLES = {"aux_0.001": ":"}
ARM_STYLES = {"R2-soft": "--", "R4d": "--", "R4o-hard": "-."}
ARM_MARKERS = {"R2-soft": "^", "R4d": "s", "R4o-hard": "D"}
DISPLAY = ["accuracy", "relative_performance", "cv", "churn", "exact_set_change",
           "oracle_overlap", "activation_coverage"]
PROPORTIONS = {"accuracy", "invalid_rate", "churn", "exact_set_change",
               "oracle_overlap", "activation_coverage"}


def dense_fullft_reference(rows, task):
    matches = [row for row in rows if row["task"] == task and row["arm"] == "dense-ft"
               and row["role"] == "best" and row["layer"] == "model"
               and row["group"] == "performance" and row["metric"] == "relative_performance"]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one aggregated Dense-fullFT best row for {task}")
    if matches[0]["mean"] is None:
        raise ValueError(f"Dense-fullFT relative performance is undefined for {task}")
    return matches[0]


def data_file(config, run_id, section, name="metrics.json", *, result_root=None):
    root = Path(result_root) if result_root is not None else root_for(config) / "results"
    result = read_json(root / "data" / section / run_id / name)
    if result["meta"]["protocol"] != protocol_id(config):
        raise ValueError("Results were produced with a different protocol")
    return result


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot render an empty table: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()} for row in rows)


def formatted(value, metric, unit=None):
    if value is None:
        return "NA"
    if metric in PROPORTIONS:
        return f"{value * 100:.2f}%"
    if metric == "relative_performance" or unit in ("percentage_points", "relative_percentage_points"):
        return f"{value:.2f}"
    return f"{value:.3e}" if value != 0 and abs(value) < 1e-4 else f"{value:.4f}"


def _write_main_table(root, run_id, aggregated):
    main = [r for r in aggregated if r["layer"] == "model" and r["role"] in ("best", "static") and r["metric"] in DISPLAY]
    save_csv(root / "main" / run_id / "best_validation.csv", main)
    report = root / "main" / run_id / "best_validation.md"
    with report.open("w", encoding="utf-8") as stream:
        stream.write("# Best-validation results\n\nSame validation set selects checkpoints and reports scores; not independent test performance.\n\n")
        stream.write("| task | arm | variant | k | metric | mean ± seed std |\n|---|---|---|---:|---|---|\n")
        for row in main:
            mean = formatted(row["mean"], row["metric"])
            std = "deterministic" if row["deterministic"] else formatted(row["std"], row["metric"])
            stream.write(f"| {row['task']} | {row['display_name']} | {row['variant']} | {row['k']} | {row['group']}/{row['metric']} | {mean} ± {std} |\n")


def main_table(config, run_id, *, result_root=None):
    aggregated = data_file(config, run_id, "aggregated", result_root=result_root)["rows"]
    root = (Path(result_root) if result_root is not None else root_for(config) / "results") / "tables"
    _write_main_table(root, run_id, aggregated)


def tables(config, run_id, *, result_root=None):
    aggregated = data_file(config, run_id, "aggregated", result_root=result_root)["rows"]
    normalized = data_file(config, run_id, "normalized", result_root=result_root)["rows"]
    paired = data_file(config, run_id, "paired_differences", result_root=result_root)["rows"]
    root = (Path(result_root) if result_root is not None else root_for(config) / "results") / "tables"
    _write_main_table(root, run_id, aggregated)
    save_csv(root / "diagnostics" / run_id / "trajectories_and_final.csv", [r for r in aggregated if r["role"] in ("trajectory", "final")])
    save_csv(root / "diagnostics" / run_id / "paired_differences.csv", paired)
    save_csv(root / "appendix" / run_id / "per_layer_per_seed.csv", normalized)


def figures(config, run_id, *, result_root=None, metrics=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    rows = data_file(config, run_id, "aggregated", result_root=result_root)["rows"]
    root = (Path(result_root) if result_root is not None else root_for(config) / "results") / "figures"
    displayed = set(DISPLAY if metrics is None else metrics)
    if not displayed <= set(DISPLAY):
        raise ValueError(f"Unsupported figure metric filter: {sorted(displayed - set(DISPLAY))}")

    def finish(fig, folder, name):
        path = root / folder / run_id / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path.with_suffix(".pdf"))
        fig.savefig(path.with_suffix(".png"), dpi=config["metrics"]["png_dpi"])
        plt.close(fig)

    selected = [r for r in rows if r["layer"] == "model" and r["role"] in ("best", "static")
                and r["arm"] not in ("dense", "dense-ft") and r["metric"] in displayed]
    grouped = defaultdict(list)
    for row in selected:
        grouped[(row["task"], row["group"], row["metric"])].append(row)
    for (task, group, metric), points in grouped.items():
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for arm, variant in sorted({(r["arm"], r["variant"]) for r in points}):
            data = sorted([r for r in points if (r["arm"], r["variant"]) == (arm, variant)], key=lambda r: r["k"])
            factor = 100 if metric in PROPORTIONS else 1
            x = [100 * r["k"] / config["model"]["num_experts"] for r in data]
            y = [np.nan if r["mean"] is None else factor * r["mean"] for r in data]
            error = [np.nan if r["std"] is None else factor * r["std"] for r in data]
            label = arm if variant == "default" else f"{arm} {variant}"
            ax.errorbar(x, y, yerr=error if any(r["std"] is not None for r in data) else None,
                        label=label, color=COLORS[arm], linestyle=ARM_STYLES.get(arm, STYLES.get(variant, "-")),
                        marker=ARM_MARKERS.get(arm, "o"), capsize=2)
        if metric == "relative_performance":
            ax.axhline(100, color="#555555", linestyle=":", label="Dense-init (100%)")
            baseline = dense_fullft_reference(rows, task)
            ax.axhline(baseline["mean"], color=COLORS["dense-ft"], linestyle="--",
                       label="Dense-fullFT best")
            if baseline["std"] is not None:
                ax.axhspan(baseline["mean"] - baseline["std"], baseline["mean"] + baseline["std"],
                           color=COLORS["dense-ft"], alpha=0.08, label="Dense-fullFT ±1 seed SD")
        if metric == "oracle_overlap":
            x = np.asarray(config["suite"]["top_k"]) / config["model"]["num_experts"] * 100
            ax.plot(x, x, color="black", linestyle=":", label="uniform-random expectation")
        percent_metric = metric in PROPORTIONS or metric == "relative_performance"
        ax.set(xlabel="Selected experts (%)", ylabel=metric + (" (%)" if percent_metric else ""),
               title=f"{task} / {group} / best-validation")
        ax.legend(fontsize=7, ncol=2)
        finish(fig, "main" if metric == "relative_performance" else "diagnostics", f"{task}_{group}_{metric}")

    trajectories = defaultdict(list)
    for row in rows:
        if row["layer"] == "model" and row["role"] == "trajectory" and row["metric"] in displayed:
            trajectories[(row["task"], row["k"], row["group"], row["metric"])].append(row)
    for (task, k, group, metric), points in trajectories.items():
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for arm, variant in sorted({(r["arm"], r["variant"]) for r in points}):
            data = sorted([r for r in points if (r["arm"], r["variant"]) == (arm, variant)], key=lambda r: r["epoch"])
            factor = 100 if metric in PROPORTIONS else 1
            x = [r["epoch"] for r in data]
            y = np.asarray([np.nan if r["mean"] is None else factor * r["mean"] for r in data])
            std = np.asarray([np.nan if r["std"] is None else factor * r["std"] for r in data])
            ax.plot(x, y, label=f"{arm} {variant}", color=COLORS[arm],
                    linestyle=ARM_STYLES.get(arm, STYLES.get(variant, "-")), marker=ARM_MARKERS.get(arm, "."))
            ax.fill_between(x, y-std, y+std, color=COLORS[arm], alpha=0.08)
        ax.set(xlabel="Original epoch (not aligned to best)", ylabel=metric + (" (%)" if metric in PROPORTIONS else ""),
               title=f"{task} / k={k} / {group}")
        ax.legend(fontsize=7, ncol=2)
        finish(fig, "diagnostics", f"trajectory_{task}_k{k}_{group}_{metric}")

    # Layer appendices preserve layer identities instead of merging expert IDs across layers.
    layers = defaultdict(list)
    for row in rows:
        if row["layer"] != "model" and row["role"] in ("best", "static") and row["metric"] in displayed:
            layers[(row["task"], row["k"], row["group"], row["metric"])].append(row)
    for (task, k, group, metric), points in layers.items():
        methods = sorted({(r["arm"], r["variant"]) for r in points})
        names = sorted({r["layer"] for r in points})
        lookup = {(r["arm"], r["variant"], r["layer"]): r["mean"] for r in points}
        array = np.array([[np.nan if lookup.get((*method, layer)) is None else lookup[(*method, layer)] for layer in names] for method in methods])
        fig, ax = plt.subplots(figsize=(11, 6))
        im = ax.imshow(array, aspect="auto")
        ax.set_xticks(range(len(names)), names, rotation=55, ha="right", fontsize=7)
        ax.set_yticks(range(len(methods)), [" ".join(m) for m in methods], fontsize=7)
        ax.set_title(f"{task} / k={k} / {group}/{metric} / best-validation")
        fig.colorbar(im, ax=ax)
        finish(fig, "appendix", f"layers_{task}_k{k}_{group}_{metric}")
