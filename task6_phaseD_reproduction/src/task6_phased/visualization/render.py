from __future__ import annotations

import csv
from collections import defaultdict
import shutil

from task6_phased.common.config import (
    ARMS, TASKS, budgets_for, conditions, routed_root, run_path, suite_experts,
)
from task6_phased.common.io import read_json
from task6_phased.metrics.performance.pipeline import best_state


COLORS = {
    "R2": "#4c78a8", "R4o": "#f58518", "R4d": "#e45756",
    "G1": "#72b7b2", "G2-0.001": "#54a24b", "G4": "#b279a2",
}
MARKERS = {"R2": "o", "R4o": "s", "R4d": "^", "G1": "D", "G2-0.001": "v", "G4": "P"}


def _data(config, run_id, name="aggregated"):
    return read_json(routed_root(config) / "results" / run_id / "data" / name / "metrics.json")["rows"]


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _copy_legacy_e64(root):
    target = root / "E64"
    copied = False
    for section in ("main", "diagnostics", "appendix"):
        source = root / section
        if source.is_dir():
            shutil.copytree(source, target / section, dirs_exist_ok=True)
            copied = True
    return copied


def _tables_for_experts(config, run_id, experts):
    rows = [row for row in _data(config, run_id) if int(row["experts"]) == experts]
    paired = [
        row for row in _data(config, run_id, "paired_differences")
        if int(row["experts"]) == experts
    ]
    seed_baseline = [
        row for row in _data(config, run_id, "seed_baseline")
        if int(row["experts"]) == experts
    ]
    root = routed_root(config) / "results" / run_id / "tables" / f"E{experts}"
    best = [row for row in rows if row["group"] == "performance" and row["role"] == "best"]
    final = [row for row in rows if row["group"] == "performance" and row["role"] == "final"]
    specialization = [
        row for row in rows if row["group"] == "specialization" and row["role"] == "best"
        and row["metric"] in (
            "populations.encoder_content.mean_js_distance_layer_mean",
            "populations.encoder_content.max_js_distance_layer_mean",
            "populations.encoder_content.mi_bits_layer_mean",
            "populations.encoder_content.nmi_domain_layer_mean",
        )
    ]
    _csv(root / "main/routed_best.csv", best)
    _csv(root / "main/specialization_summary.csv", specialization)
    _csv(root / "main/paired_best_vs_G1.csv", paired)
    _csv(root / "diagnostics/routed_final_raw.csv", final)
    for group, name in (
        ("load_balance", "load_balance"), ("selection_quality", "selection_quality"),
        ("churn", "stability"), ("specialization", "specialization_all_populations"),
    ):
        _csv(root / f"diagnostics/{name}.csv", [row for row in rows if row["group"] == group])
    js_pairs = [
        row for row in rows if row["group"] == "specialization" and row["role"] == "best"
        and ".pairwise_js_distance." in row["metric"]
    ]
    mi_by_layer = [
        row for row in rows if row["group"] == "specialization" and row["role"] == "best"
        and ".layers." in row["metric"] and row["metric"].endswith((".mi_bits", ".nmi_domain"))
    ]
    _csv(root / "diagnostics/js_pairs_and_seed_baseline.csv", [*js_pairs, *seed_baseline])
    _csv(root / "diagnostics/mi_by_layer.csv", mi_by_layer)
    population_counts = []
    for condition in conditions(config):
        if condition.experts != experts:
            continue
        state = best_state(config, condition, run_id)
        payload = read_json(
            run_path(config, "metrics/specialization", condition, run_id) / state["name"] / "metrics.json"
        )
        for population, population_values in payload["populations"].items():
            for layer, layer_values in population_values["layers"].items():
                for task, counts in layer_values["domain_counts"].items():
                    population_counts.append({
                        **condition.to_dict(), "state": state["name"], "checkpoint_role": "best",
                        "population": population, "layer": layer, "domain": task,
                        "token_count": int(sum(counts) // condition.k),
                        "assignment_count": int(sum(counts)),
                    })
    _csv(root / "diagnostics/population_token_counts.csv", population_counts)
    _csv(root / "appendix/per_layer_per_seed.csv", _data(config, run_id, "normalized"))
    lines = [
        f"# Phase D/F0 E{experts} best performance", "",
        "所有 Dense-MT delta 与 arm 配对性能差只使用 performance-best；final 仅在 diagnostics 中保留原始分数。", "",
        "| arm | k | domain | metric | mean | seed std |", "|---|---:|---|---|---:|---:|",
    ]
    for row in sorted(best, key=lambda x: (ARMS.index(x["arm"]), x["k"], x["domain"], x["metric"])):
        if row["metric"] not in ("native", "macro", "worst_domain", "delta_vs_dense_best"):
            continue
        spread = "deterministic" if row["std"] is None else f"{row['std']:.6f}"
        lines.append(
            f"| {row['arm']} | {row['k']} | {row['domain']} | {row['metric']} "
            f"| {row['mean']:.6f} | {spread} |"
        )
    (root / "main/routed_best.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def tables(config, run_id):
    root = routed_root(config) / "results" / run_id / "tables"
    if not _copy_legacy_e64(root):
        raise FileNotFoundError(
            f"Existing routed02 E64 tables are required before supplementation: {root}"
        )
    for experts in suite_experts(config):
        _tables_for_experts(config, run_id, experts)
    return root


def _figures_for_experts(config, run_id, experts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    rows = [row for row in _data(config, run_id) if int(row["experts"]) == experts]
    root = routed_root(config) / "results" / run_id / "figures" / f"E{experts}"
    ticks = [100 * k / experts for k in budgets_for(config, experts)]
    tick_labels = [f"{label}%\n({value:.6g})" for label, value in zip((10, 15, 20), ticks)]

    def save(fig, section, name):
        path = root / section / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path.with_suffix(".pdf"))
        fig.savefig(path.with_suffix(".png"), dpi=config["metrics"]["png_dpi"])
        plt.close(fig)

    def budget_points(selected, arm):
        points = sorted([row for row in selected if row["arm"] == arm], key=lambda row: row["k"])
        budgets = [row["k"] for row in points]
        if len(budgets) != len(set(budgets)):
            raise ValueError(
                f"duplicate aggregated best rows for arm={arm}, k={budgets}; "
                "rerun `aggregate` with the corrected summarizer before rendering figures"
            )
        return points

    def plot_budget(selected, ylabel, title, section, name, scale=1.0, baseline=None):
        fig, axis = plt.subplots(figsize=(8.2, 4.8))
        for arm in ARMS:
            points = budget_points(selected, arm)
            if not points:
                continue
            x = [100 * row["k"] / experts for row in points]
            y = [scale * row["mean"] for row in points]
            error = None if all(row["std"] is None for row in points) else [
                0.0 if row["std"] is None else scale * row["std"] for row in points
            ]
            axis.errorbar(x, y, yerr=error, color=COLORS[arm], marker=MARKERS[arm], label=arm, capsize=2)
        if baseline is not None:
            axis.axhline(baseline, color="black", linestyle=":", label="Dense-MT best")
        axis.set(xlabel="Selected experts (%)", ylabel=ylabel, title=f"E={experts} / {title}")
        axis.set_xticks(ticks, tick_labels)
        axis.legend(ncol=2, fontsize=8)
        save(fig, section, name)

    for task in TASKS:
        plot_budget(
            [row for row in rows if row["group"] == "performance" and row["role"] == "best"
             and row["domain"] == task and row["metric"] == "delta_vs_dense_best"],
            "Native score difference (points)", f"{task.upper()} vs Dense-MT macro-best",
            "main", f"{task}_native_delta_vs_budget", scale=100,
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
        for axis, stack in zip(axes, ("encoder", "decoder")):
            for arm in ARMS:
                points = budget_points([
                    row for row in rows if row["group"] == "load_balance" and row["role"] == "best"
                    and row["metric"] == f"tasks.{task}.{stack}_mean"
                ], arm)
                if points:
                    axis.errorbar(
                        [100 * row["k"] / experts for row in points], [row["mean"] for row in points],
                        yerr=None if all(row["std"] is None for row in points) else [
                            0 if row["std"] is None else row["std"] for row in points
                        ], color=COLORS[arm], marker=MARKERS[arm], label=arm, capsize=2,
                    )
            axis.set(title=stack.title(), xlabel="Selected experts (%)", ylabel="CV")
            axis.set_xticks(ticks, ["10%", "15%", "20%"])
        axes[1].legend(ncol=2, fontsize=8)
        fig.suptitle(f"{task.upper()} load balance by stack")
        save(fig, "diagnostics", f"{task}_load_cv")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
        for axis, field in zip(axes, ("oracle_overlap", "activation_coverage")):
            for arm in ARMS:
                points = budget_points([
                    row for row in rows if row["group"] == "selection_quality" and row["role"] == "best"
                    and row["metric"] == f"tasks.{task}.all_layer_mean.{field}"
                ], arm)
                if points:
                    axis.errorbar(
                        [100 * row["k"] / experts for row in points], [row["mean"] for row in points],
                        yerr=None if all(row["std"] is None for row in points) else [
                            0 if row["std"] is None else row["std"] for row in points
                        ], color=COLORS[arm], marker=MARKERS[arm], label=arm, capsize=2,
                    )
            axis.set(title=field.replace("_", " ").title(), xlabel="Selected experts (%)")
            axis.set_xticks(ticks, ["10%", "15%", "20%"])
        axes[1].legend(ncol=2, fontsize=8)
        fig.suptitle(f"{task.upper()} selection quality")
        save(fig, "diagnostics", f"{task}_selection_quality")
    for metric, ylabel in (("macro", "Macro native score"), ("worst_domain", "Worst-domain native score")):
        plot_budget(
            [row for row in rows if row["group"] == "performance" and row["role"] == "best"
             and row["domain"] == "all" and row["metric"] == metric],
            ylabel, f"Phase D/F0 {metric.replace('_', ' ')}", "main", f"{metric}_vs_budget", scale=100,
        )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for axis, metric, ylabel in zip(
        axes,
        ("populations.encoder_content.mean_js_distance_layer_mean", "populations.encoder_content.nmi_domain_layer_mean"),
        ("Mean JS distance", "Domain NMI"),
    ):
        for arm in ARMS:
            points = budget_points([
                row for row in rows if row["group"] == "specialization" and row["role"] == "best"
                and row["metric"] == metric
            ], arm)
            if points:
                axis.errorbar(
                    [100 * row["k"] / experts for row in points], [row["mean"] for row in points],
                    yerr=None if all(row["std"] is None for row in points) else [
                        0 if row["std"] is None else row["std"] for row in points
                    ], color=COLORS[arm], marker=MARKERS[arm], label=arm, capsize=2,
                )
        axis.set(title=ylabel, xlabel="Selected experts (%)")
        axis.set_xticks(ticks, ["10%", "15%", "20%"])
    axes[1].legend(ncol=2, fontsize=8)
    fig.suptitle("Encoder-content task specialization")
    save(fig, "main", "specialization_vs_budget")
    for task in TASKS:
        for k in budgets_for(config, experts):
            fig, axis = plt.subplots(figsize=(8.2, 4.8))
            for arm in ARMS:
                points = sorted([
                    row for row in rows if row["group"] == "churn" and row["domain"] == "structured"
                    and row["metric"] == f"tasks.{task}.all_layer_mean" and row["k"] == k and row["arm"] == arm
                    and row["role"] == "trajectory"
                ], key=lambda row: row["step"])
                if points:
                    axis.plot([100 * row["step"] / 2640 for row in points], [row["mean"] for row in points],
                              color=COLORS[arm], marker=MARKERS[arm], label=arm)
            axis.set(xlabel="Training progress (%)", ylabel="Churn", title=f"{task.upper()} churn / k={k}")
            axis.legend(ncol=2, fontsize=8)
            save(fig, "diagnostics", f"{task}_churn_k{k}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for column, population in enumerate(("encoder_prefix", "encoder_content", "decoder")):
        for row_index, (suffix, ylabel) in enumerate((
            ("mean_js_distance_layer_mean", "Mean JS distance"),
            ("nmi_domain_layer_mean", "Domain NMI"),
        )):
            axis = axes[row_index, column]
            metric = f"populations.{population}.{suffix}"
            for arm in ARMS:
                points = budget_points([
                    item for item in rows if item["group"] == "specialization" and item["role"] == "best"
                    and item["metric"] == metric
                ], arm)
                if points:
                    axis.errorbar(
                        [100 * item["k"] / experts for item in points], [item["mean"] for item in points],
                        yerr=None if all(item["std"] is None for item in points) else [
                            0 if item["std"] is None else item["std"] for item in points
                        ], color=COLORS[arm], marker=MARKERS[arm], label=arm, capsize=2,
                    )
            axis.set(title=population.replace("_", " ").title(), ylabel=ylabel)
            axis.set_xticks(ticks, ["10%", "15%", "20%"])
            if row_index == 1:
                axis.set_xlabel("Selected experts (%)")
    axes[0, 2].legend(ncol=2, fontsize=8)
    fig.suptitle("Prefix, content, and decoder specialization")
    save(fig, "diagnostics", "prefix_content_decoder")

    raw_best = defaultdict(list)
    for condition in conditions(config):
        if condition.experts != experts:
            continue
        state = best_state(config, condition, run_id)
        payload = read_json(
            run_path(config, "metrics/specialization", condition, run_id) / state["name"] / "metrics.json"
        )
        raw_best[(condition.arm, condition.k)].append((condition, state, payload))

    def js_matrix(payload):
        matrix = np.zeros((len(TASKS), len(TASKS)), dtype=float)
        layers = payload["populations"]["encoder_content"]["layers"].values()
        for left_index, left in enumerate(TASKS):
            for right_index in range(left_index + 1, len(TASKS)):
                right = TASKS[right_index]
                key = f"{left}__{right}"
                value = float(np.mean([
                    layer["pairwise_js_distance"][key] for layer in layers
                ]))
                matrix[left_index, right_index] = matrix[right_index, left_index] = value
        return matrix

    for (arm, k), entries in raw_best.items():
        matrices = [(condition, js_matrix(payload)) for condition, _, payload in entries]
        panels = matrices if len(matrices) == 1 else [*matrices, (None, np.mean([value for _, value in matrices], axis=0))]
        fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4), squeeze=False)
        for axis, (condition, matrix) in zip(axes[0], panels):
            image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
            for row_index in range(len(TASKS)):
                for column in range(len(TASKS)):
                    axis.text(column, row_index, f"{matrix[row_index, column]:.3f}", ha="center", va="center", fontsize=7)
            label = "seed mean" if condition is None else ("deterministic" if condition.seed is None else f"seed {condition.seed}")
            axis.set(title=label, xticks=range(len(TASKS)), yticks=range(len(TASKS)),
                     xticklabels=[task.upper() for task in TASKS], yticklabels=[task.upper() for task in TASKS])
            axis.tick_params(axis="x", rotation=45)
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.75, label="JS distance")
        fig.suptitle(f"Encoder-content domain JS / {arm} / k={k}")
        save(fig, "diagnostics", f"js_matrix_{arm}_k{k}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
        for condition, _, payload in entries:
            label = "deterministic" if condition.seed is None else f"seed {condition.seed}"
            encoder = payload["populations"]["encoder_content"]["layers"]
            decoder = payload["populations"]["decoder"]["layers"]
            layer_values = [
                *(encoder[f"encoder_layer_{index:02d}"] for index in range(config["model"]["encoder_layers"])),
                *(decoder[f"decoder_layer_{index:02d}"] for index in range(config["model"]["decoder_layers"])),
            ]
            for axis, field in zip(axes, ("mi_bits", "nmi_domain")):
                axis.plot(range(len(layer_values)), [value[field] for value in layer_values], marker="o", label=label)
        for axis, title in zip(axes, ("MI (bits)", "NMI")):
            boundary = config["model"]["encoder_layers"] - .5
            axis.axvline(boundary, color="grey", linestyle=":")
            axis.set(title=title, xlabel="Layer (encoder-content, then decoder)", xticks=range(12),
                     xticklabels=[f"E{i}" for i in range(6)] + [f"D{i}" for i in range(6)])
        axes[1].legend(fontsize=8)
        fig.suptitle(f"Task information by layer / {arm} / k={k}")
        save(fig, "diagnostics", f"mi_by_layer_{arm}_k{k}")

        for condition, _, payload in entries:
            fig, axes = plt.subplots(3, 4, figsize=(20, 11), squeeze=False)
            layer_specs = [
                *(('encoder_content', f"encoder_layer_{index:02d}") for index in range(config["model"]["encoder_layers"])),
                *(('decoder', f"decoder_layer_{index:02d}") for index in range(config["model"]["decoder_layers"])),
            ]
            for axis, (population, layer) in zip(axes.ravel(), layer_specs):
                values = payload["populations"][population]["layers"][layer]["domain_counts"]
                heatmap = np.stack([
                    np.asarray(values[task], dtype=float) / max(1, np.sum(values[task])) for task in TASKS
                ])
                axis.imshow(heatmap, aspect="auto", vmin=0, cmap="magma")
                axis.set(title=layer, yticks=range(4), yticklabels=[task.upper() for task in TASKS],
                         xlabel="Expert ID")
            seed_name = "static" if condition.seed is None else str(condition.seed)
            fig.suptitle(f"Domain-conditioned utilization / {arm} / k={k} / seed={seed_name}")
            save(fig, "appendix", f"utilization_heatmap_{arm}_k{k}_seed{seed_name}")
    return root


def figures(config, run_id):
    root = routed_root(config) / "results" / run_id / "figures"
    if not _copy_legacy_e64(root):
        raise FileNotFoundError(
            f"Existing routed02 E64 figures are required before supplementation: {root}"
        )
    for experts in suite_experts(config):
        _figures_for_experts(config, run_id, experts)
    return root
