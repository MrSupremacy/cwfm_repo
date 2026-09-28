from __future__ import annotations

from collections import defaultdict

from task6_phaseb.common.config import ARMS
from task6_phaseb.visualization.common import (
    COLORS,
    MARKERS,
    _artifact_rows,
    _main_role,
    _resource_rows,
    _results_root,
    data_file,
    save_csv,
)


LINESTYLES = {"R4d": "--", "G2-0.001": ":"}
BUDGETS = {
    "ratio10": {64: 6, 128: 13, 256: 26},
    "ratio20": {64: 13, 128: 26, 256: 51},
    "ratio30": {64: 19, 128: 38, 256: 77},
    "ratio40": {64: 26, 128: 51, 256: 102},
}
DISPLAY = (
    "accuracy",
    "relative_performance",
    "cv",
    "churn",
    "exact_set_change",
    "oracle_overlap",
    "adjusted_overlap",
    "activation_coverage",
    "boundary_gap",
    "normalized_boundary_gap",
    "boundary_tie_rate",
    "all_zero_rate",
)
PROPORTIONS = {
    "accuracy",
    "churn",
    "exact_set_change",
    "oracle_overlap",
    "adjusted_overlap",
    "activation_coverage",
    "boundary_tie_rate",
    "all_zero_rate",
}


def _comparison_role(row):
    return row["role"] == "final" if row["group"] == "churn" else _main_role(row)


def _budget_label(experts, k):
    for label, mapping in BUDGETS.items():
        if mapping.get(int(experts)) == int(k):
            return label
    raise ValueError(f"Unregistered matched budget: E={experts}, k={k}")


def _annotate_budget(row):
    experts, k = int(row["experts"]), int(row["k"])
    return {
        "budget": _budget_label(experts, k),
        "selected_expert_ratio": k / experts,
        **row,
    }


def _figure_path(root, experts, folder, run_id, name):
    if experts not in (128, 256):
        raise ValueError("Phase B figures are only generated for E128 and E256")
    if folder not in ("main", "diagnostics", "appendix"):
        raise ValueError(f"Unknown figure section: {folder}")
    return root / f"{experts}_experts" / folder / run_id / name


def _formatted(value, metric):
    if value is None:
        return "NA"
    if metric in PROPORTIONS:
        return f"{100 * value:.2f}%"
    if metric == "relative_performance":
        return f"{value:.2f}"
    return f"{value:.3e}" if value != 0 and abs(value) < 1.0e-4 else f"{value:.4f}"


def _write_expert_report(path, experts, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Phase B/F0 E{experts} best-validation results",
        "",
        "同一 validation set 用于选择 checkpoint 和报告结果；这些数值不是独立 test estimate。",
        "",
        "| task | arm | k | selected ratio | metric | mean ± seed std |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        mean = _formatted(row["mean"], row["metric"])
        std = "deterministic" if row["deterministic"] else _formatted(row["std"], row["metric"])
        lines.append(
            f"| {row['task']} | {row['arm']} | {row['k']} | {100 * row['selected_expert_ratio']:.2f}% "
            f"| {row['group']}/{row['metric']} | {mean} ± {std} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tables(config, run_id, *, result_root=None):
    aggregated = data_file(config, run_id, "aggregated", result_root=result_root)["rows"]
    normalized = data_file(config, run_id, "normalized", result_root=result_root)["rows"]
    paired = data_file(config, run_id, "paired_differences", result_root=result_root)["rows"]
    changes = data_file(config, run_id, "gap_changes", result_root=result_root)["rows"]
    root = _results_root(config, result_root) / "tables"
    artifact = _artifact_rows(config, run_id)
    resources = _resource_rows(config, run_id)

    for experts in (128, 256):
        expert_root = root / f"{experts}_experts"
        main = sorted(
            [
                _annotate_budget(row)
                for row in aggregated
                if row["experts"] == experts
                and row["layer"] == "model"
                and row["metric"] in DISPLAY
                and _main_role(row)
            ],
            key=lambda row: (
                row["task"], row["group"], row["metric"], row["k"], ARMS.index(row["arm"])
            ),
        )
        trajectories = sorted(
            [
                _annotate_budget(row)
                for row in aggregated
                if row["experts"] == experts
                and row["role"] in ("trajectory", "final")
                and row["metric"] in DISPLAY
            ],
            key=lambda row: (
                row["task"], row["k"], row["group"], row["metric"],
                ARMS.index(row["arm"]), row["epoch"] if row["epoch"] is not None else -1,
            ),
        )
        paired_expert = sorted(
            [_annotate_budget(row) for row in paired if row["experts"] == experts],
            key=lambda row: (
                row["task"], row["k"], row["metric"], row["role"], row["comparison"]
            ),
        )
        per_layer = [row for row in normalized if row.get("experts") == experts]
        completeness = [row for row in artifact if row["experts"] == experts]
        expert_resources = [row for row in resources if row["experts"] == experts]

        save_csv(expert_root / "main" / run_id / "best_validation.csv", main)
        _write_expert_report(expert_root / "main" / run_id / "best_validation.md", experts, main)
        save_csv(expert_root / "diagnostics" / run_id / "trajectories_and_final.csv", trajectories)
        save_csv(expert_root / "diagnostics" / run_id / "paired_differences_vs_G1.csv", paired_expert)
        save_csv(expert_root / "appendix" / run_id / "per_layer_per_seed.csv", per_layer)
        save_csv(expert_root / "appendix" / run_id / "condition_artifact_completeness.csv", completeness)
        save_csv(expert_root / "appendix" / run_id / "budget_parameters_artifact_counts.csv", expert_resources)

    comparison = sorted(
        [
            _annotate_budget(row)
            for row in aggregated
            if row.get("experts") in (64, 128, 256)
            and row["layer"] == "model"
            and row["metric"] in DISPLAY
            and _comparison_role(row)
        ],
        key=lambda row: (
            row["task"], row["budget"], row["group"], row["metric"],
            row["experts"], ARMS.index(row["arm"]),
        ),
    )
    paired_comparison = sorted(
        [_annotate_budget(row) for row in paired],
        key=lambda row: (
            row["task"], row["budget"], row["metric"], row["role"],
            row["experts"], row["comparison"],
        ),
    )
    change_comparison = sorted(
        changes,
        key=lambda row: (
            row["task"], row["budget"], row["metric"], row["role"],
            row["experts"], row["comparison"],
        ),
    )
    comparison_root = root / "expert_comparison" / run_id
    dense = [
        row
        for row in aggregated
        if row["arm"] == "dense"
        and row["group"] == "performance"
        and row["layer"] == "model"
        and row["role"] == "static"
        and row["metric"] in ("accuracy", "relative_performance")
    ]
    save_csv(comparison_root / "00_dense_init_reference.csv", dense)
    save_csv(comparison_root / "01_matched_ratio_metrics.csv", comparison)
    save_csv(comparison_root / "02_paired_G1_gaps.csv", paired_comparison)
    save_csv(comparison_root / "03_gap_changes_vs_E64.csv", change_comparison)
    save_csv(comparison_root / "04_budget_parameters_artifact_counts.csv", resources)
    (comparison_root / "README.md").write_text(
        "# E64/E128/E256 comparison tables\n\n"
        "四档 ratio 使用预先登记的整数 k 匹配。此目录只生成表格，不生成跨 E 图。\n",
        encoding="utf-8",
    )
    print(f"Wrote per-E tables and E64/E128/E256 comparison tables under {root}")


def figures(config, run_id, *, result_root=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = data_file(config, run_id, "aggregated", result_root=result_root)["rows"]
    root = _results_root(config, result_root) / "figures"
    written = 0

    def finish(fig, experts, folder, name):
        nonlocal written
        path = _figure_path(root, experts, folder, run_id, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path.with_suffix(".pdf"))
        fig.savefig(path.with_suffix(".png"), dpi=config["metrics"]["png_dpi"])
        plt.close(fig)
        written += 1

    for experts in (128, 256):
        selected = [
            row
            for row in rows
            if row.get("experts") == experts
            and row["layer"] == "model"
            and row["metric"] in DISPLAY
            and row["arm"] != "dense"
            and _main_role(row)
        ]
        grouped = defaultdict(list)
        for row in selected:
            grouped[(row["task"], row["group"], row["metric"])].append(row)

        for (task, group, metric), points in sorted(grouped.items()):
            fig, axis = plt.subplots(figsize=(8, 4.8))
            for arm in ARMS:
                variants = sorted({row["variant"] for row in points if row["arm"] == arm})
                for variant in variants:
                    data = sorted(
                        [row for row in points if (row["arm"], row["variant"]) == (arm, variant)],
                        key=lambda row: row["k"],
                    )
                    if not data:
                        continue
                    factor = 100 if metric in PROPORTIONS else 1
                    x = [100 * row["k"] / experts for row in data]
                    y = [np.nan if row["mean"] is None else factor * row["mean"] for row in data]
                    error = [np.nan if row["std"] is None else factor * row["std"] for row in data]
                    label = arm if variant == "default" else f"{arm} {variant}"
                    axis.errorbar(
                        x,
                        y,
                        yerr=error if any(row["std"] is not None for row in data) else None,
                        color=COLORS[arm],
                        marker=MARKERS[arm],
                        linestyle=LINESTYLES.get(arm, "-"),
                        label=label,
                        capsize=2,
                    )
            if metric == "relative_performance":
                axis.axhline(100, color="black", linestyle=":", label="Dense-init (100%)")
            if metric == "oracle_overlap":
                x = np.asarray(config["experiment"]["budgets"][experts], dtype=float) / experts * 100
                axis.plot(x, x, color="black", linestyle=":", label="uniform-random expectation")
            if metric == "adjusted_overlap":
                axis.axhline(0, color="black", linestyle=":", label="uniform-random expectation")
            axis.set(
                xlabel="Selected experts (%)",
                ylabel=metric + (" (%)" if metric in PROPORTIONS else ""),
                title=f"{task.upper()} / E={experts} / {group} / best-validation",
            )
            axis.legend(fontsize=7, ncol=2)
            folder = "main" if metric == "relative_performance" else "diagnostics"
            finish(fig, experts, folder, f"{task}_{group}_{metric}")

        trajectories = defaultdict(list)
        for row in rows:
            if (
                row.get("experts") == experts
                and row["layer"] == "model"
                and row["role"] == "trajectory"
                and row["metric"] in DISPLAY
            ):
                trajectories[(row["task"], row["k"], row["group"], row["metric"])].append(row)
        for (task, k, group, metric), points in sorted(trajectories.items()):
            fig, axis = plt.subplots(figsize=(8, 4.8))
            for arm in ARMS:
                variants = sorted({row["variant"] for row in points if row["arm"] == arm})
                for variant in variants:
                    data = sorted(
                        [row for row in points if (row["arm"], row["variant"]) == (arm, variant)],
                        key=lambda row: row["epoch"],
                    )
                    if not data:
                        continue
                    factor = 100 if metric in PROPORTIONS else 1
                    x = [row["epoch"] for row in data]
                    y = np.asarray([
                        np.nan if row["mean"] is None else factor * row["mean"] for row in data
                    ])
                    std = np.asarray([
                        np.nan if row["std"] is None else factor * row["std"] for row in data
                    ])
                    label = arm if variant == "default" else f"{arm} {variant}"
                    axis.plot(
                        x,
                        y,
                        color=COLORS[arm],
                        marker=MARKERS[arm],
                        linestyle=LINESTYLES.get(arm, "-"),
                        label=label,
                    )
                    axis.fill_between(x, y - std, y + std, color=COLORS[arm], alpha=0.08)
            axis.set(
                xlabel="Original epoch (not aligned to best)",
                ylabel=metric + (" (%)" if metric in PROPORTIONS else ""),
                title=f"{task.upper()} / E={experts} / k={k} / {group}",
            )
            axis.legend(fontsize=7, ncol=2)
            finish(fig, experts, "diagnostics", f"trajectory_{task}_k{k}_{group}_{metric}")

        layers = defaultdict(list)
        for row in rows:
            if (
                row.get("experts") == experts
                and row["layer"] != "model"
                and row["metric"] in DISPLAY
                and _main_role(row)
            ):
                layers[(row["task"], row["k"], row["group"], row["metric"])].append(row)
        for (task, k, group, metric), points in sorted(layers.items()):
            methods = [
                (arm, variant)
                for arm in ARMS
                for variant in sorted({row["variant"] for row in points if row["arm"] == arm})
            ]
            names = sorted({row["layer"] for row in points})
            lookup = {
                (row["arm"], row["variant"], row["layer"]): row["mean"] for row in points
            }
            values = np.array([
                [
                    np.nan if lookup.get((*method, layer)) is None else lookup[(*method, layer)]
                    for layer in names
                ]
                for method in methods
            ])
            fig, axis = plt.subplots(figsize=(11, 6))
            image = axis.imshow(values, aspect="auto")
            axis.set_xticks(range(len(names)), names, rotation=55, ha="right", fontsize=7)
            axis.set_yticks(range(len(methods)), [" ".join(method) for method in methods], fontsize=7)
            axis.set_title(
                f"{task.upper()} / E={experts} / k={k} / {group}/{metric} / best-validation"
            )
            fig.colorbar(image, ax=axis)
            finish(fig, experts, "appendix", f"layers_{task}_k{k}_{group}_{metric}")

    print(
        f"Wrote {written} within-E figure groups as PNG/PDF under {root}; "
        "no cross-E figures were produced"
    )
