from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from task8_p01.common.config import BUDGETS, EXPERTS, MODES, TASKS, resolve_path


COLORS = {"M00": "#4c78a8", "M01": "#f58518", "M10": "#54a24b", "M11": "#e45756"}
MARKERS = {"M00": "o", "M01": "s", "M10": "D", "M11": "^"}


def _read(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def render_figures(config, run_id="p01"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = resolve_path(config, config["execution"]["output_root"]) / "results" / run_id
    rows = _read(base / "tables/p1_full_four_cell.csv")
    effects = _read(base / "tables/p1_effects.csv")
    root = base / "figures"
    root.mkdir(parents=True, exist_ok=True)
    dpi = int(config["reporting"]["png_dpi"])

    def save(fig, name):
        fig.savefig(root / f"{name}.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(root / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    def role_match(row, mode, role):
        return row["checkpoint_role"] == ("static" if mode == "M00" else role)

    for role in ("init", "best"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
        for axis, experts in zip(axes, EXPERTS):
            for mode in MODES:
                points = []
                for k in BUDGETS[experts]:
                    values = [
                        float(row["native"]) for row in rows
                        if int(row["E"]) == experts and int(row["k"]) == k
                        and row["task"] == "__macro__" and row["mode"] == mode
                        and role_match(row, mode, role)
                    ]
                    if values:
                        points.append((100 * k / experts, float(np.mean(values)), None if len(values) == 1 else float(np.std(values, ddof=1))))
                if points:
                    axis.errorbar(
                        [p[0] for p in points], [100 * p[1] for p in points],
                        yerr=[0 if p[2] is None else 100 * p[2] for p in points],
                        color=COLORS[mode], marker=MARKERS[mode], capsize=2, label=mode,
                    )
            axis.set(title=f"E={experts}", xlabel="Selected experts (%)")
            axis.set_xticks([100 * k / experts for k in BUDGETS[experts]], ["10%", "15%", "20%"])
        axes[0].set_ylabel("Native macro score (points)")
        axes[-1].legend(ncol=2, fontsize=8)
        fig.suptitle(f"Task 8 P1: selector × aggregation ({role})")
        save(fig, f"p1_four_cell_macro_{role}")

    fields = ("aggregation_R2", "aggregation_R4", "selection_uniform", "selection_soft", "interaction")
    for role in ("init", "best"):
        macro = [row for row in effects if row["task"] == "__macro__" and row["checkpoint_role"] == role]
        labels, matrix = [], []
        for experts in EXPERTS:
            for k in BUDGETS[experts]:
                selected = [row for row in macro if int(row["E"]) == experts and int(row["k"]) == k]
                if selected:
                    labels.append(f"E{experts}/k{k}")
                    matrix.append([100 * np.mean([float(row[field]) for row in selected]) for field in fields])
        if matrix:
            values = np.asarray(matrix)
            limit = max(0.01, float(np.max(np.abs(values))))
            fig, axis = plt.subplots(figsize=(9, 6))
            image = axis.imshow(values, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    axis.text(j, i, f"{values[i, j]:+.2f}", ha="center", va="center", fontsize=8)
            axis.set(xticks=range(len(fields)), xticklabels=fields, yticks=range(len(labels)), yticklabels=labels)
            axis.tick_params(axis="x", rotation=30)
            fig.colorbar(image, ax=axis, label="Δ native macro (points)")
            axis.set_title(f"P1 factorial effects ({role})")
            save(fig, f"p1_effect_heatmap_{role}")

    for role in ("init", "best"):
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
        for axis, task in zip(axes, TASKS):
            task_rows = [row for row in effects if row["task"] == task and row["checkpoint_role"] == role]
            x = np.arange(len(task_rows))
            for offset, mode in zip((-0.2, 0, 0.2), ("M01", "M10", "M11")):
                axis.scatter(x + offset, [100 * (float(row[mode]) - float(row["M00"])) for row in task_rows],
                             s=15, color=COLORS[mode], marker=MARKERS[mode], label=mode)
            axis.axhline(0, color="black", linewidth=.8, linestyle=":")
            axis.set(title=task.upper(), xlabel="D9" if role == "init" else "D9 × seed")
        axes[0].set_ylabel("Δ native vs M00 (points)")
        axes[-1].legend(fontsize=8)
        fig.suptitle(f"P1 task-level effects ({role})")
        save(fig, f"p1_task_effects_{role}")

    replay_path = base / "tables/p1_local_replay.csv"
    if replay_path.is_file() and replay_path.stat().st_size:
        replay = _read(replay_path)
        layers = [f"encoder_layer_{i:02d}" for i in range(6)] + [f"decoder_layer_{i:02d}" for i in range(6)]
        for role in ("init", "best"):
            fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
            for axis, trace_source in zip(axes, ("dense", "M00", "M11")):
                for mode in MODES:
                    y = []
                    for layer in layers:
                        selected = [
                            float(row["relative_l2_error_mean"]) for row in replay
                            if row["trace_source"] == trace_source and row["layer"] == layer
                            and row["mode"] == mode and row["checkpoint_role"] == role
                        ]
                        y.append(np.mean(selected) if selected else np.nan)
                    axis.plot(range(12), y, color=COLORS[mode], marker=MARKERS[mode], label=mode)
                axis.axvline(5.5, color="grey", linestyle=":")
                axis.set(title=f"{trace_source} trace", xlabel="Layer", xticks=range(12),
                         xticklabels=[f"E{i}" for i in range(6)] + [f"D{i}" for i in range(6)])
            axes[0].set_ylabel("Relative FFN output L2 error")
            axes[-1].legend(ncol=2, fontsize=8)
            fig.suptitle(f"P1 fixed-hidden local replay ({role})")
            save(fig, f"p1_local_replay_by_layer_{role}")
    return root
