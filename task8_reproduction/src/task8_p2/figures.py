from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path

import numpy as np

from task8_p01.common.config import BUDGETS, EXPERTS, TASKS
from task8_p01.common.io import write_json
from task8_p2.reporting import read_rows
from task8_p2.progress import log


E_COLORS = {64: "#4c78a8", 128: "#f58518", 256: "#e45756"}
M_COLORS = {"1": "#e45756", "2": "#f58518", "4": "#4c78a8", "8": "#54a24b", "uniform": "#777777"}
N_COLORS = {"N0": "#777777", "N1": "#54a24b", "N2": "#f58518", "N3": "#4c78a8", "N4": "#e45756"}
LAYERS = [f"encoder_layer_{i:02d}" for i in range(6)] + [f"decoder_layer_{i:02d}" for i in range(6)]


def global_row(row):
    return all(row.get(key) == "__all__" for key in ("task", "layer_id", "stack", "token_group"))


def layer_or_global(row):
    return row["task"] == "__all__" and row["token_group"] == "__all__" and (row["layer_id"] != "__all__" or row["stack"] == "__all__")


def pooled_rank(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["rank"])].append(row)
    x, means, stds = [], [], []
    for rank, values in sorted(grouped.items()):
        counts = np.asarray([int(r["n_units"]) for r in values])
        mean = np.asarray([float(r["mean_p"]) for r in values])
        var = np.asarray([float(r["variance_p"]) if r["variance_p"] else 0 for r in values])
        total = counts.sum()
        m = np.sum(counts*mean)/total
        variance = np.sum((counts-1)*var+counts*np.square(mean-m))/(total-1) if total > 1 else 0
        x.append(rank)
        means.append(m)
        stds.append(math.sqrt(max(0, variance)))
    return np.asarray(x), np.asarray(means), np.asarray(stds)


def render_figures(config, root, *, local_only=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False})
    root = Path(root)
    output = root/"figures"
    output.mkdir()
    tables = root/"tables"
    names = []
    present_cells = set()

    def save(fig, name):
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(output/f"{name}.png", dpi=int(config["reporting"]["png_dpi"]), bbox_inches="tight")
        fig.savefig(output/f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        names.append(name)
        log(f"FIGURE {name}.png/.pdf")

    def grid(title, ylabel):
        fig, axes = plt.subplots(3, 3, figsize=(15, 11))
        for i, e in enumerate(EXPERTS):
            for j, k in enumerate(BUDGETS[e]):
                axis = axes[i, j]
                axis.set_title(f"E={e}, k={k} | {('low','mid','high')[j]} ({100*k/e:.5f}%)")
                axis.grid(alpha=.18)
                if (e,k) not in present_cells:
                    axis.text(.5,.5,"Not evaluated (smoke)",ha="center",va="center",color="grey",transform=axis.transAxes)
                if j == 0:
                    axis.set_ylabel(ylabel)
        fig.suptitle(title, fontsize=13)
        return fig, axes

    def cell(rows, e, k, **conditions):
        return [r for r in rows if int(r["E"]) == e and int(r["k"]) == k and
                all(str(r.get(key)) == str(value) for key, value in conditions.items())]

    support = {
        mode: list(read_rows(tables/filename, layer_or_global)) for mode, filename in
        (("natural", "p2_n2a_support_natural.csv"), ("shared_hidden", "p2_n2a_support_shared_hidden.csv"))
    }
    weights = list(read_rows(tables/"p2_n2a_weight_summary.csv", layer_or_global))
    present_cells = {(int(r["E"]),int(r["k"])) for r in support["natural"]}
    fig, axes = grid("N2-a natural rollout: expert support overlap", "Overlap@k")
    control, control_axes = grid("N2-a: natural vs Dense shared-hidden support", "Overlap@k")
    wfig, waxes = grid("N2-a natural rollout: departure from uniform weights", "Weight variance v(alpha)")
    for i, e in enumerate(EXPERTS):
        for j, k in enumerate(BUDGETS[e]):
            axis, ca, wa = axes[i,j], control_axes[i,j], waxes[i,j]
            for mode, color in (("natural", "#e45756"), ("shared_hidden", "#4c78a8")):
                selected = cell(support[mode], e, k)
                for seed in sorted({r["seed"] for r in selected}):
                    y = [np.mean([float(r["overlap_at_k"]) for r in selected if r["layer_id"] == layer and r["seed"] == seed])
                         if any(r["layer_id"] == layer and r["seed"] == seed for r in selected) else np.nan for layer in LAYERS]
                    if mode == "natural":
                        axis.plot(range(12), y, color=color, alpha=.3, linewidth=.8)
                y = [np.mean([float(r["overlap_at_k"]) for r in selected if r["layer_id"] == layer])
                     if any(r["layer_id"] == layer for r in selected) else np.nan for layer in LAYERS]
                ca.plot(range(12), y, color=color, marker="o", markersize=3, label=mode)
                if mode == "natural":
                    axis.plot(range(12), y, color=color, marker="o", markersize=3, linewidth=2)
            selected = cell(weights, e, k, forward_mode="natural")
            y = [np.mean([float(r["weight_variance"]) for r in selected if r["layer_id"] == layer])
                 if any(r["layer_id"] == layer for r in selected) else np.nan for layer in LAYERS]
            wa.plot(range(12), y, color="#e45756", marker="s", markersize=3)
            for ax in (axis, ca, wa):
                ax.axvline(5.5, color="grey", linestyle=":", linewidth=.8)
                ax.set_xticks(range(12), [f"E{x}" for x in range(6)]+[f"D{x}" for x in range(6)], rotation=45)
                ax.set_xlabel("FFN layer")
            axis.set_ylim(0, 1.02)
            ca.set_ylim(0, 1.02)
    control_axes[0,2].legend()
    save(fig, "p2_n2a_support_overlap_natural_d9")
    save(control, "p2_n2a_support_natural_vs_shared_by_layer")
    save(wfig, "p2_n2a_weight_variance_natural_d9")

    # Bounded views from large rank CSVs. Per-task/layer rows remain inspectable
    # in tables; no million-row materialization is needed for the main figures.
    rank_rows = list(read_rows(tables/"p2_n2a_weight_rank_profile.csv", lambda r:
                         r["task"] == "__all__" and r["layer_id"] == "__all__" and r["token_group"] == "__all__"))
    for mode in ("natural", "shared_hidden"):
        fig, axes = grid(f"N2-a rank-weight profiles ({mode}); band = unit SD", "Normalized weight p(rank)")
        for i,e in enumerate(EXPERTS):
            for j,k in enumerate(BUDGETS[e]):
                axis = axes[i,j]
                selected = cell(rank_rows, e, k, forward_mode=mode, stack="__all__")
                for seed in sorted({r["seed"] for r in selected}):
                    x,y,_ = pooled_rank([r for r in selected if r["seed"] == seed])
                    axis.plot(x,y,color="#e45756",linewidth=.8,alpha=.35)
                x,y,s = pooled_rank(selected)
                if len(x):
                    axis.plot(x,y,color="#e45756",linewidth=2,label="seed-pooled mean")
                    axis.fill_between(x, np.maximum(0,y-s), np.minimum(1,y+s), color="#e45756",alpha=.15,label="mean +/- 1 unit SD")
                axis.axhline(1/k,color="grey",linestyle="--",label="uniform 1/k")
                axis.set_xlabel("Weight rank")
                axis.set_ylim(bottom=0)
        axes[0,2].legend()
        save(fig, f"p2_n2a_rank_weight_{mode}_d9")
    fig,axes = grid("N2-a natural rank profiles: encoder / decoder", "Normalized weight p(rank)")
    for i,e in enumerate(EXPERTS):
        for j,k in enumerate(BUDGETS[e]):
            for stack,color in (("encoder","#4c78a8"),("decoder","#e45756")):
                x,y,_ = pooled_rank(cell(rank_rows,e,k,forward_mode="natural",stack=stack))
                axes[i,j].plot(x,y,color=color,label=stack)
            axes[i,j].axhline(1/k,color="grey",linestyle="--")
            axes[i,j].set_xlabel("Weight rank")
    axes[0,2].legend()
    save(fig,"p2_n2a_rank_weight_natural_by_stack")

    fig,axis = plt.subplots(figsize=(8,6))
    tier_markers = {"low":"o","mid":"s","high":"D"}
    for mode in ("natural","shared_hidden"):
        for row in support[mode]:
            if not global_row(row):
                continue
            wr = cell(weights,int(row["E"]),int(row["k"]),seed=row["seed"],forward_mode=mode,layer_id="__all__",stack="__all__")
            if wr:
                color = E_COLORS[int(row["E"])]
                axis.scatter(float(row["overlap_at_k"]),float(wr[0]["weight_variance"]),
                             marker=tier_markers[row["budget_tier"]],s=40,edgecolor=color,
                             facecolor=color if mode=="natural" else "none",alpha=.7)
    axis.set(xlabel="Mean overlap@k",ylabel="Mean weight variance v(alpha)",title="N2-a support vs weighting: filled=natural, open=shared hidden",xlim=(0,1.02))
    axis.grid(alpha=.2)
    handles = [Line2D([],[],marker="o",linestyle="",color=c,label=f"E={e}") for e,c in E_COLORS.items()]
    handles += [Line2D([],[],marker=m,linestyle="",color="grey",label=t) for t,m in tier_markers.items()]
    axis.legend(handles=handles,ncol=2)
    save(fig,"p2_n2a_overlap_vs_weight_variance")

    numeric = list(read_rows(tables/"p2_n1_numeric_chain.csv",global_row))
    nranks = list(read_rows(tables/"p2_n1_rank_profile.csv",global_row))
    for support_name in ("I4","I2"):
        fig,axes = grid(f"N1 fixed Dense hidden, fixed {support_name}: N0-N4", "Normalized weight p(rank)")
        labels,matrix = [],[]
        for i,e in enumerate(EXPERTS):
            for j,k in enumerate(BUDGETS[e]):
                for state in ("N0","N1","N2","N3","N4"):
                    selected = cell(nranks,e,k,support_source=support_name,n_state=state)
                    x,y,_ = pooled_rank(selected)
                    axes[i,j].plot(x,y,color=N_COLORS[state],label=state,linestyle="--" if state=="N0" else "-")
                axes[i,j].set_xlabel("Weight rank")
                selected = cell(numeric,e,k,support_source=support_name)
                labels.append(f"E{e}/k{k}")
                matrix.append([np.mean([float(r["delta_weight_variance_to_previous"]) for r in selected if r["n_state"]==s])
                               if any(r["n_state"]==s for r in selected) else np.nan for s in ("N1","N2","N3","N4")])
        axes[0,2].legend(ncol=3)
        save(fig,f"p2_n1_rank_chain_{support_name}_d9")
        fig,axis = plt.subplots(figsize=(9,6))
        values = np.asarray(matrix)
        limit = max(.001,float(np.nanmax(np.abs(values)))) if np.isfinite(values).any() else 1
        im = axis.imshow(values,aspect="auto",cmap="coolwarm",vmin=-limit,vmax=limit)
        for row in range(9):
            for col in range(4):
                if np.isfinite(values[row,col]):
                    axis.text(col,row,f"{values[row,col]:+.4f}",ha="center",va="center",fontsize=8)
        axis.set(xticks=range(4),xticklabels=["unit cosine N1-N0","scale N2-N1","input eps N3-N2","summary eps N4-N3"],
                 yticks=range(9),yticklabels=labels,title=f"N1 sequential increments in weight variance ({support_name})")
        axis.tick_params(axis="x",rotation=20)
        fig.colorbar(im,ax=axis,label="Delta weight variance (not score points)")
        save(fig,f"p2_n1_variance_increments_{support_name}")

    temperature = list(read_rows(tables/"p2_n3_local_temperature.csv",global_row))
    tranks = list(read_rows(tables/"p2_n3_rank_profile.csv",global_row))
    for mode in ("natural_R4d_replay","shared_hidden"):
        fig,axes = grid(f"N3 fixed-hidden temperature weighting ({mode})", "Mean weight variance v(alpha)")
        rfig,raxes = grid(f"N3 rank profiles ({mode}); band = unit SD", "Normalized weight p(rank)")
        for i,e in enumerate(EXPERTS):
            for j,k in enumerate(BUDGETS[e]):
                selected = cell(temperature,e,k,forward_mode=mode)
                for seed in sorted({r["seed"] for r in selected}):
                    vals = [r for r in selected if r["seed"]==seed]
                    y = [np.mean([float(r["weight_variance"]) for r in vals if r["multiplier"]==str(m)]) for m in (1,2,4,8)]
                    axes[i,j].plot(range(4),y,color="#e45756",alpha=.3,linewidth=.8)
                y = [np.mean([float(r["weight_variance"]) for r in selected if r["multiplier"]==str(m)])
                     if any(r["multiplier"]==str(m) for r in selected) else np.nan for m in (1,2,4,8)]
                axes[i,j].plot(range(4),y,color="#e45756",marker="o",linewidth=2)
                axes[i,j].axhline(0,color="grey",linestyle="--",linewidth=.8)
                axes[i,j].set(xticks=range(4),xticklabels=[1,2,4,8],xlabel="Temperature multiplier m")
                for m in (1,2,4,8):
                    x,y,s = pooled_rank(cell(tranks,e,k,forward_mode=mode,multiplier=m))
                    if len(x):
                        raxes[i,j].plot(x,y,color=M_COLORS[str(m)],label=f"m={m}")
                        raxes[i,j].fill_between(x,np.maximum(0,y-s),np.minimum(1,y+s),color=M_COLORS[str(m)],alpha=.08)
                raxes[i,j].axhline(1/k,color="grey",linestyle="--",label="uniform 1/k")
                raxes[i,j].set_xlabel("Weight rank")
        raxes[0,2].legend(ncol=2)
        save(fig,f"p2_n3_temperature_weight_variance_{mode}_d9")
        save(rfig,f"p2_n3_temperature_rank_{mode}_d9")

    if not local_only:
        full = list(read_rows(tables/"p2_n3_full_eval.csv"))
        for task in ("__macro__","__worst__",*TASKS):
            fig,axes = grid(f"N3 temperature performance: {task}; thin=seed, thick=mean", "Native score (points)")
            for i,e in enumerate(EXPERTS):
                for j,k in enumerate(BUDGETS[e]):
                    selected = cell(full,e,k,task=task)
                    for seed in sorted({r["seed"] for r in selected}):
                        vals = [r for r in selected if r["seed"]==seed]
                        y = [100*float(next(r["native"] for r in vals if r["multiplier"]==str(m))) for m in (1,2,4,8,"uniform")]
                        axes[i,j].plot(range(5),y,color="#e45756",alpha=.35,linewidth=.8,marker="o",markersize=2)
                    y = [100*np.mean([float(r["native"]) for r in selected if r["multiplier"]==str(m)])
                         if any(r["multiplier"]==str(m) for r in selected) else np.nan for m in (1,2,4,8,"uniform")]
                    axes[i,j].plot(range(5),y,color="#e45756",linewidth=2,marker="o",markersize=4)
                    sd = [100*np.std([float(r["native"]) for r in selected if r["multiplier"]==str(m)],ddof=1)
                          if sum(r["multiplier"]==str(m) for r in selected)>1 else 0 for m in (1,2,4,8,"uniform")]
                    axes[i,j].fill_between(range(5),np.asarray(y)-sd,np.asarray(y)+sd,color="#e45756",alpha=.12)
                    axes[i,j].set(xticks=range(5),xticklabels=["1","2","4","8","uniform"],xlabel="m (uniform = limit reference)")
            save(fig,f"p2_n3_temperature_performance_{task.strip('_')}_d9")
        fig,axis = plt.subplots(figsize=(9,6))
        for e in EXPERTS:
            for k in BUDGETS[e]:
                for seed in ("0","1","2"):
                    points = []
                    for m in (1,2,4,8,"uniform"):
                        macro = cell(full,e,k,seed=seed,task="__macro__",multiplier=m)
                        local = cell(temperature,e,k,seed=seed,forward_mode="natural_R4d_replay",multiplier=m)
                        if macro and (local or m=="uniform"):
                            x = 0 if m=="uniform" else float(local[0]["weight_variance"])
                            points.append((x,100*float(macro[0]["native"])))
                    if points:
                        axis.plot([p[0] for p in points],[p[1] for p in points],color=E_COLORS[e],alpha=.25,linewidth=.8)
                        axis.scatter([p[0] for p in points],[p[1] for p in points],color=E_COLORS[e],
                                     marker=tier_markers[("low","mid","high")[BUDGETS[e].index(k)]],alpha=.7,s=26)
        axis.set(xlabel="Mean weight variance on cached original M11 hidden",ylabel="Full-model native macro (points)",
                 title="N3 paired temperature trajectories (local variance vs full-model performance)")
        axis.grid(alpha=.2)
        axis.legend(handles=handles,ncol=2)
        save(fig,"p2_n3_performance_vs_weight_variance")
    write_json(output/"figure_index.json", {"figures": names, "formats": ["png","pdf"],
               "rank_band": "pooled nonpadding token-layer unit SD; not standard error or seed SD",
               "full_performance": "seed means; uniform is a limit reference",
               "n3_scatter": "x uses fixed original M11 trace, not changed-temperature full-model hidden"})
    return output
