from __future__ import annotations

import argparse
import json

from task8_p01.common.config import Snapshot, TASKS, cells, repository_root
from task8_p01.common.io import write_csv, write_json
from task8_p2.config import best_snapshots, load_config, matrix, output_root, protocol_id, validate_run_id
from task8_p2.progress import Progress, heartbeat, log


def parser():
    value = argparse.ArgumentParser(description="Task 8 P2-N: N2-a -> N1 -> N3; no training")
    subs = value.add_subparsers(dest="command",required=True)
    for command in ("matrix","preflight","prepare","local","full","worker","results","smoke","launch"):
        p = subs.add_parser(command)
        p.add_argument("--suite",default=str(repository_root()/"configs/suites/p2_n_forward.yaml"))
        p.add_argument("--local",default=None)
        if command in ("local","full"):
            p.add_argument("--experts",type=int,required=True)
            p.add_argument("--k",type=int,required=True)
            p.add_argument("--seed",type=int,required=True)
        if command == "worker":
            p.add_argument("--stage",choices=("local","full"),required=True)
            p.add_argument("--shard",type=int,required=True)
            p.add_argument("--shards",type=int,default=8)
        if command in ("results","smoke","launch"):
            p.add_argument("--run-id",required=True)
        if command in ("results","launch"):
            p.add_argument("--local-only",action="store_true")
        if command == "launch":
            p.add_argument("--gpus",type=int,default=8)
            p.add_argument("--full-only",action="store_true")
        if command == "results":
            p.add_argument("--no-figures",action="store_true")
        if command == "smoke":
            p.add_argument("--experts",type=int)
            p.add_argument("--k",type=int)
            p.add_argument("--seed",type=int,default=0)
            p.add_argument("--samples-per-task",type=int,default=2)
            p.add_argument("--no-figures",action="store_true")
    return value


def preflight(ctx):
    from task6_phased.data.datasets import load_raw
    from task8_p2.evaluation import p1_endpoint
    ctx.panel
    for task in TASKS:
        with heartbeat(f"verify parquet {task}",ctx.config["p2"]["heartbeat_seconds"]):
            load_raw(ctx.task6,task)  # checks source parquet hashes and counts
    rows = []
    progress = Progress("P2 best / reused endpoints inventory",27,ctx.config["p2"]["progress_interval_seconds"])
    for i,snapshot in enumerate(best_snapshots()):
        _, checkpoint = ctx.snapshot(snapshot)
        assets = ctx.assets(snapshot.experts)[3]
        for mode in ("M11","M10"):
            ref = p1_endpoint(ctx,snapshot,checkpoint,assets,mode)
            rows.append({**snapshot.to_dict(), "checkpoint_step":checkpoint["step"],
                         "checkpoint_hash":checkpoint["state_sha256"], "mode":mode,
                         "p1_artifact":ref["artifact_path"],"status":"verified"})
        progress.update(i+1)
    root = output_root(ctx.config)/"artifacts"/"preflight"/protocol_id(ctx.config)[:16]
    root.mkdir(parents=True,exist_ok=True)
    write_csv(root/"p2_asset_endpoint_inventory.csv",rows)
    write_json(root/"preflight.json",{"identity":ctx.identity(),"passed":True,"best_snapshots":27,
                                     "p1_endpoints":54,"new_training_runs":0})
    log(f"P2_PREFLIGHT_OK inventory={root}")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    config = load_config(args.suite,args.local)
    if hasattr(args,"run_id"):
        validate_run_id(args.run_id)
    if args.command == "matrix":
        print(json.dumps({**matrix(),"cells":[c.to_dict() for c in cells()],"output_root":str(output_root(config)),
                          "protocol":protocol_id(config)},ensure_ascii=False,indent=2))
        return
    if args.command == "launch":
        if args.local_only and args.full_only:
            raise ValueError("--local-only and --full-only are mutually exclusive")
        from task8_p2.launch import launch
        launch(config,args.suite,args.local,args.run_id,args.gpus,args.local_only,args.full_only)
        return
    from task8_p2.context import Context
    if args.command == "smoke":
        if not 1 <= args.samples_per_task <= 128:
            raise ValueError("Smoke samples-per-task must be between 1 and 128")
        config["p2"]["smoke_samples_per_task"] = args.samples_per_task
    ctx = Context(config,smoke=args.command=="smoke")
    if args.command == "preflight":
        preflight(ctx)
    elif args.command == "prepare":
        from task8_p2.traces import prepare
        prepare(ctx)
    elif args.command in ("local","full"):
        snapshot = Snapshot(args.experts,args.k,args.seed,"best")
        if args.command == "local":
            from task8_p2.local import analyze_snapshot
            analyze_snapshot(ctx,snapshot)
        else:
            from task8_p2.evaluation import full_snapshot
            full_snapshot(ctx,snapshot)
    elif args.command == "worker":
        if args.shards < 1 or not 0 <= args.shard < args.shards:
            raise ValueError("Invalid shard index/count")
        jobs = [s for i,s in enumerate(best_snapshots()) if i%args.shards==args.shard]
        log(f"WORKER stage={args.stage} shard={args.shard}/{args.shards} best_snapshots={len(jobs)}")
        from task8_p2.local import analyze_snapshot
        from task8_p2.evaluation import full_snapshot
        progress = Progress(f"{args.stage} snapshot jobs",len(jobs),config["p2"]["progress_interval_seconds"])
        for i,snapshot in enumerate(jobs):
            (analyze_snapshot if args.stage=="local" else full_snapshot)(ctx,snapshot)
            progress.update(i+1,force=True)
    elif args.command == "results":
        from task8_p2.reporting import aggregate
        aggregate(ctx,best_snapshots(),args.run_id,local_only=args.local_only,figures=not args.no_figures)
    elif args.command == "smoke":
        from task8_p2.traces import build_trace
        from task8_p2.local import analyze_snapshot
        from task8_p2.evaluation import full_snapshot
        from task8_p2.reporting import aggregate
        if (args.experts is None)!=(args.k is None):
            raise ValueError("Smoke --experts and --k must be supplied together")
        specs = [(args.experts,args.k)] if args.experts is not None else [(64,6),(128,26),(256,51)]
        snapshots = [Snapshot(e,k,args.seed,"best") for e,k in specs]
        build_trace(ctx,"dense")
        for snapshot in snapshots:
            build_trace(ctx,"natural_R2",experts=snapshot.experts,k=snapshot.k)
            analyze_snapshot(ctx,snapshot)
            full_snapshot(ctx,snapshot)
        aggregate(ctx,snapshots,args.run_id,figures=not args.no_figures)
        log(f"P2_SMOKE_OK run_id={args.run_id} samples_per_task={args.samples_per_task}")


if __name__ == "__main__":
    main()
