from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from task8_p01.common.io import write_json
from task8_p2.config import code_id, output_root, protocol_id, validate_run_id
from task8_p2.progress import log


def launch(config, suite, local, run_id, gpu_count=8, local_only=False, full_only=False):
    import torch
    validate_run_id(run_id)
    if gpu_count != 8 or torch.cuda.device_count() < 8:
        raise ValueError(f"Formal launch requires eight visible CUDA devices; found {torch.cuda.device_count()}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else [str(i) for i in range(torch.cuda.device_count())]
    devices = devices[:8]
    if len(devices) != 8 or len(set(devices)) != 8:
        raise ValueError("CUDA_VISIBLE_DEVICES does not identify eight distinct devices")
    log(f"P2_EIGHT_GPU_PREFLIGHT_OK devices={','.join(devices)}")
    logs = output_root(config)/"logs"/run_id
    logs.mkdir(parents=True, exist_ok=True)
    write_json(logs/"launch_manifest.json", {"run_id": run_id, "code_hash": code_id(), "protocol": protocol_id(config),
               "devices": devices, "local_only": local_only, "full_only": full_only,
               "suite": str(Path(suite).resolve()), "local": str(Path(local).resolve()) if local else None})
    common = ["--suite", str(Path(suite).resolve())]
    if local:
        common += ["--local", str(Path(local).resolve())]

    def stage(label, jobs):
        children, readers = [], []
        try:
            for shard, arguments in enumerate(jobs):
                logfile = logs/f"{label}_gpu_{shard}.log"
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": devices[shard],
                       "TASK8_WORKER": f"{label}/gpu{shard}", "PYTHONUNBUFFERED": "1"}
                process = subprocess.Popen([sys.executable,"-u","-m","task8_p2",*arguments,*common],
                                           env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                children.append(process)
                def read_output(proc=process, destination=logfile):
                    with destination.open("a",encoding="utf-8",newline="\n") as stream:
                        for line in proc.stdout:
                            stream.write(line)
                            stream.flush()
                            print(line.rstrip(),flush=True)
                thread = threading.Thread(target=read_output,daemon=True)
                thread.start()
                readers.append(thread)
            last = time.monotonic()
            while True:
                states = [p.poll() for p in children]
                if any(state is not None and state != 0 for state in states):
                    raise RuntimeError(f"{label} worker failed; exit codes={states}; logs={logs}")
                if all(state == 0 for state in states):
                    break
                if time.monotonic()-last >= config["p2"]["heartbeat_seconds"]:
                    log(f"STAGE {label} workers_complete={sum(s==0 for s in states)}/{len(states)} logs={logs}")
                    last = time.monotonic()
                time.sleep(.5)
            log(f"STAGE_COMPLETE {label}")
        finally:
            for p in children:
                if p.poll() is None:
                    p.terminate()
            for p in children:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            for thread in readers:
                thread.join(timeout=10)

    stage("preflight",[["preflight"]])
    if not full_only:
        stage("prepare",[["prepare"]])
        stage("local",[["worker","--stage","local","--shard",str(s),"--shards","8"] for s in range(8)])
    if not local_only:
        stage("full",[["worker","--stage","full","--shard",str(s),"--shards","8"] for s in range(8)])
    args = ["results","--run-id",run_id]
    if local_only:
        args += ["--local-only"]
    stage("results",[args])
    log(f"TASK8_P2_N_COMPLETE run_id={run_id} results={output_root(config)/'results'/run_id}")
