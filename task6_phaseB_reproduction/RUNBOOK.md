# Phase B/F0 运行手册

## 1. 配置输入

代码库不复制 Task 5 的 dense checkpoint 和数据。服务器上复制 `configs/local/server.example.yaml` 为 `configs/local/server.yaml`，填写：

- Task 5 的 SST-2/MNLI dense checkpoint 与本地数据路径；
- 本仓库 `inputs/expert_splits/` 下四套 E128/E256 split；
- Phase A/F0 `normalized/.../metrics.json`、该文件 SHA256 与 source run id；
- `/mnt/luoyulin_ckpt/fanxuankai/task6_phaseB` 输出根目录。

本仓库使用当前激活的训练环境，不创建或依赖仓库内 `.venv`。如 Python 命令不是 `python`，启动时设置 `PYTHON=/实际/python`。

## 2. 配置和输入核验

```bash
PYTHON=python bash scripts/run.sh matrix --suite configs/suites/phase_b_f0.yaml --local configs/local/server.yaml --list
PYTHON=python bash scripts/00_preflight/run.sh --suite configs/suites/phase_b_f0.yaml --local configs/local/server.yaml --run-id phaseb01
```

第一条必须报告 `training_runs=240`、`static_states=16`、`routed_states=2656`。preflight 会验证 dense、数据、四套 split 的路径、hash、层数、容量和 checkpoint 绑定。

## 3. 一条命令运行完整正式实验

```bash
PYTHON=python bash scripts/90_formal/run_all_8gpu.sh phaseb01
```

启动器写死 8 卡，并依次执行 preflight、E64 结果导入、prepare、validate、240 个训练 run、A 采集、best 选择、B/C/D 采集、全部约定 metric、合并汇总和 PNG/PDF/CSV/Markdown 输出。E64 阶段只导入已有统计结果，不训练、不采集也不复制权重。每个 GPU 进程只看到一张卡，单 condition 的 `world_size` 始终为 1。

## 4. 分阶段运行

所有阶段共用以下参数：

```bash
COMMON="--suite configs/suites/phase_b_f0.yaml --local configs/local/server.yaml --run-id phaseb01"
```

```bash
bash scripts/55_import_e64/run.sh $COMMON
bash scripts/10_prepare/run.sh $COMMON
bash scripts/20_validate/run.sh $COMMON
bash scripts/30_train/run.sh $COMMON --shard-count 8 --shard-index 0
bash scripts/40_capture/run.sh $COMMON --part A --shard-count 8 --shard-index 0
bash scripts/40_capture/run.sh $COMMON --part select-best
bash scripts/40_capture/run.sh $COMMON --part diagnostics --shard-count 8 --shard-index 0
bash scripts/50_metrics/run.sh $COMMON --metric all --shard-count 8 --shard-index 0
bash scripts/60_aggregate/run.sh $COMMON
bash scripts/70_tables/run.sh $COMMON
bash scripts/80_figures/run.sh $COMMON
```

`shard-index` 需分别运行 0–7。也可用 `--task`、`--experts`、`--arm`、`--k` 和 `--seed` 选择一个 condition。R2 不进入 train。

## 5. 中断恢复

训练只允许在完整 epoch checkpoint 上恢复。选定唯一 condition，并传 checkpoint 目录名：

```bash
bash scripts/30_train/run.sh $COMMON --task sst2 --experts 256 --arm G4 --k 51 --seed 0 --resume step_1534
```

若 capture 已完整，可加 `--skip-complete` 核验后复用。训练和 capture 已完成后，可用：

```bash
PYTHON=python METRIC_SHARDS=8 bash scripts/90_formal/finalize_cpu.sh phaseb01
```

离线重算 metric、汇总、表格和图片，不修改 checkpoint/capture。

## 6. 输出

- `runs/train/`：router-only checkpoint；
- `runs/capture/`：A/B/C/D；
- `runs/metrics/`：performance、CV、churn、overlap/adjusted overlap、coverage、oracle resolution；
- `results/data/`：E128/E256 明细统一存储，并保留聚合所需的 E64 导入行、paired G1 gap 与相对 E64 gap change；
- `results/tables/{128_experts,256_experts}/`：每个 E 各自的 main、diagnostics、appendix 表；
- `results/tables/expert_comparison/`：E64/E128/E256 matched-ratio 对比表，不生成跨 E 图；
- `results/figures/{128_experts,256_experts}/`：每个 E 各自的 main、diagnostics、appendix PNG/PDF。
