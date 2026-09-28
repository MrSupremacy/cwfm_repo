# Phase D/F0 运行手册

## 1. 路径与环境

代码库：/mnt/luoyulin_code/fanxuankai/task6_phaseD_reproduction

大产物：/mnt/luoyulin_ckpt/fanxuankai/task6_phaseD

仓库不创建 .venv，默认复用已经通过 Phase A 验收的 task6_reproduction/.venv；也可用 PYTHON 显式指定同一训练解释器。复制
configs/local/server.example.yaml 为 configs/local/server.yaml 后再运行。

## 2. 输入资产

inputs 下需要：

- pretrained/t5-small：未经任务微调的原始 ReLU T5-small；
- data/glue/{sst2,mnli,qnli,qqp}：固定 parquet；
- dense_mt/best：Dense 审核后由脚本导出；
- expert_splits/E64_seed1：已完成的 E64 split；
- expert_splits/E128_seed1、E256_seed1：必须从本仓库导出的同一 Dense-MT best 独立生成，禁止复用 Phase B 的 task-specific 权重或 labels。

下载数据：

~~~bash
bash scripts/05_data/download.sh --suite configs/suites/dense_mt.yaml --local configs/local/server.yaml --run-id data01
~~~

如果服务器下载慢，可在本地用同一命令下载到临时 input_root，完整复制 inputs/data/glue 到共享 code 路径并核验文件 hash，随后删除本地临时副本。

## 3. Dense-MT 阶段

只做环境和资产核验：

~~~bash
bash scripts/00_preflight/dense.sh --suite configs/suites/dense_mt.yaml --local configs/local/server.yaml --run-id dense01
~~~

首次在新机器上运行一次正式 batch=256 的真实前向/反向显存检查；它不创建 checkpoint，也不执行 optimizer step：

~~~bash
bash scripts/00_preflight/dense_batch_smoke.sh --suite configs/suites/dense_mt.yaml --local configs/local/server.yaml --run-id dense01
~~~

单卡完成训练、四域完整验证、macro-best 选择和 Dense 图表：

~~~bash
CUDA_VISIBLE_DEVICES=0 PYTHON=python bash scripts/90_formal/run_dense_single_gpu.sh dense01
~~~

产物位于 output_root/dense_mt/runs/{train,evaluation}/dense01 和 output_root/dense_mt/results/dense01。该命令在 DENSE_MT_REVIEW_READY 停止，不自动继续 split。检查四域 native 轨迹、macro/worst、greedy invalid rate、loss/LR/exposure 和所选 best 后，显式导出：

~~~bash
bash scripts/10_dense/export_best.sh --suite configs/suites/dense_mt.yaml --local configs/local/server.yaml --run-id dense01
~~~

导出目标为共享 code 下 inputs/dense_mt/best；目标存在时拒绝覆盖。

## 4. Split 阶段

使用旧版 k-means-constrained==0.9.1 和 random_state=1，为同一 Dense-MT best 分别生成 E128/E256：

~~~bash
PYTHON=/root/workspace/task4_reproduction/.venv/bin/python bash scripts/90_formal/run_split_cpu.sh
~~~

如果在新开发机运行，所选 Python 环境必须精确满足 split 配置中的版本。split 读取已导出的 Dense-MT best，输出到共享 code 下 `inputs/expert_splits/E128_seed1` 与 `E256_seed1`。E128 每 expert 16 个神经元，E256 每 expert 8 个；15% 档只改变 top-k，不另生成 split。

## 5. Routed F0 阶段

补测矩阵核验应输出 training_runs=90、static_states=6、router_checkpoints=990、routed_states=996：

~~~bash
bash scripts/run.sh matrix --suite configs/suites/phase_d_f0.yaml --local configs/local/server.yaml --run-id routed01
~~~

全流程：

~~~bash
PYTHON=python bash scripts/90_formal/run_routed_stage.sh routed02
~~~

该入口只展开新的 E128/E256 condition，依次 prepare/preflight、八卡 condition 级训练、A、best 选择、B/C/D、metric、汇总、制表和制图。每个 condition 仍为单卡单进程。汇总会只读导入 `routed02` 已有 E64 normalized/seed-baseline 数据；`data/` 统一保存三档 E，`tables/` 与 `figures/` 下分别使用 `E64/`、`E128/`、`E256/`。旧 E64 成品会复制到 E64 子目录，不重新打开旧 checkpoint。

单条件调试或恢复：

~~~bash
bash scripts/30_router/train.sh --suite configs/suites/phase_d_f0.yaml --local configs/local/server.yaml --run-id routed02 --experts 128 --arm G4 --k 19 --seed 0
bash scripts/30_router/train.sh --suite configs/suites/phase_d_f0.yaml --local configs/local/server.yaml --run-id routed02 --experts 128 --arm G4 --k 19 --seed 0 --resume step_1320
~~~

## 6. 结果口径

- Dense-MT：四域 native/greedy、macro、worst-domain 的 11 点轨迹及 best。
- Routed：performance、CV、churn、oracle overlap、activation coverage、encoder-content JS distance 与 expert-domain MI/NMI。
- 所有相对 Dense-MT 和 arm 间性能差只使用 performance-best。
- Final 原始性能仅进入 diagnostics。
- 结果同时输出底层 JSON/CSV、Markdown 表及 300 dpi PNG/PDF。
