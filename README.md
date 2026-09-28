# cwfm_repo

Task 5、Task 6、Task 8 的 T5-Small / Intrinsic MoE 路由实验代码备份。备份日期：**2026-09-28**。以经过实验使用的远程工作目录为权威来源，保留五个独立 `*_reproduction` 代码库的边界、包名与内部结构。

这是**轻量源码快照**：包含代码、配置、脚本、测试、数据 schema、运行手册及环境档案；**不包含数据集、预训练模型、dense/router checkpoint、expert split 数组、正式结果或旧报告**。仅 clone 本库不能立即复现完整实验；须先找回下文列出的外部资产。

## 目录与研究关系

```text
cwfm_repo/
├── README.md
├── ENVIRONMENT.md                 实测完整依赖、系统/CUDA、历史聚类环境及恢复方法
├── BACKUP_AUDIT.md                一致性、隐私、排除内容和验证边界
├── environment/                  三套 Python 包清单、系统包清单、split 版本证据
├── backup/                       原始源码 SHA256/权限、差异清单、排除清单、测试汇总
├── task5_reproduction/            单任务 router-only 基线及 Phase A/F0 补测
├── task6_reproduction/            Phase A/F1 full-finetuning
├── task6_phaseB_reproduction/      Phase B/F0 expert-count scaling
├── task6_phaseD_reproduction/      Dense-MT + 多任务 Phase D/F0
└── task8_reproduction/            P0/P1 与 P2-N 机制诊断
```

五个代码库继续并列：它们是独立 Python 项目，Task8 按路径复用 Phase D 的源代码。保留这一层级可减少已有脚本和跨库路径的迁移改动；按 Task 分组由本页导航表达，不拆分或合并实现。原 Task4 代码不在本次范围内，其输出资产仍是部分实验的前置条件。

| 代码库 | 实际实现与用途 | 主要前置依赖 | 入口 |
|---|---|---|---|
| [Task5](task5_reproduction/README.md) | SST-2/MNLI，E64；R1–R4、R4-R2Init、G0–G4 的 frozen-backbone 路由对照；另含 R2-soft/R4-hard 的 F0 补测；训练→capture→指标→汇总/绘图 | Task4 导出的 task-specific dense、tokenizer、GLUE、balanced splits | [RUNBOOK](task5_reproduction/RUNBOOK.md)，`python -m task5` |
| [Task6 A](task6_reproduction/README.md) | 八个稀疏臂 fullFT，外加匹配的 Dense-fullFT；SST-2/MNLI、E64、三 seeds | Task5 的输入资产；跨范式比较另需 F0 汇总 | [RUNBOOK](task6_reproduction/RUNBOOK.md)，`python -m task6` |
| [Task6 B](task6_phaseB_reproduction/README.md) | F0、E128/E256、R2/R4o/R4d/G1/G2-0.001/G4；240 个训练 run；E64 基线只读导入 | 单任务 dense/data、高 E splits、Phase A/F0 的 E64 汇总 | [RUNBOOK](task6_phaseB_reproduction/RUNBOOK.md)，`python -m task6_phaseb` |
| [Task6 D](task6_phaseD_reproduction/README.md) | 四任务 Dense-MT fullFT；审核导出 best 后，E64/E128/E256 六臂 F0；包含负载、churn、overlap/coverage、domain JS/MI | 原始 pretrained T5-Small、SST-2/MNLI-matched/QNLI/QQP；后续需导出的 Dense-MT 和其 splits | [RUNBOOK](task6_phaseD_reproduction/RUNBOOK.md)，`python -m task6_phased` |
| [Task8](task8_reproduction/README.md) | P0 身份/端点检查，P1 selector×aggregation 四格与 replay，P2-N 的 N2-a/N1/N3 数值及温度诊断；本实现不做新训练 | Phase D 源码、Dense-MT、split、四任务数据、R4d init/best、P1 panel/catalog/端点产物 | [RUNBOOK](task8_reproduction/RUNBOOK.md)、[P2_RUNBOOK](task8_reproduction/P2_RUNBOOK.md)，`python -m task8_p01` / `python -m task8_p2` |

研究线索：Task5 建立路由对照 → Task6 检查 fullFT、专家粒度与多任务 setting → Task8 拆解 R4d/R2 的选择、加权及数值因素。不要把 handout 中的全部计划当作已实现：本备份不宣称完成 Task6 Phase C、D-F1/F2、LODO，或 Task8 P3/P4、P2-O/N2-b。

这些代码以 dense-mask 为主验证路由方法，不能据此声称有实际稀疏 kernel 加速。历史报告中的科学结论没有在这次备份中重新实验验证。

## 环境与开始使用

先读 [ENVIRONMENT.md](ENVIRONMENT.md)。当前实测主环境是 Linux / Python 3.12.3 / PyTorch 2.11.0+cu126 / CUDA 12.6。`/opt/task5-venv` 与 Task6 的 `.venv` 都继承系统 site-packages；完整依赖不能只看各库 `pyproject.toml` 或旧 Docker requirements。

在已经恢复依赖的 Linux 环境中，从 monorepo 根目录执行：

```bash
export CWFM_ROOT="$(pwd)"
export PYTHON=/opt/task5-venv/bin/python  # 新机可改为已准备好的解释器
export TASK6_ROOT="$CWFM_ROOT/task6_phaseD_reproduction"

"$PYTHON" -m pip install --no-deps -e ./task5_reproduction
"$PYTHON" -m pip install --no-deps -e ./task6_reproduction
"$PYTHON" -m pip install --no-deps -e ./task6_phaseB_reproduction
"$PYTHON" -m pip install --no-deps -e ./task6_phaseD_reproduction
"$PYTHON" -m pip install --no-deps -e ./task8_reproduction
"$PYTHON" -m pip check
```

代码配置按原远程文件原样保留，**包括审计通过的机器路径配置**；这是总备份对部分旧 README 中“不提交机器配置”说明的有意补充。旧挂载路径是恢复线索，不代表新机器上存在这些目录。迁移前检查：

1. 每个 `configs/local/*.yaml` 内的输入、输出、临时目录及继承关系；不要只改一个 output_root。
2. Task8 的 `execution.task6_repo`、`task6_local`、`TASK6_ROOT`；其 `scripts/05_assets/link_server_assets.sh` 还单独硬编码旧 Phase D 路径，直接执行前须调整。
3. Task8 P2 的 `p1_output_root` 是旧 P1 产物位置，不能误填成 P2 输出；数据和 source hash 不匹配时不能绕过校验复用。
4. Phase D/Task8 启动器的旧 Task6 `.venv` 回退路径；显式指定 PYTHON 可避免选错环境。
5. 正式八卡启动器需要足够 GPU；当前采集开发机只可见一张 A800，不应直接照抄八卡命令。

恢复资产、路径和环境后，按各库 RUNBOOK 的 prepare/preflight/Phase 0/smoke 顺序执行。Phase D 的 Dense 审核导出门保持不变。

## 本次明确缺失的资产

| 所属 | 未下载、未上传的内容 | 恢复时的用途 |
|---|---|---|
| Task5 | 整个 `inputs/`，包括 dense、tokenizer、GLUE parquet、expert_splits、provenance；`artifacts/`、`runs/`、`tmp/` 与外部结果 | 训练及 A–E capture；共激活、probe、正式 resume |
| Task6 A | 整个 `inputs/`、本地 `.venv/`、tmp；外部 fullFT/checkpoint/capture/结果 | fullFT、恢复状态、F0/F1 对照 |
| Task6 B | 整个 `inputs/`，含高 E split 和可能的链接；外部结果和 E64/F0 导入数据 | 新 E 实验与跨 E 汇总 |
| Task6 D | 整个 `inputs/`：pretrained、四任务 data、Dense-MT best、E64/E128/E256 splits；dense02/routed02 的外部结果与训练状态 | Dense-MT、router-only、Task8 前置资产 |
| Task8 | 整个 `inputs/`，包括 dense/data/split 链接、r4d_init/r4d_best、r4d_catalog；外部 P0/P1/P2 captures、hidden cache、预测与表图 | 身份校验、旧端点复用、P1/P2 重评 |
| 通用 | 缓存、日志、构建元数据、历史报告和图表、过期人工备份、Git 历史、镜像层与离线 wheel | 本仓库仅保留源码快照及可读环境记录 |

**不自动解引用任何被排除的符号链接**，避免把共享存储上的大资产带进 Git。少量 split 环境版本是从远端 manifest 提取的元数据，保存在 environment 中；模型和数组本身没有下载。

本地 handout、analysis、feedback、report、学生包及 Task6 preposition 副本仅用于理解，不属于上传来源。本次远端历史报告也作保守排除；旧 README/RUNBOOK 指向它们的链接可能无法打开，详见 [BACKUP_AUDIT.md](BACKUP_AUDIT.md)。

## 一致性与验证

- 528 个远程文件，合计 1,173,468 bytes；下载后逐文件 SHA256 与审计时远端一致，未修改原实验实现。
- 本地与远端共同文件无实质差异；Task5 的 100 个文件仅换行不同。远端特有的当前 B/D 部署配置已保留；过期副本与报告未保留。
- 原有无真实资产测试 173 项通过，零跳过：Task5 76、Task6 A 14、B 34、D 24、Task8 25。
- 297 个 Python 文件语法解析、103 个 Bash 文件语法检查通过。
- 此次未重跑正式训练、完整 validation 或 CUDA smoke；完整可运行性仍依赖缺失资产与新机环境验收。

参见 [源码指纹](backup/source-manifest.json)、[逐文件差异](backup/local-remote-comparison.json)、[排除清单](backup/exclusions.json)、[验证记录](backup/validation.json)。没有引入原工作目录的 Git 历史；本库从审计后的快照建立新历史。未新增开源授权声明，仓库按私人研究备份使用。
