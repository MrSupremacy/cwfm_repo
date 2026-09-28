# Task 5 全流程唯一运行手册

**2026-09 补测入口**：已完成 `formal20260830a`、只补 R2-soft/R4-hard 时直接看 **§17**，不重跑前面的部署、smoke、prepare 或旧臂。仍是 F0（冻结 backbone），不是 fullFT。

这是本代码库唯一需要照着执行的运行文档。顺序固定为：

```text
部署与配置 → 环境/输入检查 → smoke全链路 → 正式prepare与Phase 0
→ 全量训练 → A选best → B/C/D/E全量采集 → 六组metric → 汇总 → 表格与图片
```

不要把smoke产物当正式结果，也不要在训练过程中顺便采集正式指标。每一阶段必须等前一阶段的所有进程成功退出后再开始。

## 1. 只填写一次的变量

在服务器打开终端，执行：

```bash
cd "<task5_reproduction_absolute_path>"

export PYTHON="<python_executable_absolute_path>"
export SMOKE_ID="<new_smoke_run_id>"
export RUN_ID="<new_formal_run_id>"
export OUTPUT_ROOT="<formal_output_root_absolute_path>"
export GPU_ID="<physical_gpu_id>"
export SHARD_COUNT="<number_of_parallel_gpu_processes>"
export SHARD_INDEX="<this_process_shard_index>"

export MAIN_SUITE="configs/suites/main.yaml"
export SMOKE_SUITE="configs/suites/smoke.yaml"
export SMOKE_LOCAL="configs/local/server.yaml"
export FORMAL_LOCAL="configs/local/formal.yaml"
```

需要填写：

- `<task5_reproduction_absolute_path>`：本仓库在服务器上的绝对路径。
- `<python_executable_absolute_path>`：准备运行实验的Python，例如conda/venv里的`bin/python`。
- `<new_smoke_run_id>`：本次smoke的全新标识，例如`smoke03`。
- `<new_formal_run_id>`：整套正式实验统一使用的全新标识，例如`main01`。
- `<formal_output_root_absolute_path>`：正式checkpoint、capture和结果的独立大容量目录。
- `<physical_gpu_id>`：该进程使用的物理GPU编号；脚本内部仍把它视为`cuda:0`。
- `<number_of_parallel_gpu_processes>`：同时使用几张GPU；单卡填`1`。
- `<this_process_shard_index>`：当前终端负责的分片编号，范围为`0`到`SHARD_COUNT-1`；单卡填`0`。

每开一个新的训练/采集终端，都要重新执行本节变量。多卡时每个终端使用不同`GPU_ID`和`SHARD_INDEX`，
但必须使用相同的`RUN_ID`、`SHARD_COUNT`、suite、local配置和源码。

## 2. 搬迁代码与配置

必须复制整个`task5_reproduction/`目录。仅执行Git clone不够，因为`inputs/`和实际的local配置默认不进Git。
完整交付应包含：

```text
inputs/dense/{sst2,mnli}/
inputs/expert_splits/{sst2,mnli}/
inputs/datasets/glue/{sst2,mnli}/
inputs/provenance.json
configs/local/server.yaml
```

正式输出不要和smoke混放。先创建正式local配置：

```bash
cp configs/local/formal.example.yaml configs/local/formal.yaml
mkdir -p "$OUTPUT_ROOT"
```

然后编辑`configs/local/formal.yaml`，将：

```yaml
output_root: /SET_ME/task5_formal_outputs
```

改为：

```yaml
output_root: "<formal_output_root_absolute_path>"
```

这里的`<formal_output_root_absolute_path>`必须与第一节的`OUTPUT_ROOT`相同，并保证有足够空间和写权限。
不要给`configs/local/server.yaml`增加正式output root；smoke依靠它把产物留在仓库的`tmp/smoke/`。

如果当前交付已经包含上述inputs，不执行任何导入。只有在原Task4服务器上、且inputs确实缺失时才运行：

```bash
bash scripts/10_prepare/import_task4_assets.sh \
  --source-root "<task4_reproduction_absolute_path>" \
  --copy
```

需要填写：`<task4_reproduction_absolute_path>`是旧Task4工程的绝对路径。正式新服务器不依赖Task4目录。

## 3. 安装并核验环境

先让集群提供与驱动兼容的PyTorch CUDA环境，再安装本项目。不要为了本项目降级或改装共享环境。

```bash
"$PYTHON" -m pip install -c configs/local/server-tested.txt -e '.[model]'

"$PYTHON" --version
nvidia-smi
"$PYTHON" -c 'import torch, transformers, datasets, pyarrow, yaml; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available()); print("transformers", transformers.__version__, "datasets", datasets.__version__, "pyarrow", pyarrow.__version__, "yaml", yaml.__version__)'
```

必须确认：

- `torch.cuda.is_available()`为`True`；目标GPU和驱动可见。
- PyArrow为25.0.1，PyYAML为6.0.3。
- 主要版本与`configs/local/server-tested.txt`一致。
- 正式目录磁盘容量、inode和写权限充足。

## 4. 配置、代码和真实输入检查

先检查完整main矩阵，不访问模型：

```bash
bash scripts/00_preflight/run.sh \
  --suite "$MAIN_SUITE" \
  --config-only
```

应看到：242个condition、192个training run、2162个logical state。

再运行完整测试。正式Linux模型环境应全部通过，不应出现依赖缺失类skip：

```bash
bash scripts/20_validate/test_local.sh \
  --report tmp/local-tests/before_formal.json
```

最后核验真实dense、expert split和四个Parquet文件：

```bash
bash scripts/00_preflight/run.sh \
  --suite "$MAIN_SUITE" \
  --local "$FORMAL_LOCAL" \
  --run-id "$RUN_ID"
```

任何失败都先停止，不要启动训练。

## 5. 完整smoke test

smoke使用每任务512条训练样本、32条validation、`k=13`、`seed=0`，但仍贯通训练、A–E、全部metric、汇总和绘图。

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/run.sh smoke \
  --suite "$SMOKE_SUITE" \
  --local "$SMOKE_LOCAL" \
  --run-id "$SMOKE_ID"
```

成功后运行只读审计：

```bash
CUDA_VISIBLE_DEVICES="" bash scripts/20_validate/audit_smoke.sh \
  --suite "$SMOKE_SUITE" \
  --local "$SMOKE_LOCAL" \
  --run-id "$SMOKE_ID" \
  --report tmp/server-tests/smoke_audit.json
```

审计必须以`passed: true`结束。smoke内已经包含Phase 0，但它不是正式main的prepare产物。
可在另一个终端用`watch -n 1 nvidia-smi`观察显存；若需要调整batch/chunk，必须在正式运行前固定配置并用新smoke ID重跑。

## 6. 正式prepare与Phase 0

正式main必须重新prepare：它会生成固定probe、静态router快照以及train/validation预处理缓存。

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/10_prepare/run.sh \
  --suite "$MAIN_SUITE" \
  --local "$FORMAL_LOCAL" \
  --run-id "$RUN_ID"
```

然后对两任务、全部路由类型和四档预算执行dense等价性/有限值验证：

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/20_validate/run.sh \
  --suite "$MAIN_SUITE" \
  --local "$FORMAL_LOCAL" \
  --run-id "$RUN_ID"
```

两任务都必须输出`Phase0 passed`。R3保持RMS+`1.0e-6`；R2/R3集合一致率只记录，不作为门槛。

## 7. 全量训练

每个进程只使用一张GPU，不运行DDP。单卡时`SHARD_COUNT=1, SHARD_INDEX=0`，执行一次即可。
多卡时在每张GPU对应的终端分别执行同一命令，仅改变`GPU_ID`和`SHARD_INDEX`：

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/30_train/run.sh \
  --suite "$MAIN_SUITE" \
  --local "$FORMAL_LOCAL" \
  --run-id "$RUN_ID" \
  --shard-count "$SHARD_COUNT" \
  --shard-index "$SHARD_INDEX"
```

必须等全部分片以退出码0结束。正式矩阵包含192个训练run，每个保存step 0和epoch 1–10，共2112个训练状态。
训练阶段不采集正式validation/probe数据。

可核对checkpoint完整标记数量：

```bash
find "$OUTPUT_ROOT/runs/train" \
  -path "*/$RUN_ID/checkpoints/*/complete.json" | wc -l
```

预期为`2112`。数量不符时不要开始A采集。

## 8. Capture A：完整validation预测

与训练相同，可在每张GPU终端按condition分片。所有训练状态、静态arm和dense都会采集A：

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/40_capture/run.sh \
  --part A \
  --suite "$MAIN_SUITE" \
  --local "$FORMAL_LOCAL" \
  --run-id "$RUN_ID" \
  --shard-count "$SHARD_COUNT" \
  --shard-index "$SHARD_INDEX"
```

全部分片成功后核对：

```bash
find "$OUTPUT_ROOT/runs/capture/validation" \
  -path "*/$RUN_ID/*/A/complete.json" | wc -l
```

预期为`2162`。A使用SST-2完整872条validation、MNLI完整9815条validation_matched。

## 9. 离线选择best

该步骤只读取A，不加载模型、不占GPU。每个`task × arm × variant × ratio × seed`独立选validation accuracy最高的状态，
同分取更早step；不会跨seed、ratio或G2档位挑选。

```bash
CUDA_VISIBLE_DEVICES="" bash scripts/40_capture/run.sh \
  --part select-best \
  --suite "$MAIN_SUITE" \
  --local "$FORMAL_LOCAL" \
  --run-id "$RUN_ID"
```

## 10. Capture B/C/D：负载与probe

best选完后再执行diagnostics。它会采集：best/final/static的完整validation负载B；全部所需状态的固定probe选择C；
R4与R4-R2Init全轨迹以及其他arm的best/final/static完整激活和D。best与final相同时自动复用同一状态。

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/40_capture/run.sh \
  --part diagnostics \
  --suite "$MAIN_SUITE" \
  --local "$FORMAL_LOCAL" \
  --run-id "$RUN_ID" \
  --shard-count "$SHARD_COUNT" \
  --shard-index "$SHARD_INDEX"
```

必须等所有分片完成。

## 11. Capture E：dense共激活

E按任务共享，只能每任务运行一次，不能按arm或condition分片。单卡顺序运行：

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/40_capture/run.sh \
  --part E --task sst2 \
  --suite "$MAIN_SUITE" \
  --local "$FORMAL_LOCAL" \
  --run-id "$RUN_ID"

CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/40_capture/run.sh \
  --part E --task mnli \
  --suite "$MAIN_SUITE" \
  --local "$FORMAL_LOCAL" \
  --run-id "$RUN_ID"
```

完成后应有`2任务 × 12层 = 24`份共激活矩阵。至此所有昂贵模型前向结束，后续阶段不再加载模型。

## 12. 六组metric

以下全部是离线计算，建议逐个运行，便于定位问题：

```bash
CUDA_VISIBLE_DEVICES="" bash scripts/50_metrics/run.sh --metric performance \
  --suite "$MAIN_SUITE" --local "$FORMAL_LOCAL" --run-id "$RUN_ID"

CUDA_VISIBLE_DEVICES="" bash scripts/50_metrics/run.sh --metric load_balance \
  --suite "$MAIN_SUITE" --local "$FORMAL_LOCAL" --run-id "$RUN_ID"

CUDA_VISIBLE_DEVICES="" bash scripts/50_metrics/run.sh --metric churn \
  --suite "$MAIN_SUITE" --local "$FORMAL_LOCAL" --run-id "$RUN_ID"

CUDA_VISIBLE_DEVICES="" bash scripts/50_metrics/run.sh --metric oracle_overlap \
  --suite "$MAIN_SUITE" --local "$FORMAL_LOCAL" --run-id "$RUN_ID"

CUDA_VISIBLE_DEVICES="" bash scripts/50_metrics/run.sh --metric activation_coverage \
  --suite "$MAIN_SUITE" --local "$FORMAL_LOCAL" --run-id "$RUN_ID"

CUDA_VISIBLE_DEVICES="" bash scripts/50_metrics/run.sh --metric coactivation_consistency \
  --suite "$MAIN_SUITE" --local "$FORMAL_LOCAL" --run-id "$RUN_ID"
```

六组分别得到：任务性能/relative performance、负载均衡、路由churn、oracle overlap、activation coverage、
coactivation consistency。只有六组都完成后才能统一汇总。

## 13. 汇总、表格与图片

汇总会严格检查condition、状态、seed、层和metric是否齐全；缺失时直接失败，不会静默丢行。

```bash
CUDA_VISIBLE_DEVICES="" bash scripts/60_aggregate/run.sh \
  --suite "$MAIN_SUITE" --local "$FORMAL_LOCAL" --run-id "$RUN_ID"

CUDA_VISIBLE_DEVICES="" bash scripts/70_tables/run.sh \
  --suite "$MAIN_SUITE" --local "$FORMAL_LOCAL" --run-id "$RUN_ID"

CUDA_VISIBLE_DEVICES="" bash scripts/80_figures/run.sh \
  --suite "$MAIN_SUITE" --local "$FORMAL_LOCAL" --run-id "$RUN_ID"
```

主要产物位于：

```text
<formal_output_root_absolute_path>/results/data/normalized/<formal_run_id>/
<formal_output_root_absolute_path>/results/data/aggregated/<formal_run_id>/
<formal_output_root_absolute_path>/results/tables/main/<formal_run_id>/
<formal_output_root_absolute_path>/results/tables/diagnostics/<formal_run_id>/
<formal_output_root_absolute_path>/results/tables/appendix/<formal_run_id>/
<formal_output_root_absolute_path>/results/figures/main/<formal_run_id>/
<formal_output_root_absolute_path>/results/figures/diagnostics/<formal_run_id>/
<formal_output_root_absolute_path>/results/figures/appendix/<formal_run_id>/
```

这里的两个占位符分别填写第一节的`OUTPUT_ROOT`和`RUN_ID`。图片同时输出300dpi PNG与PDF；主表同时保留完整精度CSV。
相对性能主图为`sst2_performance_relative_performance.*`与`mnli_performance_relative_performance.*`。

## 14. 中断、重跑与禁止事项

- 不同源码、配置、输入或实验协议使用新的`RUN_ID`；不要把旧capture冒充新协议产物。
- prepare、validate和train产物默认不覆盖。不要直接重跑同一已完成run。
- capture中断后，先保留错误日志；把对应未完成目录移到备份位置，再用同一命令加`--skip-complete`。该参数会校验并复用完整capture。
- metric、aggregate、tables和figures属于派生产物，可在相同原始capture上重新计算。
- 不要因OOM自动改变batch、梯度累积、AMP或TF32；停止运行，统一修改配置，重新smoke并使用新run ID。
- 不要多个进程写同一shard，不要在所有训练完成前开始A，不要在所有A完成前选best，不要重复运行共享E。

单个训练condition从epoch边界续训时：

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/30_train/run.sh \
  --suite "$MAIN_SUITE" \
  --local "$FORMAL_LOCAL" \
  --run-id "$RUN_ID" \
  --task "<task>" \
  --arm "<arm>" \
  --variant "<variant>" \
  --k "<k>" \
  --seed "<seed>" \
  --resume "<checkpoint_directory_name>"
```

需要填写：

- `<task>`：`sst2`或`mnli`。
- `<arm>`：可训练arm之一，`R4/R4-R2Init/G1/G2/G3/G4`。
- `<variant>`：通常为`default`；G2为`aux_0.001/aux_0.01/aux_0.1`之一。
- `<k>`：`6/13/19/26`之一。
- `<seed>`：`0/1/2`之一。
- `<checkpoint_directory_name>`：该condition已有的epoch边界目录名，例如`step_264`，不是任意外部路径。

续训只修复这一个condition；之后仍要确认整个192-run矩阵完整。若同一分片已有其他已完成run，不要直接从头重放整个训练分片。

## 15. 最终完成判据

正式实验只有同时满足以下条件才算工程流程完成：

1. 192个训练run、2112个训练状态完整。
2. A包含2162个状态，best选择全部生成。
3. B/C/D与两任务E均完成，E共24层矩阵。
4. 六组metric命令全部成功。
5. aggregate没有缺失condition/seed/layer报错。
6. tables和figures均成功，PNG/PDF成对存在。
7. 原始capture保留；最终数值读取Parquet/JSON/CSV，不从终端日志手工拼接。

这些结果仍是同一validation集合上选best并报告的描述性结果，不应表述为独立test-set结论或统计显著性证明。

## 16. 在既有 formal20260830a 上补充 R4-R2Init

这一节只用于已经完成的旧主矩阵`formal20260830a`。它不重做旧arm，不重采dense A或共享E。新arm仍使用当前新protocol；旧probe/static/E与旧汇总先按base protocol和输入哈希校验。最终aggregate写组合视图前，会把旧normalized/aggregated/paired结果保存到：

```text
<formal_output_root_absolute_path>/results/data/extension_base/formal20260830a/
```

先执行只读检查；它必须显示24个training run、264个logical state，并确认两任务复用的prepared protocol：

```bash
bash scripts/00_preflight/run.sh \
  --suite configs/suites/main.yaml \
  --local configs/local/formal_r4_r2init.yaml \
  --run-id formal20260830a \
  --arm R4-R2Init
```

若希望由单一脚本依次完成训练、A、best、B/C/D、六组metric、合并及重绘，在单机可见GPU编号为`0..N-1`时运行：

```bash
SHARD_COUNT="<single_node_gpu_count>" \
  bash scripts/90_formal/extend_r4_r2init.sh formal20260830a
```

`<single_node_gpu_count>`填写该节点实际用于本实验的GPU数；单卡填`1`，8卡填`8`。这是单机多进程独立condition分片，不是DDP。脚本绝不会运行prepare或共享E；路径由`configs/local/formal_r4_r2init.yaml`固定到原正式输出根。

如需逐阶段人工控制，则给第7–13节对应命令都增加`--arm R4-R2Init`，并把`--local "$FORMAL_LOCAL"`替换为`--local configs/local/formal_r4_r2init.yaml`。capture不得使用`--part all/E`，只依次运行`A`、`select-best`、`diagnostics`；aggregate/tables/figures不带arm筛选，由扩展配置负责安全合并。

## 17. Phase A F0：只补 R2-soft 与 R4-hard

前提：服务器已有 `formal20260830a` 的原主实验及 R4-R2Init v3 完成产物；保持原 `configs/local/formal.yaml` 和 inputs。此次不更新 backbone/expert、dense、不聚类。原 `main.yaml` 矩阵保持不变；补测使用 `phaseA_f0.yaml`，继承完全相同的训练/采集/指标设置，仅改变比较臂列表。扩展配置仅允许新两臂产生记录。

| 新臂 | 本次工作 | 采集 |
|---|---|---|
| R2-soft | 不训练，`2 tasks × 4 k = 8` 个静态状态，路径用 `seed_fixed` | A、B、C+D 各一次；不造三份重复、不测 churn |
| R4-hard | `2 × 4 × 3 = 24` 次 router-only 训练，seeds 0/1/2；`24 × 11 = 264` 点 | 全部 A、C+D；best/final 的 B 去重；十段相邻 churn |

共新增 272 个逻辑状态、272 组 A、272 组 C+D、32–56 组 B；复用既有 dense A/E、probe 和 centroid。R4-hard 初始化与 R4 同 seed/同层配对，训练/评测前向权重为 1，训练反向采用 soft 系数 ST；不是给旧 R4 checkpoint 临时改推理权重。

在**正式八卡节点**的代码库内执行：

```bash
cd <task5_repo_absolute_path>
export PYTHON=/opt/task5-venv/bin/python
GPU_IDS=0,1,2,3,4,5,6,7 SHARD_COUNT=8 \
  bash scripts/90_formal/extend_phaseA_f0.sh formal20260830a all
```

`<task5_repo_absolute_path>` 为该节点上的代码库绝对路径；当前测试服务器是 `/mnt/luoyulin_code/fanxuankai/task5_reproduction`。`PYTHON` 沿用已测试环境；若正式节点的虚拟环境位置不同，只替换解释器位置。GPU_IDS 填本次分配的卡号；八张卡各跑独立条件（24 次训练均分为每卡 3 次），每个模型仍只用一张卡，**不是 DDP，不用 torchrun**。在持久终端会话中运行，避免 SSH 断开中止前台进程。

脚本依次完成：旧数据只读身份核验 → 只训练 R4-hard → 两新臂 A → 各 seed 选 best → B/C+D → 六组离线 metric → 指定八臂汇总/表图。无需用户再跑 smoke/Phase 0；开头的自动核验不是重新做实验。任一步失败即停止后续阶段。所有 capture 候选必须齐全，不能缺 seed/epoch 后静默平均。

原始产物仍是原输出根的同级 arm：

```text
<formal_output_root>/runs/train/<task>/R4-hard/default/k_<k>/seed_<0|1|2>/formal20260830a/
<formal_output_root>/runs/capture/{validation,probe}/<task>/{R2-soft,R4-hard}/default/k_<k>/seed_<fixed|0|1|2>/formal20260830a/
<formal_output_root>/runs/metrics/<metric>/<task>/{R2-soft,R4-hard}/default/k_<k>/seed_<fixed|0|1|2>/formal20260830a/
<formal_output_root>/task6_phaseA_F0_result/{data,tables,figures}/...
```

当前 `<formal_output_root>` 为 `/mnt/luoyulin_ckpt/fanxuankai/task5_reproduction_output`。新图表只比较 R2、R2-soft、R4、R4-R2Init、R4-hard、G1、G2/aux_0.001、G4。`results/` 及旧 arm 的单条件评分不覆盖。旧 normalized 文件以 SHA256 固定，继承其原始精度和 seed 记录；新指标保持旧六组口径，包括共享原 dense 的共激活，**不换成 F1 当前模型的 E**。旧 G1/G2/G4 没有全程 D，不补造 overlap/coverage 轨迹；它们的 best/final 仍参加比较。

可按同一入口分阶段，末尾 `all` 改为 `train`、`capture`、`metrics`、`report`。已有全部 ckpt 后从 `capture` 开始，不再调用 `train`；完整 capture 可经校验跳过；metric 可离线重算。训练目录已存在时脚本拒绝覆盖，不自动重启训练；中途训练失败可用原 `--resume step_<step>` 单条件恢复机制并配上本节 suite/local，再完成剩余条件。不要重命名或删除既有 ckpt 来绕过检查。

只重绘这八臂的表图（不调用模型）：

```bash
bash scripts/90_formal/extend_phaseA_f0.sh formal20260830a report
```

只重写独立结果目录内的派生数据，不改变旧结果。此扩展固定绑定 `formal20260830a`；若换 run、配置或输入，需要重新确认来源，不能仅修改哈希使检查通过。
