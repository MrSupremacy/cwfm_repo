# 开发与实验环境档案

采集日期：2026-09-28；远程观测时间：`2026-09-28T14:37:45Z`。本文件来自开发机只读实测、已备份的运行脚本与共享 split manifest。它记录当前可恢复的环境证据，不把历史环境当作当前实测，不包含任何登录凭据。

## 1. 最重要的恢复结论

通常使用 `export PYTHON=/opt/task5-venv/bin/python`。但该环境通过 `--system-site-packages` 创建，**系统 PyTorch/CUDA 等是实验环境的一部分**。不能仅复制 `/opt/task5-venv` 就认为恢复完整。

另有 `task6_reproduction/.venv`，同样继承系统包；Phase D 和 Task8 脚本未设置 `PYTHON` 时会优先尝试它。它与主环境不是完全相同，差异见下表。两套虚拟环境和系统 Python 的 `pip check` 均通过。

聚类使用另一台旧开发机上的 `/root/workspace/task4_reproduction/.venv/bin/python`。本次连接该开发机超时，无法取得其全部传递依赖、系统或驱动快照；已从当前共享存储的 E64/E128/E256 原始 split manifests 交叉确认六项精确版本。**这部分历史环境仍存在不可恢复的信息缺口**，不可将下方六项约束误称完整锁文件。

## 2. 当前训练开发机：系统、硬件和工具链

| 项目 | 实测结果 |
|---|---|
| 用户空间 | Ubuntu 24.04.4 LTS (Noble) |
| 宿主内核 | Linux 5.15.0-72-generic，x86_64；与容器用户空间版本不同 |
| CPU | 2 × Intel Xeon Platinum 8350C 2.60GHz；64 物理核 / 128 逻辑 CPU |
| 内存 | 1,079,938,793,472 bytes（约 1005.77 GiB）；无 swap |
| 当前可见 GPU | 1 × NVIDIA A800-SXM4-80GB，compute capability 8.0 |
| GPU 显存 | nvidia-smi 81920 MiB；torch 可分配属性为 85,174,583,296 bytes |
| NVIDIA driver | 535.161.08 |
| CUDA toolkit / nvcc | 12.6 / V12.6.85；`CUDA_HOME=/usr/local/cuda` |
| PyTorch | 2.11.0+cu126，CUDA build 12.6 |
| cuDNN | torch 查询整数 91002；Python 包 nvidia-cudnn-cu12 9.10.2.21 |
| Python | 3.12.3，GCC 13.3.0；采集环境使用 `/usr/bin/python3.12` |
| BLAS / CPU backend | PyTorch MKL 2024.2、MKL-DNN 3.10.2、OpenMP 4.5；完整 build config 保存在包快照 JSON |
| GPU 探测 | 三个解释器均 `torch.cuda.is_available() == True` |

当前可见 GPU 只有一张。代码含八卡独立条件分片启动器，**不能据此写成当前机器实测有八张卡**。本次只确认 CUDA 初始化/设备探测及 CPU 测试，未进行完整 CUDA 训练回归。没有做驱动升级或安装依赖。

实测工具输出、操作系统版本保存在 [system.json](environment/system.json)。全部 `420` 条 dpkg 包记录见 [system-packages.tsv](environment/system-packages.tsv)，记录包名、版本、架构；未包含仓库凭据、APT 私有源、网卡/MAC、主机名或硬件 UUID。

原仓库 Dockerfile 引用 `ccr-registry.baidubce.com/aihc/aibox-pytorch:v1.0-torch2.11.0-cu12.6`。这是源码里的基础镜像声明；本次未验证镜像可拉取、digest 或它与正在运行容器完全一致，也未备份镜像层。仅靠旧 Dockerfile 的精简 requirements 不能替代本次全量包清单，尤其不含独立 split 环境。

## 3. Python 环境及差异

| 解释器 | 档案前缀 | 去重后的生效分发包数 | pip check |
|---|---|---:|---|
| `/opt/task5-venv/bin/python` | task5-venv | 108 | No broken requirements found. |
| `/mnt/luoyulin_code/fanxuankai/task6_reproduction/.venv/bin/python` | task6-venv | 108 | No broken requirements found. |
| `/usr/bin/python3` | system-python | 77 | No broken requirements found. |

`importlib.metadata.distributions()` 在继承系统包时会返回被遮蔽的重复版本，例如 fsspec。完整 JSON 同时保留发现的分发列表和 `importlib.metadata.version(name)` 解析出的生效版本。requirements 文件只使用生效版本，不会同时锁定相互冲突的同名版本。

| 两套 venv 的差异包 | task5-venv | task6-venv |
|---|---|---|
| fonttools | 4.63.0 | 4.64.0 |
| kiwisolver | 1.5.0 | 1.5.1 |
| regex | 2026.7.19 | 2026.9.3 |
| task5-routing-testbed | 0.1.0 | — |
| task6-phasea-fullft | — | 0.1.0 |

### 主要运行依赖（task5-venv）

| 包 | 实际生效版本 |
|---|---|
| numpy | 2.4.3 |
| torch | 2.11.0+cu126 |
| transformers | 4.57.6 |
| datasets | 3.6.0 |
| pyarrow | 25.0.1 |
| matplotlib | 3.11.1 |
| PyYAML | 6.0.3 |
| tokenizers | 0.22.2 |
| safetensors | 0.8.0 |
| sentencepiece | 0.2.2 |
| protobuf | 6.33.6 |
| fsspec | 2025.3.0 |
| pandas | 3.0.5 |
| triton | 3.6.0 |
| nvidia-cudnn-cu12 | 9.10.2.21 |
| nvidia-nccl-cu12 | 2.28.9 |

所有生效包的完整对照见本文末尾；每套环境还有 `environment/*-packages.json` 和 `*-requirements.txt`。requirements 是**环境记录，不是已验证可跨机器安装的锁文件**：没有 wheel 哈希或离线 wheelhouse，原供应商特定的 CUDA 构建也可能需要原镜像/可信包源。没有把包索引令牌或私有下载 URL 写入档案。

## 4. Expert split 的历史环境

训练主环境和 Task6 venv 中均未发现 `k-means-constrained`、scikit-learn、SciPy、OR-Tools、joblib。它们用于生成 split，已有 split 的训练/评估无需重新聚类。

| 包 | E64 / E128 / E256 manifest 一致记录 |
|---|---|
| k-means-constrained | 0.9.1 |
| scikit-learn | 1.9.0 |
| numpy | 2.4.3 |
| scipy | 1.18.0 |
| ortools | 9.15.6755 |
| joblib | 1.5.3 |

证据：[split-manifest-versions.json](environment/split-manifest-versions.json)、[split-constraints.txt](environment/split-constraints.txt) 和 `task6_phaseD_reproduction/configs/split/`。旧环境 Python 3.12.3 来自历史分析材料，未在本次不可达的旧机重测；这些本地材料没有上传。

Phase D 的 `_require_versions` 会逐项强校验上述六个版本。每层 W1 行 L2 normalize，seed=1，`k-means++`、`n_init=10`、`max_iter=300`、`tol=1e-4`、`n_jobs=1`，每簇严格等量。恢复时优先找回原 split，勿无说明地用新依赖重新聚类替代旧资产。

## 5. 运行约定与环境变量

- Shell：Linux Bash。各目录的 `_python.sh` 固定工作目录并追加本目录 `src` 到 `PYTHONPATH`；不要使用 Windows 原生运行行为代替已验证的 Linux 行为。
- `PYTHON`：优先显式指定新机器解释器。Task5 正式脚本默认 `/opt/task5-venv/bin/python`；Task6 A 优先显式 PYTHON/本地 venv；Task6 B 使用 PYTHON 或 python；Phase D/Task8 还含旧 Task6 venv 路径回退。
- Task8：设置 `TASK6_ROOT` 指向同级 `task6_phaseD_reproduction`；同时修改 `configs/local/*.yaml` 的 `execution.task6_repo`。`scripts/05_assets/link_server_assets.sh` 另有硬编码旧路径，运行前必须检查并改到新路径；变量本身不会覆盖该脚本内的赋值。
- `CUDA_VISIBLE_DEVICES`：按实际可用 GPU 选择；多卡是独立 condition 分片，`WORLD_SIZE=1`，不是 DDP。
- 数值约定：FP32，AMP/TF32/gradient checkpointing 关闭；deterministic algorithms；cuDNN benchmark 关闭；`CUBLAS_WORKSPACE_CONFIG=:4096:8`。
- `HF_HUB_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`：源码会启用离线读取，本备份没有模型/数据，不能靠首次启动自动联网补齐。
- `TOKENIZERS_PARALLELISM=false`、`PYTHONDONTWRITEBYTECODE=1`、`PYTHONUNBUFFERED=1` 是部分启动器显式设置；CPU 收尾使用 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。本次测试另将 OPENBLAS_NUM_THREADS=1 并隐藏 GPU。
- 非交互 SSH 会话仅观测到 allowlist 中的 `CUDA_HOME`；没有推断它等于历史训练进程的完整环境，也没有读取 shell history 或密钥文件。

## 6. 新机器恢复次序

1. 准备 Linux x86_64、Python 3.12.3 和可用 NVIDIA runtime。以本文件及原镜像为版本证据，先恢复兼容的 `torch==2.11.0+cu126`，再恢复其余依赖；如替换驱动/镜像/版本，记录差异并重跑 CUDA smoke，不假定逐位一致。
2. 参照 `environment/task5-venv-requirements.txt` 还原有效包集合；若原镜像存在且可信，可重建 `--system-site-packages` venv。清洁独立 venv 也可承载同样生效依赖，但仍须验证；本次未执行新机安装测试。
3. 在 monorepo 根目录，用目标解释器分别执行 `-m pip install --no-deps -e ./task5_reproduction`、`-e ./task6_reproduction`、`-e ./task6_phaseB_reproduction`、`-e ./task6_phaseD_reproduction`、`-e ./task8_reproduction`。不要从 PyPI 安装同名实验包。
4. 运行 `"$PYTHON" -m pip check`，核对 Python/torch/CUDA、包版本，恢复 README 所列资产，再更新每个子库 local 配置的所有绝对路径。
5. 如果确需重建 split，另设聚类环境并应用 split 精确约束；不可把当前主训练环境当作已有聚类环境。
6. 先运行本文档记录的无资产测试，随后按各自 RUNBOOK 执行目标机器的 Phase 0、真实资产 smoke 和恢复一致性检查，再启动正式工作。

## 7. 本次验证

五个隔离源码副本的现有测试：Task5 76、Task6 A 14、B 34、D 24、Task8 25，合计 173，通过且零跳过。另有 297 个 Python 文件 AST 检查和 103 个 shell 文件 `bash -n` 通过。测试使用 `/opt/task5-venv/bin/python`，隐藏 GPU；原实验目录和正式结果未被写入。Task8 首次尝试 pytest 因未安装而失败，改用其 README 指定的 unittest 后通过；没有为此安装 pytest。

验证记录见 [backup/validation.json](backup/validation.json)。本次没有重跑大规模训练、真实数据端点或科学结论，因此不能把代码备份验收解释为所有实验结果重新验证。

## 8. 全部 Python 生效版本

| 分发包 | task5-venv | task6-venv | system-python |
|---|---|---|---|
| aiohappyeyeballs | 2.7.1 | 2.7.1 | — |
| aiohttp | 3.14.3 | 3.14.3 | — |
| aiosignal | 1.4.0 | 1.4.0 | — |
| asttokens | 3.0.1 | 3.0.1 | 3.0.1 |
| attrs | 26.1.0 | 26.1.0 | — |
| build | 1.4.0 | 1.4.0 | 1.4.0 |
| certifi | 2026.2.25 | 2026.2.25 | 2026.2.25 |
| charset-normalizer | 3.4.6 | 3.4.6 | 3.4.6 |
| click | 8.3.1 | 8.3.1 | 8.3.1 |
| cmake | 4.2.3 | 4.2.3 | 4.2.3 |
| contourpy | 1.3.3 | 1.3.3 | — |
| cuda-bindings | 12.9.4 | 12.9.4 | 12.9.4 |
| cuda-pathfinder | 1.2.2 | 1.2.2 | 1.2.2 |
| cuda-toolkit | 12.6.3 | 12.6.3 | 12.6.3 |
| cycler | 0.12.1 | 0.12.1 | — |
| datasets | 3.6.0 | 3.6.0 | — |
| decorator | 5.2.1 | 5.2.1 | 5.2.1 |
| dill | 0.3.8 | 0.3.8 | — |
| dnspython | 2.8.0 | 2.8.0 | 2.8.0 |
| executing | 2.2.1 | 2.2.1 | 2.2.1 |
| expecttest | 0.3.0 | 0.3.0 | 0.3.0 |
| filelock | 3.25.2 | 3.25.2 | 3.25.2 |
| fonttools | 4.63.0 | 4.64.0 | — |
| frozenlist | 1.8.0 | 1.8.0 | — |
| fsspec | 2025.3.0 | 2025.3.0 | 2026.2.0 |
| hf-xet | 1.6.0 | 1.6.0 | — |
| huggingface_hub | 0.36.2 | 0.36.2 | — |
| hypothesis | 6.151.9 | 6.151.9 | 6.151.9 |
| idna | 3.11 | 3.11 | 3.11 |
| importlib_metadata | 9.0.0 | 9.0.0 | 9.0.0 |
| ipython | 9.11.0 | 9.11.0 | 9.11.0 |
| ipython_pygments_lexers | 1.1.1 | 1.1.1 | 1.1.1 |
| jedi | 0.19.2 | 0.19.2 | 0.19.2 |
| Jinja2 | 3.1.6 | 3.1.6 | 3.1.6 |
| kiwisolver | 1.5.0 | 1.5.1 | — |
| lintrunner | 0.13.0 | 0.13.0 | 0.13.0 |
| MarkupSafe | 3.0.3 | 3.0.3 | 3.0.3 |
| matplotlib | 3.11.1 | 3.11.1 | — |
| matplotlib-inline | 0.2.1 | 0.2.1 | 0.2.1 |
| mpmath | 1.3.0 | 1.3.0 | 1.3.0 |
| multidict | 6.7.1 | 6.7.1 | — |
| multiprocess | 0.70.16 | 0.70.16 | — |
| networkx | 3.6.1 | 3.6.1 | 3.6.1 |
| ninja | 1.13.0 | 1.13.0 | 1.13.0 |
| numpy | 2.4.3 | 2.4.3 | 2.4.3 |
| nvidia-cublas-cu12 | 12.6.4.1 | 12.6.4.1 | 12.6.4.1 |
| nvidia-cuda-cupti-cu12 | 12.6.80 | 12.6.80 | 12.6.80 |
| nvidia-cuda-nvrtc-cu12 | 12.6.85 | 12.6.85 | 12.6.85 |
| nvidia-cuda-runtime-cu12 | 12.6.77 | 12.6.77 | 12.6.77 |
| nvidia-cudnn-cu12 | 9.10.2.21 | 9.10.2.21 | 9.10.2.21 |
| nvidia-cufft-cu12 | 11.3.0.4 | 11.3.0.4 | 11.3.0.4 |
| nvidia-cufile-cu12 | 1.11.1.6 | 1.11.1.6 | 1.11.1.6 |
| nvidia-curand-cu12 | 10.3.7.77 | 10.3.7.77 | 10.3.7.77 |
| nvidia-cusolver-cu12 | 11.7.1.2 | 11.7.1.2 | 11.7.1.2 |
| nvidia-cusparse-cu12 | 12.5.4.2 | 12.5.4.2 | 12.5.4.2 |
| nvidia-cusparselt-cu12 | 0.7.1 | 0.7.1 | 0.7.1 |
| nvidia-nccl-cu12 | 2.28.9 | 2.28.9 | 2.28.9 |
| nvidia-nvjitlink-cu12 | 12.6.85 | 12.6.85 | 12.6.85 |
| nvidia-nvshmem-cu12 | 3.4.5 | 3.4.5 | 3.4.5 |
| nvidia-nvtx-cu12 | 12.6.77 | 12.6.77 | 12.6.77 |
| optree | 0.19.0 | 0.19.0 | 0.19.0 |
| packaging | 26.0 | 26.0 | 26.0 |
| pandas | 3.0.5 | 3.0.5 | — |
| parso | 0.8.6 | 0.8.6 | 0.8.6 |
| pexpect | 4.9.0 | 4.9.0 | 4.9.0 |
| pillow | 12.1.1 | 12.1.1 | 12.1.1 |
| pip | 24.0 | 24.0 | 26.0.1 |
| prompt_toolkit | 3.0.52 | 3.0.52 | 3.0.52 |
| propcache | 0.5.2 | 0.5.2 | — |
| protobuf | 6.33.6 | 6.33.6 | — |
| psutil | 7.2.2 | 7.2.2 | 7.2.2 |
| ptyprocess | 0.7.0 | 0.7.0 | 0.7.0 |
| pure_eval | 0.2.3 | 0.2.3 | 0.2.3 |
| pyarrow | 25.0.1 | 25.0.1 | — |
| Pygments | 2.19.2 | 2.19.2 | 2.19.2 |
| pyparsing | 3.3.2 | 3.3.2 | — |
| pyproject_hooks | 1.2.0 | 1.2.0 | 1.2.0 |
| python-dateutil | 2.9.0.post0 | 2.9.0.post0 | — |
| python-etcd | 0.4.5 | 0.4.5 | 0.4.5 |
| PyYAML | 6.0.3 | 6.0.3 | 6.0.3 |
| regex | 2026.7.19 | 2026.9.3 | — |
| requests | 2.32.5 | 2.32.5 | 2.32.5 |
| safetensors | 0.8.0 | 0.8.0 | — |
| sentencepiece | 0.2.2 | 0.2.2 | — |
| setuptools | 81.0.0 | 81.0.0 | 81.0.0 |
| six | 1.17.0 | 1.17.0 | 1.17.0 |
| sortedcontainers | 2.4.0 | 2.4.0 | 2.4.0 |
| spin | 0.17 | 0.17 | 0.17 |
| stack-data | 0.6.3 | 0.6.3 | 0.6.3 |
| sympy | 1.14.0 | 1.14.0 | 1.14.0 |
| task5-routing-testbed | 0.1.0 | — | — |
| task6-phasea-fullft | — | 0.1.0 | — |
| tokenizers | 0.22.2 | 0.22.2 | — |
| torch | 2.11.0+cu126 | 2.11.0+cu126 | 2.11.0+cu126 |
| torchaudio | 2.11.0+cu126 | 2.11.0+cu126 | 2.11.0+cu126 |
| torchelastic | 0.2.2 | 0.2.2 | 0.2.2 |
| torchvision | 0.26.0+cu126 | 0.26.0+cu126 | 0.26.0+cu126 |
| tqdm | 4.70.0 | 4.70.0 | — |
| traitlets | 5.14.3 | 5.14.3 | 5.14.3 |
| transformers | 4.57.6 | 4.57.6 | — |
| triton | 3.6.0 | 3.6.0 | 3.6.0 |
| typing_extensions | 4.15.0 | 4.15.0 | 4.15.0 |
| urllib3 | 2.6.3 | 2.6.3 | 2.6.3 |
| uv | 0.10.12 | 0.10.12 | 0.10.12 |
| wcwidth | 0.6.0 | 0.6.0 | 0.6.0 |
| wheel | 0.46.3 | 0.46.3 | 0.46.3 |
| xxhash | 4.0.1 | 4.0.1 | — |
| yarl | 1.24.5 | 1.24.5 | — |
| zipp | 3.23.0 | 3.23.0 | 3.23.0 |
