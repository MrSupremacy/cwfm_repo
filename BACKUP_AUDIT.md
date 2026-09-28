# 2026-09-28 备份审计说明

## 来源与保真边界

权威来源是指定远程资产根目录内五个 `*_reproduction` 工作目录。先阅读本地 task5/6/8 handout、analysis 和蓝图，确定代码职责与依赖；这些本地材料没有复制进本仓库。

在远端先做文件清单、SHA256、隐私模式检查，再用明确文件白名单导出。归档生成时再次核验 SHA256，接收后逐文件再核验。保留 528 个原始文件（1,173,468 bytes），原始字节和 Linux mode 记录在 `backup/source-manifest.json`；没有重构、合并或修订实验算法。

源码目录保留并列，避免破坏独立包边界与 Task8 → Phase D 的代码依赖。总 README、环境文档、审计材料和 Git 规则为本次新增文件。旧代码中的绝对挂载路径原样保留，并在总 README 列出迁移检查项。

原远程文件混用 CRLF/LF，部分 YAML 的实际 CRLF 与原 `.gitattributes` 的 LF 声明不一致。为保证首次快照的 Git blob 也逐字节匹配远端，本地导入时使用 `.git/info/attributes` 的 `* -text` 禁用自动转换；原子库属性文件不改动。未来在其他机器编辑并提交时，Git 可能按原 YAML/LF 规则正规化换行；可用 source-manifest 对照判断仅换行变化，勿误认作算法改动。

## 本地与远端对比

以下比较覆盖剔除大资产/缓存/安装元数据后的远端轻量候选文件，包括随后被排除的历史报告；不是对整套 checkpoint/data 的哈希审计。

| 代码库 | 字节一致 | 仅 CRLF/LF 差异 | 仅远端存在 | 实质差异 | 仅本地存在 |
|---|---:|---:|---:|---:|---:|
| Task5 | 37 | 100 | 44 | 0 | 0 |
| Task6 A | 102 | 0 | 0 | 0 | 0 |
| Task6 B | 98 | 0 | 1 | 0 | 0 |
| Task6 D | 125 | 0 | 4 | 0 | 0 |
| Task8 | 81 | 0 | 0 | 0 | 0 |

具体列表见 `backup/local-remote-comparison.json`。远端特有的 B/D `configs/local/server.yaml` 是当前部署配置，纳入备份；Task5 的远端特有文件主要是历史报告/图表，按本次范围排除。Phase D 三个过期副本排除：

- `configs/local/server.yaml.before_e128_e256`
- `src/task6_phased/aggregation/pipeline.py.pre_plot_fix`
- `src/task6_phased/visualization/render.py.pre_plot_fix`

## 隐私检查和文档排除

扫描候选文本中的常见 GitHub/Hugging Face/API token、私钥头、密码/密钥赋值、私网 IP、SSH 连接信息、邮箱和机器绝对路径；同时检查文件类型、符号链接、大文件和生成目录。审计通过的上传文件未发现实际凭据、私钥、私网 IP 或邮箱。`token` 等机器学习术语、公开 JSON schema URL、供应商基础镜像名不按密钥误删。

路径类命中不等同于凭据泄露。运行配置、脚本与 RUNBOOK 中的原挂载路径和环境路径是复现所需的部署线索，保留在此私有备份中；未复制登录地址、SSH 密钥、shell history、完整环境变量、Git 凭据或主机/GPU 标识。

为避免把服务器验收记录和研究报告混入源码备份，**整个 Task5/Task6 A 的 `docs/reports/` 均在下载前排除**。其中以下文件包含机器目录、服务器运行细节或验收证据，作保守隐私/范围排除；并不意味着已发现其中有密码：

- `task5_reproduction/docs/reports/server_validation.md`
- `task5_reproduction/docs/reports/yaml_shell_migration.md`
- `task5_reproduction/docs/reports/phaseA_f0_validation.md`
- `task5_reproduction/docs/reports/formal20260830a_analysis/experiment_report_draft.md`
- `task5_reproduction/docs/reports/formal20260830a_analysis_v2/experiment_report_v2.md`
- `task5_reproduction/docs/reports/server_evidence/` 下全部文件。
- `task6_reproduction/docs/reports/server_acceptance_20260905.md`

同目录中的 local_validation、report_template 和历史图表也因统一排除报告目录而未纳入。**这些原文件未进入本地 cwfm_repo，也未进入导出归档**；逐项路径和排除原因见 `backup/exclusions.json`。旧说明里的历史证据链接因此可能缺失；本次验证记录是新生成的独立汇总，不替代历史证据。

这属于规则扫描与人工复核，不能作为不存在任何未知敏感信息的数学保证；本次创建的仓库保持 private。

## 其他未备份内容

五库全部 inputs、数据和 checkpoint、正式输出/日志、缓存、虚拟环境、egg-info、符号链接目标、旧 Git 历史不在本次范围。没有下载本地 handout/analysis/report/feedback/学生包。`schemas/results/` 是代码协议，已保留，未按实验 results 目录误删。

环境记录收录三套解释器的完整生效包列表、发现的重复分发、系统 dpkg 清单、CPU/GPU/CUDA/驱动和工具链。历史 Task4 split 开发机连接超时，六项聚类版本从共享原始 manifest 核实，但旧机的完整环境无法实测；详见 ENVIRONMENT.md。备份不包含可直接搬迁的 venv、容器镜像或 wheelhouse。

## 验证结论

使用远程已安装的 `/opt/task5-venv/bin/python`，将同一白名单源码复制到隔离临时目录后执行各项目现有测试。隐藏 GPU、不读取正式模型数据、不改动原始实验目录。173 项测试全部通过，零跳过；Python AST 297 文件和 Bash 语法 103 文件通过。

Task8 最初尝试 pytest 时发现环境未安装该工具；随后按该库 README 的 `unittest discover` 命令通过全部 25 项，没有修改依赖环境。

本次验证支持“所备份源码与远端一致，现有无资产测试通过”，不支持“重新证明所有科学结果无误”或“clone 后不补资产即可运行”。后续恢复正式实验仍须执行真实资产、CUDA、环境/协议身份与 checkpoint 完整性检查。
