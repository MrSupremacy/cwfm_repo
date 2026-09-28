# Intrinsic Router in MoE - Preliminary

独立实验代码库：在既有T5-small dense checkpoint与balanced K-Means expert拆分上比较R1–R4、R4-R2Init、G0–G4。
现已支持 Phase A **F0 补测** R2-soft（静态）和 R4-hard（系数ST、router-only）；只补新臂及独立八臂图表的入口见 [RUNBOOK §17](RUNBOOK.md#17-phase-a-f0只补-r2-soft-与-r4-hard)。本次不实现 fullFT。
不导入旧MoEfication/EMoE工程，不重训dense，不重新聚类，不下载模型或数据。

本仓库上传源码、通用配置与配置模板、运行脚本、测试和工程文档。
真实模型checkpoint、数据集、expert split、机器专用配置和完整实验输出不上传；
使用时按[输入协议](docs/protocols/inputs.md)准备本地资产，并复制配置模板填写自己的路径。

## 从这里开始

环境测试、smoke、正式训练、A–E采集、六组metric、汇总和制图的全部可执行步骤，统一放在：

**[Task 5 全流程唯一运行手册](RUNBOOK.md)**

请只按这一份手册执行。`docs/reports/`是历史验收证据，`docs/protocols/`和`docs/design/`解释设计，
都不是第二套运行说明；旧服务器手册也只保留指向本手册的链接。

## 已冻结的关键决定

- R3保留RMS、`1.0e-6`、冻结原始centroid和hard sum；R2/R3集合一致率只记录，不作为门槛。
- M19未定：G2三档、G3、G4分别完整报告，不自动挑选“best patched gate”。
- 单卡单进程、FP32、关闭AMP/TF32/梯度checkpoint；多卡按独立condition分片，不使用DDP。
- 先完成全部训练，再统一采集；metric/汇总/绘图离线消费capture，不混入训练循环。
- best按每个task×arm×variant×ratio×seed在完整validation上独立选择，同分取更早step。

## 目录

```text
configs/              YAML实验配置和本机路径覆盖
src/task5/            数据、底座、路由、训练、采集、指标、汇总、可视化
scripts/00..80/       Linux Shell阶段入口
schemas/              A–E、metric及公共元数据结构
tests/                unit、integration、regression、smoke
docs/                 设计、协议、决定和历史验收报告
inputs/               [本地准备，不上传] dense、expert split、数据与来源摘要
artifacts/            [运行产生] probe、静态router、共享E
runs/                 [运行产生] validate、train、capture、metrics
results/              [运行产生] 聚合数据、表格、图片与报告
tmp/                  [运行产生] smoke、缓存和测试报告
```

## 当前验收状态

- 小服务器A800完成真实权重的四阶段smoke与R4/G3/G4精确续训对照。
- YAML/Shell迁移后，A800服务器67项测试全部通过、0跳过、0失败。
- 已核验输入副本共32个文件；无需原Task4目录即可运行。
- 当前完整主矩阵为192组训练；既有168组正式结果可显式扩展24组R4-R2Init，旧原始产物保持不变。
- 正式实验`formal20260830a`及新增24组R4-R2Init的结果已完成汇总；完整产物与分析报告另行保存，不包含在此源码仓库中。

工程证据见[服务器验证报告](docs/reports/server_validation.md)和
[YAML/Shell迁移验收](docs/reports/yaml_shell_migration.md)。方法与数据契约见
[架构与路由说明](docs/design/architecture.md)和
[capture及metric协议](docs/protocols/metrics.md)。
