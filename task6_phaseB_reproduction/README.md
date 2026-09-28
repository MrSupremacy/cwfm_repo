# Task 6 Phase B/F0 reproduction

本仓库独立实现 Phase B 的冻结 backbone 实验。新实验只运行 E128/E256；E64 与 Dense-init 从 Phase A/F0 的规范化结果只读导入。E128/E256 分别生成同一 E 内随预算变化的图表，E64/E128/E256 仅生成 matched-ratio 对比表。

固定实验口径：

- task：SST-2、MNLI-matched；
- arm：R2、R4o、R4d、G1、G2-0.001、G4；
- budget：E64 为 6/13/19/26，E128 为 13/26/38/51，E256 为 26/51/77/102；
- R2 不训练；其余五臂只训练 router，3 seeds，10 epochs；
- 负载均衡只计算 CV；不实现 Gini、maximum share 或 coactivation；
- 每个 condition 单卡单进程，正式启动器把 240 个训练 run 固定分到 8 张卡。

正式矩阵为 240 个训练 run、16 个 R2 静态 condition、2640 个 router checkpoint 和 2656 个新 routed states。训练、采集、metric、汇总与表图严格分阶段执行。

完整命令、输入准备和恢复方法见 [RUNBOOK.md](RUNBOOK.md)。
