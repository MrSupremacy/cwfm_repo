# 当前决定与优先级

优先级：用户最新明确决定 > 更新后的03–06文档 > 旧handout/proposal/参考代码默认值。
本库可独立运行，不需要运行时读取这些说明文档或旧仓库。

## 最新覆盖

1. **R3**：保留原参考RMS公式、epsilon=1e-6、除sqrt(D)、temperature=1，原始均值centroid冻结、无权hard top-k。
   R2仍使用L2 cosine。取消旧文档的“选中集合必须完全一致”门槛，保留同输入诊断。
   回归测试包含允许R2/R3排序不同的小范数反例，防止以后误把R3改成R2。
2. **缓存隔离**：用户确认所有预处理缓存放新库tmp/preprocessing，不写回原输入。
3. **M19**：未确认，不自动选择best patched gate；G2三档/G3/G4分别保留。
4. **保存恢复**：11点全部保存router、optimizer、scheduler、shuffle/dropout/noise RNG；不保存冻结dense副本。
   epoch边界可完整续训。中断时最多重做未保存epoch，不声称恢复任意microbatch位置。
5. **服务器输入**：固定 MoEfication 两任务 seed0/checkpoint-best 与 replicate0/parameter_seed0。
   核验后复制到新库 inputs，保留32个文件的来源与SHA256；不使用EMoE资产，不改旧文件。
6. **PyArrow**：用户批准沿用服务器25.0.1，通过真实Parquet测试后将新库依赖固定为25.0.1。
7. **R4-R2Init补充实验**：除summary初始化精确复用R2/R3共享K-Means centroid外，与R4的RMS 1e-6打分、soft权重和自由训练完全相同；三个seed复用原编号。完整新增24个训练run、264个状态及对应A/C/D，best/final采B，共享旧dense A与E。
   对既有`formal20260830a`采用显式扩展协议：新arm不冒充旧protocol，旧prepared/E/dense只读校验复用，组合汇总前保留旧结果快照。

## 服务器适配修复

2026-09-04 Phase A F0 补测：R2-soft 使用固定 raw centroid、L2 cosine（eps=1e-12）、选中项 k×softmax，温度1且无 sqrt(D) 缩放；只测 seed_fixed。R4-hard 与 R4 同 seed/layer 正交初始化、RMS 1e-6 打分；系数 `1 + (w_soft - w_soft.detach())`，前向精确1、反向保留 soft 导数，不 detach hidden 或 expert 输出。Backbone 始终冻结，只更新路由；保留24次训练、264点及全程D/churn。

补测配置在 `configs/suites/phaseA_f0.yaml` 和 `configs/extensions/phaseA_f0.yaml`，不改变旧 main 矩阵。旧 v3 原精度 normalized 数据只读且固定SHA256，新数据保留新protocol；独立 `task6_phaseA_F0_result` 仅绘制用户指定八臂。运行命令统一见 RUNBOOK §17。新 protocol 对 Python 源码换行符规范化后哈希，避免 Windows/Linux checkout 仅因 CRLF/LF 不同而使新产物无法互用；旧记录仍使用显式原 protocol，不改写历史身份。

- datasets内部fingerprint缩为32字符，避免四进程后缀使其超过64字符上限；完整SHA256缓存目录与声明不变。
- 每次日志上下文退出时解除代理与该次文件的关联；第三方库保留的旧代理改为转发到当前日志流。
  不关闭终端、不复开旧日志；正常退出、异常退出、连续任务、嵌套与Linux fork均有回归测试。
- Windows测试明确以UTF-8读取本库日志，与写入编码一致。这些修复不改变路由或训练算法。

## 实现显式化（不新增实验变量）

- 原始神经元顺序+同一dense-mask底座，q/E/labels保持一致。
- G1–G4初始化复现Linear默认的±1/sqrt(D)均匀分布，W/b均有；同seed同layer的公共gate配对。
- init/noise流用SHA256派生的独立torch Generator；shuffle独立Generator(run seed)，worker base seed不消费shuffle流；dropout流独立重置/保存。
- G3零初始化无bias noise matrix；G4用softmax(clean)+beta选择，clean selected logits计算k×softmax权重。
- 各run单GPU，accumulation1，全FP32/TF32关闭；不实现DDP、AMP、activation checkpointing的未确认分支。
- 未决定数据格式时不伪造资产。已知旧artifact格式先适配，实际服务器资产仍需核验。
- R4是自由学习的summary vectors，并非严格参数派生；R3→R4不是单因素消融。保持该解释边界。
