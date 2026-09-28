# 架构与四阶段边界

实验配置使用YAML，并通过PyYAML SafeLoader解析；重复键、非字符串键、非法`extends`和Python对象标签都会拒绝。
suite通过`extends`合并分区配置，`configs/local/`最后覆盖机器路径/设备。`pyproject.toml`仅是Python打包标准元数据，
不是实验配置。Linux阶段入口为Shell薄封装，模型、训练、采集和指标逻辑仍统一位于Python源码中。
公共信息写在产物目录中的 `complete.json` / `context.json`，观测记录不重复写task、seed、checkpoint。
`protocol` 绑定模型/数据/训练/采集配置与相关源码摘要；改动模型计算需新run并重做相应验证，不混用旧capture。
对已完成主矩阵追加R4-R2Init时使用显式扩展配置：新arm写新protocol，旧prepared/E/dense按记录的base protocol和输入身份只读复用；组合汇总写入前保留旧结果快照。
离线metric/汇总/绘图代码及metrics配置单独记为`analysis`，更改它们不使checkpoint/capture失效；重算指标后再汇总即可。

## 源码分工

| 目录 | 职责 |
|---|---|
| common | 配置、身份、I/O、随机流、日志、共享上下文 |
| data | 本地数据、模板/verbalizer、缓存、动态padding、固定probe |
| substrate | 验证既有labels、原神经元顺序的dense-mask FFN、T5接入、Phase0 |
| routing | R1–R4/R4-R2Init/G0–G4；FP32打分、确定性ties、独立噪声、aux与bias状态 |
| training | 只训练router、Adam/scheduler、epoch边界checkpoint及续训 |
| capture | 生成A、teacher-forcing B、逐token C/D、dense E、压缩分片 |
| metrics | 仅消费原始数据；performance、load、churn、三个probe |
| aggregation | 12层宏平均、各seed独立best、样本std、配对性能差 |
| visualization | 仅从results/data导出表格和PDF/PNG |

## T5接入

保留每层原 `wi/wo/act/dropout`，不改变attention、norm、residual或原神经元顺序。
在未mask/加权/dropout的ReLU激活上按labels求q；再把expert系数映射回神经元位置，经过原dropout和W2。
`force_all=True` 绕过router及系数，执行无权全选，专门用于dense等价性测试。
只有router参数可训练；冻结W1/W2不意味着切断hidden input的梯度。

encoder/decoder stack的pre-hook传入本次真实input token IDs；G0生成时随cache使用当前decoder token，不能重复使用标签或原前缀。
teacher-forcing统计mask来自attention_mask/labels!=-100；decoder start即便等于pad ID也不因此删除。

## 产物依赖

```text
inputs只读 → prepare：probe成员/静态centroid/hash/labels快照
               ↓
train：router + optimizer/scheduler/RNG → 11个状态
               ↓
capture A → correct count选best → B(best/final) + C/D(按矩阵) + 共享E
               ↓
metrics → normalized/aggregated → tables/figures → 人工分析报告
```

原始产物按 `task/arm/variant/k/seed/run_id/state` 定位，不使用全局registry或自动“取最新”。
checkpoint只保存router及恢复状态，冻结底座通过内容身份引用。
所有11点都可在epoch边界续训；不支持恢复到未保存的epoch中间。
`.incomplete_*` 是未提交checkpoint的临时目录，保留供排错，不当作候选状态。

## 缓存

`tmp/preprocessing/raw/` 放本地parquet加载缓存；`tokenized/<task>/<population>/<identity>/` 放map缓存。
identity包括已核验资产、数据fingerprint/数量、tokenizer结构/版本、模板/verbalizer、长度、padding及workers。
prepare预先生成train/validation缓存，再并行启动训练。缓存可以重建；不是实验原始观测，也不参与选best。

## 局部与共享统计

D永远来自当前arm的实际hidden轨迹，保存全部64个q；`sum(q)`就是coverage分母，不重复存一列。
E来自每任务固定dense完整validation，保留原神经元顺序、完整对角线、FP32 sum及N。
metric阶段读取prepare的labels小快照，计算 `C_sum/N → Q`，不需要重新访问远程dense或数据集。
