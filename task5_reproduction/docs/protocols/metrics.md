# 已落实的采集与指标口径

| 原始类型 | 采集范围 | 保存内容 |
|---|---|---|
| A | 所有候选完整validation；dense/static一次 | sample_id、原始预测文本、right、valid |
| B | 稀疏static及可训练best/final，完整validation teacher forcing | 每层T及64个assignment counts |
| C | 所有稀疏static、可训练全部11点；固定probe | sample_id、位置、层、k个expert |
| D | static、R4与R4-R2Init全部点、G1–G4的best/final | 在C同一行追加完整64维q |
| E | 每任务固定dense、完整validation一次 | 每层FP32 coactivation_sum和实际N |

A/B为各状态各一份Parquet；C/D按层每65,536条有效token分片，ZSTD level3；E按层NPZ。
不保存逐token全部hidden/logits或所有神经元激活；不会因计算新的一遍既定metric而补跑模型。

## 归约

- accuracy：right/全部样本；invalid仍算错。relative=100×acc/同任务实测dense acc。
- best：逐条件逐seed在11点中按正确数最大、同分最早成功step选拔，包括step0。不挑seed。
- load：先全validation累计，再计算p=count/(T×k)、population CV、未额外有限样本修正的Gini、max(p)。
- churn：相邻保存点的 `1-|交集|/k`；exact_set_change只作诊断。保留best之后的轨迹，终态为epoch9→10。
- overlap：当前局部q的稳定top-k与实际集合交集/k；并列仅按expert ID，随机期望k/E。
- coverage：逐token `sum(q[selected])/sum(q)`；只跳过严格全零并计数。不加epsilon，不使用总分子/总分母代替。
- 共激活：先C_sum/N，再按原labels计算跨expert神经元均值Q，再对选中expert无序pair平均。
- 各层先独立token平均，再12层等权宏平均；层/expert std用ddof0，seed std用ddof1。空层/NA不静默删除。
- 共激活model selected先宏平均，random按重复编号先宏平均，再计算model excess、ratio和2.5%/97.5%线性分位区间；不平均区间端点。
- 每层selected/random/excess/ratio及其层间std另存；原始共激活幅值不定义“最差层”。

随机参照100次，独立seed0。每token从64个expert无放回取k；NumPy PCG64均匀随机rank法，流按task/layer/k/N/C身份/重复编号派生。
分块不改变随机集合；缓存复用跨arm/seed/checkpoint。只保存每重复均值，不保存随机集合。
随机区间不是训练seed或总体性能置信区间。ratio仅在mean尺度random>1e-12时定义。

输出的mean±std来自各seed独立best；final单列。配对性能差先逐seed相减，再算sample std，accuracy差用百分点。
静态R1–R3不伪造seed重复，G0的三个hash seed正常汇总。M19不构造自动赢家。
图表仅消费results/data；百分比显示2位，其他小数4位，极小值科学计数；底层值不舍入。
