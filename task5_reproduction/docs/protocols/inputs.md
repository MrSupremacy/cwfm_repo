# 既有输入约定

只支持任务已确认的本地资产；不自动下载、微调或重新聚类。
路径通过 `configs/local/server.yaml` 绑定。本次服务器交付已将核验后的原资产复制到本库 `inputs/`，
原目录不变；配置使用仓库相对路径，迁移时无需继续依赖 Task 4。仍支持显式绑定外部只读路径。

## Dense / tokenizer

每任务一个已有T5-small目录：`config.json`、`model.safetensors`或`pytorch_model.bin`，以及原tokenizer文件。
结构核验：非gated ReLU、D=512、FFN=2048、encoder/decoder各6层。
当前适配已知非分片checkpoint，不把其他模型/分片格式静默当作同一底座。
模型及tokenizer均local_files_only；实际generation config、tokenizer侧向/特殊token及库版本在prepare保存。

## K-Means artifact

每任务固定一个目录，沿用旧仓库的 `manifest.json` 与 `labels.npz` 格式。
manifest需包含 task、method=`parameter`、checkpoint_sha256、files.labels.npz。
checkpoint关联摘要兼容旧工程：按文件名排序，对config及非分片权重的文件名与SHA256字节串求摘要。
labels键为 `encoder_layer_00..05`、`decoder_layer_00..05`，整数数组长度2048，值0..63，每expert恰32个。
不以浮点labels自动转整数蒙混检查。

centroid重新从**同一冻结W1的原始行均值**计算，不重新聚类；prepare保存小型centroid/labels快照。
split seed不是router seed。本次固定原复现 `replicate0/parameter_seed0`；两任务 labels 摘要均与
preposition 报告一致。所有 arm/ratio/router seed 共用同任务的这一套 split。

## 数据集

- SST-2：train 67,349 / validation 872；字段sentence、label。
- MNLI：train 392,702 / validation_matched 9,815；字段premise、hypothesis、label。
- 标签编号与模板遵守configs/base/default.yaml。保留重复样本，不新增过滤/清洗/重采样。
- `local_files`绑定train和validation parquet；`load_from_disk`绑定完整DatasetDict目录的dataset字段。
- 数量/标签异常直接报错。smoke也先核验正式输入，再固定取前512/32条，不把错误输入裁成目标数量。
- sample_id采用原split行号；tokenization/子集/打乱后保持该ID。

所有预处理缓存显式放到输出根目录的 `tmp/preprocessing/`，不使用原数据目录保存新缓存。
用户指定的output_root应与输入目录分开，不能指向dense/split/数据目录内部。

完整输入 SHA256 保留在缓存目录名与 declaration 中；传给 datasets.map 的内部 fingerprint
取前32字符，为多进程追加分片后缀留空间，不改变样本、tokenization或缓存身份校验。
