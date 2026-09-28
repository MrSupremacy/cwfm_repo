# 数据约定

这些schema描述单条记录或产物公共结构，不是registry。运行时的数组dtype、T×k守恒、完整成员/层/seed、
hash和checkpoint关联检查由对应Python模块执行，不依赖JSON Schema库。

A 的逐任务 candidate/greedy 记录使用 gzip JSON，B/C/D 使用 Parquet；C/D 中 D 是否必需由目录公共 header.with_q 决定。数组的精确宽度由
condition中的E和k约束：B与D宽度为E，C宽度为k；JSON Schema描述 Phase D 的 E64/E128/E256 外部范围，
运行时执行精确检查。
complete.json中的files只包含本产物文件摘要，不把logs加入统计数据。
JSON里未定义metric用null（NA），禁止NaN/Infinity；FP32/整数位宽见协议文档与实际Parquet schema。

