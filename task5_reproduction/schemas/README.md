# 数据约定

这些schema描述单条记录或产物公共结构，不是registry。运行时的数组dtype、T×k守恒、完整成员/层/seed、
hash和checkpoint关联检查由对应Python模块执行，不依赖JSON Schema库。

A/B/C/D物理格式是Parquet；C/D中D是否必需由目录公共header.with_q决定。
E物理格式是每层NPZ，layer_id来自文件名；schema展示其概念数组结构，不要求转换为巨型JSON。
complete.json中的files只包含本产物文件摘要，不把logs加入统计数据。
JSON里未定义metric用null（NA），禁止NaN/Infinity；FP32/整数位宽见协议文档与实际Parquet schema。
