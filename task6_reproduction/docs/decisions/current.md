# Frozen Phase A/F1 decisions

- Arms: R2, R2-soft, R4o, R4d, R4o-hard, G1, G2-0.001 and G4.
- Dense baselines: fixed `Dense-init` remains the 100% relative-performance denominator; `Dense-fullFT` trains for ten epochs at each task × seed, with no `k` expansion, and selects best by the same validation rule.
- Expert construction: existing balanced parameter K-Means labels only.
- Full finetuning: backbone LR `1e-5`; router LR `3e-4`; Adam; FP32; TF32 off.
- Ten epochs; batch 256; default accumulation 1; 5% linear warmup followed by linear decay; global gradient clip 1.
- All scheduled checkpoints retain the full model, optimizer, scheduler and RNG state.
- Captures: A for every trained state and both static dense references; B at routed best/final states; C+D for every routed state.
- Metrics: performance for every model; load-balance CV, churn, oracle overlap and activation coverage for routed models.
- Compatibility: an earlier routed-only run keeps its original checkout/protocol through aggregation; the six Dense-fullFT runs are appended from lightweight result files with source hashes, never by relabeling old checkpoints. The extension redraws only accuracy/relative-performance figures and the main table; other rendered artifacts are copied unchanged.
