# Metrics protocol

- Performance: exact normalized label-string accuracy, invalid-output rate, and `100 × model_accuracy / Dense-init_accuracy`. Best is maximum validation correct count, with the earliest optimizer step breaking ties. `Dense-init` is the fixed 100% denominator. `Dense-fullFT` is trained for ten epochs on three seeds and contributes its seed-aggregated best value as a horizontal reference in each relative-performance figure.
- Load balance: for expert counts `n_e`, `CV = std_pop(n_e) / mean(n_e)`. Counts must sum to `T k` and no expert count may exceed `T`.
- Churn: `1 - |S_t ∩ S_(t-1)| / k`; exact-set-change is the indicator that the intersection has size below `k`. Sets are aligned by sample ID, token position and layer.
- Oracle overlap: `|S ∩ top-k(q)| / k`, where `q` is recomputed from current W1 and current hidden states.
- Activation coverage: selected activation mass divided by total activation mass for tokens with positive total mass. Zero-mass token counts remain explicit.

Layer summaries use population standard deviation (`ddof=0`); seed summaries use sample standard deviation (`ddof=1`). Load balance does not compute Gini or maximum share.
