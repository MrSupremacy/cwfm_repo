# Architecture

The repository implements only Task 6 Phase A/F1 with K-Means expert labels. The dependency direction is `common/data/routing → substrate/context → training/capture → metrics → aggregation/visualization`.

The dense T5 FFN modules retain their original W1/W2 parameters inside `MaskedFFN`. Every model parameter remains trainable. Router parameters form a second Adam parameter group; R2 and R2-soft have no router parameters. Fixed initial raw centroids and expert labels are prepared once from the verified dense checkpoint. `Dense-fullFT` uses the same full backbone training group without a router, masking or expert budget.

Every trained condition is identified by `(task, arm, variant, k, seed, run_id)`. The routed Cartesian product contains 2 tasks × 8 arms × 4 budgets × 3 seeds = 192 independent runs. The matched dense baseline adds 2 tasks × 3 seeds = 6 runs without `k`, for 198 trained runs in total. Each run saves the complete model and training state at step 0 and after each of 10 epochs.

Capture A follows every trained checkpoint and the two static `Dense-init` references. Routed checkpoints additionally receive C+D throughout training and B at the validation-selected best/final states. Metrics only read completed captures. Result aggregation requires all seeds and all scheduled state roles; missing data causes failure.
