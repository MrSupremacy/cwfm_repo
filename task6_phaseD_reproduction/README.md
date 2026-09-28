# Task 6 Phase D/F0 reproduction

Independent two-stage repository for:

1. four-domain Dense-MT full fine-tuning from the original pretrained ReLU T5-small;
2. E64/E128/E256 frozen-backbone routed experiments on the reviewed Dense-MT macro-best.

The four domains are SST-2, MNLI-matched, QNLI and QQP. SST-2/MNLI/QNLI use
accuracy; QQP uses the mean of accuracy and positive-class F1 for the native
macro score.

The completed E64 matrix uses k={6,10,13}. The supplement uses E128
k={13,19,26} and E256 k={26,38,51}; all are displayed as 10%/15%/20%.
Each E uses the same six arms R2/R4o/R4d/G1/G2-0.001/G4, three train seeds for
the five learnable arms, and one deterministic R2 state per k. The supplement
adds 90 trained runs, 6 static states and 990 router checkpoints to the same
`routed02` run id without reopening or relabeling the completed E64 protocol.

Dense training, Dense evaluation/reporting, best export, split generation and
routed training are separate commands. No command automatically crosses the
Dense review gate. Large outputs use /mnt/luoyulin_ckpt; pretrained/data,
exported Dense best and all three seed-1 splits use /mnt/luoyulin_code. E128
and E256 splits are regenerated from the Phase D Dense-MT best; Phase B
task-specific weights and labels are not valid inputs.

The Dense preflight has a separate formal-batch smoke command that performs one
batch=256 forward/backward without an optimizer step or checkpoint.

See RUNBOOK.md for execution.
