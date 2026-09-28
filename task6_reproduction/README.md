# Task 6 Phase A / F1

Independent full-finetuning testbed for eight routing methods on the existing
T5-small SST-2/MNLI dense checkpoints and balanced K-Means expert splits. The
original dense checkpoints remain fixed `Dense-init` references, while a
matched `Dense-fullFT` baseline is trained for ten epochs on all three seeds.

The formal scope is R2, R2-soft, R4o, R4d, R4o-hard, G1, G2-0.001 and G4 at
`k={6,13,19,26}` with seeds `0,1,2`. Every arm trains the full backbone. The
matched dense baseline has no routing or `k` expansion. The repository does not
implement random expert splits, G0/G3, co-activation, Gini or maximum-share
metrics.

Use [RUNBOOK.md](RUNBOOK.md) as the only execution guide. Formal experiments
must not start until Phase 0, the fullFT smoke suite, resume equivalence and the
two heavy-path preflight checks pass on the target server.

Large inputs and outputs are not tracked by Git. `configs/local/*.yaml` binds
verified Task 5 inputs read-only and writes Task 6 outputs to a separate root.
For a routed-only run already in progress under the earlier protocol, use the
additive Dense-fullFT workflow in RUNBOOK §5 instead of loading its checkpoints
with this revised protocol.
