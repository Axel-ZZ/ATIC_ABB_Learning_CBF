#!/usr/bin/env bash
# Grid sweep over lam_constraint x lam_unsafe for the minibatch neural-CBF
# trainer, gamma pinned at the best descent value (0.05), full-length runs.
# Goal: push min_r up WITHOUT collapsing the unsafe margin.
# Each config gets its own wandb run name + out-dir so nothing is overwritten.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET="${DATASET:-maze}"
NUM_EPOCHS="${NUM_EPOCHS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
GAMMA="${GAMMA:-0.05}"
read -ra LAM_CONSTRAINTS <<< "${LAM_CONSTRAINTS:-150 300}"
read -ra LAM_UNSAFES     <<< "${LAM_UNSAFES:-2 8 20}"

for lc in "${LAM_CONSTRAINTS[@]}"; do
  for lu in "${LAM_UNSAFES[@]}"; do
    name="lc${lc}_lu${lu}"
    echo "============================== $name =============================="
    python "$HERE/train_nn_cbf_minibatch.py" \
      --dataset "$DATASET" \
      --num-epochs "$NUM_EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --gamma "$GAMMA" \
      --lam-constraint "$lc" \
      --lam-unsafe "$lu" \
      --wandb-project neural-cbf \
      --wandb-run-name "sweep_${DATASET}_${name}" \
      --wandb-mode online \
      --out-dir "$HERE/../runs/models/sweep_${DATASET}_lam_unsafe/${name}"
  done
done
echo "[sweep done] models under runs/models/sweep_${DATASET}_lam_unsafe/"
