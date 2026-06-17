#!/usr/bin/env bash
# Loss-change sweep: make SEPARATION dominate so h is forced off the flat-0
# attractor. Sweeps lam_safe x lam_unsafe (both raised well above the descent
# weight), with lam_constraint held modest and gamma pinned. Goal: get
# safe viol% below 100 (i.e. h actually reaches the safe margin somewhere).
# Each config gets its own wandb run name + out-dir.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET="${DATASET:-maze}"
NUM_EPOCHS="${NUM_EPOCHS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
GAMMA="${GAMMA:-0.05}"
LAM_CONSTRAINT="${LAM_CONSTRAINT:-15}"
read -ra LAM_SAFES   <<< "${LAM_SAFES:-20 50}"
read -ra LAM_UNSAFES <<< "${LAM_UNSAFES:-20 50}"

for ls in "${LAM_SAFES[@]}"; do
  for lu in "${LAM_UNSAFES[@]}"; do
    name="ls${ls}_lu${lu}"
    echo "============================== $name =============================="
    python "$HERE/train_nn_cbf_minibatch.py" \
      --dataset "$DATASET" \
      --num-epochs "$NUM_EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --gamma "$GAMMA" \
      --lam-constraint "$LAM_CONSTRAINT" \
      --lam-safe "$ls" \
      --lam-unsafe "$lu" \
      --wandb-project neural-cbf \
      --wandb-run-name "sweep_${DATASET}_${name}" \
      --wandb-mode online \
      --out-dir "$HERE/../runs/models/sweep_${DATASET}_separation/${name}"
  done
done
echo "[sweep done] models under runs/models/sweep_${DATASET}_separation/"
