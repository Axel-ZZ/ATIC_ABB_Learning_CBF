#!/usr/bin/env bash
# Grid sweep over lam_constraint x gamma for the minibatch neural-CBF trainer.
# Each config gets its own wandb run name and its own out-dir so nothing is
# overwritten. Override the grids/dataset/epochs from the environment, e.g.:
#   DATASET=maze NUM_EPOCHS=20000 LAM_CONSTRAINTS="15 50 150" GAMMAS="0.0 0.02 0.05" ./sweep.sh

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET="${DATASET:-maze}"
NUM_EPOCHS="${NUM_EPOCHS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
read -ra LAM_CONSTRAINTS <<< "${LAM_CONSTRAINTS:-15 50 150}"
read -ra GAMMAS          <<< "${GAMMAS:-0.0 0.02 0.05}"

for lc in "${LAM_CONSTRAINTS[@]}"; do
  for g in "${GAMMAS[@]}"; do
    name="lc${lc}_g${g}"
    echo "============================== $name =============================="
    python "$HERE/train_nn_cbf_minibatch.py" \
      --dataset "$DATASET" \
      --num-epochs "$NUM_EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --lam-constraint "$lc" \
      --gamma "$g" \
      --wandb-project neural-cbf \
      --wandb-run-name "sweep_${DATASET}_${name}" \
      --wandb-mode online \
      --out-dir "$HERE/../runs/models/sweep_${DATASET}/${name}"
  done
done
echo "[sweep done] models under runs/models/sweep_${DATASET}/"
