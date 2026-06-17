#!/usr/bin/env bash
# Train the neural CBF on one of the stage-2 datasets.
# Edit the flags below by hand, or call train_nn_cbf.py directly (e.g. from a
# sweep script) with the same --flags.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec python "$HERE/train_nn_cbf.py" \
  --dataset single_obstacle \
  --num-epochs 50000 \
  --lr 0.1 \
  --seed 5433 \
  --safe-value 1.5 \
  --unsafe-value 0.5 \
  --gamma 0.05 \
  --lam-constraint 15.0 \
  --lam-boundary 1.0 \
  --lam-safe 2.0 \
  --lam-unsafe 2.0 \
  --lam-param 0.1 \
  --log-every 1000 \
  --eval-every 10000 \
  --wandb-project neural-cbf \
  --wandb-mode online
