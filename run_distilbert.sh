#!/usr/bin/env bash
set -euo pipefail

python distilbert_classifier.py --use-full --device cuda --epochs 3 \
  --model-name vinai/bertweet-base \
  --val-size 0.05 --metric accuracy --patience 2 --min-delta 0.001 \
  --per-device-train-batch-size 256 --per-device-eval-batch-size 256 \
  --lr 2e-5 --weight-decay 0.01 --betas 0.9 0.999 0.9999 --alpha-mix 2.0 \
  --beta3-warmup 2000 --alpha-warmup 2000 \
  --gradient-accumulation-steps 2 \
  --eta-min 1e-6 \
  --output baseline_submission_distilbert.csv \
  # --use-swa --swa-start-epoch 3 --swa-freq 200 --swa-lr 3e-5 \
  
