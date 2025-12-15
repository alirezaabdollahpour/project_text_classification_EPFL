#!/usr/bin/env bash
set -euo pipefail

# Paths for precomputed GloVe-style embeddings and vocabulary.
EMB_PATH=${EMB_PATH:-embeddings.npy}
VOCAB_PATH=${VOCAB_PATH:-vocab.pkl}

if [[ ! -f "$EMB_PATH" ]]; then
  echo "Missing embeddings file: $EMB_PATH (override with EMB_PATH=/path/to/embeddings.npy)" >&2
  exit 1
fi

if [[ ! -f "$VOCAB_PATH" ]]; then
  echo "Missing vocab file: $VOCAB_PATH (override with VOCAB_PATH=/path/to/vocab.pkl)" >&2
  exit 1
fi

python distilbert_classifier.py --use-full --device cuda --epochs 3 \
  --model-name vinai/bertweet-base \
  --freeze-encoder \
  --val-size 0.05 --metric accuracy --patience 2 --min-delta 0.001 \
  --per-device-train-batch-size 256 --per-device-eval-batch-size 256 \
  --lr 1e-3 --weight-decay 0.001 --betas 0.9 0.999 0.9999 --alpha-mix 8.0 \
  --beta3-warmup 2000 --alpha-warmup 2000 \
  --gradient-accumulation-steps 2 \
  --output baseline_submission_distilbert_head.csv \
  --eval-steps 500 \
  --embedding-path "$EMB_PATH" --embedding-vocab "$VOCAB_PATH" \
  --estimate-embedding-scale --estimate-sample-size 10000 \
  # --demojize-emojis \
  # --min-clean-tokens 3 \
  # --outlier-url-mention-ratio 0.7 \
  # --outlier-allcaps-ratio 0.9 \
  # --augment-prob 0.05 --augment-max-per-class 50000 \
  # --imbalance-threshold 1.1

# Optional extras:
#  --max-clean-tokens 80 \
#  --use-lora \
#  --lora-target-modules 'auto' \
#  --lora-r 32 --lora-alpha 16 --lora-dropout 0.05 \
#  --fusion-dropout 0.1 \
#  --use-linear-scheduler --lr-warmup-steps 1000 \
#  --use-swa --swa-start-epoch 1 --swa-freq 300 --swa-lr 2e-5 \
