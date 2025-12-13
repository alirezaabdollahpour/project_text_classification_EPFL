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
  --val-size 0.05 --metric accuracy --patience 2 --min-delta 0.001 \
  --per-device-train-batch-size 256 --per-device-eval-batch-size 256 \
  --lr 4e-5 --weight-decay 0.01 --betas 0.9 0.999 0.9999 --alpha-mix 2.0 \
  --beta3-warmup 2000 --alpha-warmup 2000 \
  --gradient-accumulation-steps 2 \
  --eta-min 1e-6 \
  --output baseline_submission_distilbert.csv \
  --embedding-path "$EMB_PATH" --embedding-vocab "$VOCAB_PATH" \
  # --use-swa --swa-start-epoch 1 --swa-freq 300 --swa-lr 2e-5 \