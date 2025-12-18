# EPFL ML Project 2 — Tweet Sentiment (Text Classification)

This repository contains code for the EPFL ML Project 2 **tweet sentiment** task: predict whether a tweet originally contained a positive `:)` or negative `:(` emoticon, using only the remaining text.

## 1) Setup

- **Python**: 3.10+ recommended
- **Install PyTorch**: follow the official instructions for your platform (CPU/CUDA).
- **Install the remaining dependencies** (after PyTorch):

```bash
python3 -m pip install -U pip
python3 -m pip install numpy scipy scikit-learn transformers datasets
# Optional (only if you use LoRA)
python3 -m pip install peft
# Optional (only if you enable --demojize-emojis)
python3 -m pip install emoji
```

## 2) Data

Place the AIcrowd dataset files under:

```
project_text_classification_EPFL/
  data/twitter-datasets/
    train_pos.txt
    train_neg.txt
    train_pos_full.txt
    train_neg_full.txt
    test_data.txt
```

You can download the dataset from the AIcrowd challenge page (see `project2_description.pdf`).

### Optional: download via script

A helper `download_dataset.py` exists, but it depends on the `aicrowd` Python package / credentials. If it fails, download manually and unzip into `data/twitter-datasets/`.

## 3) Quickstart — Baselines

All commands below assume you run them from `project_text_classification_EPFL/`.

### A) Hashed n-gram baseline (fast, no embeddings needed)

```bash
python3 baseline_classifier.py --use-full --device auto \
  --representation hash --ngram-max 2 --num-features 262144 \
  --epochs 5 --batch-size 2048 \
  --output baseline_hash.csv
```

### B) Embedding-bag baseline (requires `vocab.pkl` + `embeddings.npy`)

If you already have `vocab.pkl` and `embeddings.npy`, you can run:

```bash
python3 baseline_classifier.py --use-full --device auto \
  --representation vocab --vocab-path vocab.pkl --embeddings-path embeddings.npy \
  --epochs 5 --batch-size 2048 \
  --output baseline_vocab.csv
```

## 4) (Optional) Build in-domain GloVe resources

This pipeline builds a vocabulary, a co-occurrence matrix, then trains GloVe embeddings.

```bash
./build_vocab.sh
./cut_vocab.sh
python3 pickle_vocab.py
python3 cooc.py
python3 glove_solution.py
```

Outputs:
- `vocab.pkl`
- `cooc.pkl`
- `embeddings.npy`

Notes:
- `build_vocab.sh` / `cut_vocab.sh` use standard Unix tools (`cat`, `sed`, `sort`, `uniq`, `grep`).
- `cooc.py` can take time and uses the smaller `train_pos.txt` / `train_neg.txt` by default.

## 5) BERTweet fine-tuning (3 setups)

The main transformer script is `distilbert_classifier.py` (it supports BERTweet via `--model-name vinai/bertweet-base`).

### A) Head-only training (frozen encoder)

```bash
python3 distilbert_classifier.py --use-full --device cuda \
  --model-name vinai/bertweet-base --freeze-encoder \
  --val-size 0.05 --epochs 3 \
  --output bertweet_head.csv
```

Optional: include averaged-GloVe fusion (needs `embeddings.npy` + `vocab.pkl`):

```bash
python3 distilbert_classifier.py --use-full --device cuda \
  --model-name vinai/bertweet-base --freeze-encoder \
  --val-size 0.05 --epochs 3 \
  --embedding-path embeddings.npy --embedding-vocab vocab.pkl \
  --estimate-embedding-scale \
  --output bertweet_head_fusion.csv
```

### B) LoRA (parameter-efficient fine-tuning)

Requires `peft`:

```bash
python3 distilbert_classifier.py --use-full --device cuda \
  --model-name vinai/bertweet-base --use-lora --lora-target-modules auto \
  --val-size 0.05 --epochs 3 \
  --output bertweet_lora.csv
```

### C) Full fine-tuning (+ optional SWA)

```bash
python3 distilbert_classifier.py --use-full --device cuda \
  --model-name vinai/bertweet-base \
  --val-size 0.05 --epochs 3 \
  --output bertweet_full.csv
```

Optional SWA:

```bash
python3 distilbert_classifier.py --use-full --device cuda \
  --model-name vinai/bertweet-base \
  --val-size 0.05 --epochs 3 \
  --use-swa --swa-start-epoch 1 --swa-freq 300 \
  --output bertweet_full_swa.csv
```

A recommended configuration used during our runs is documented in `run_distilbert.sh`.

## 6) Submission format

All training scripts write a CSV with header:

```
Id,Prediction
```

Predictions are mapped to **{-1, +1}** (negative → `-1`, positive → `+1`). Upload the CSV to AIcrowd.
