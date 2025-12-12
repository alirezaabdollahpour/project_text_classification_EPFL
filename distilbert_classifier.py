from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


DEFAULT_DATA_DIR = Path("data/twitter-datasets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for tweet sentiment classification.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory with twitter datasets.")
    parser.add_argument("--use-full", action="store_true", help="Use full training set (2.5M tweets).")
    parser.add_argument("--limit-per-class", type=int, default=None, help="Optional cap per class for quick runs.")
    parser.add_argument("--val-size", type=float, default=0.05, help="Validation split fraction.")
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased", help="HF model checkpoint.")
    parser.add_argument("--max-length", type=int, default=96, help="Max token length for tokenizer.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup-ratio", type=float, default=0.06, help="Warmup ratio for LR scheduler.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=32, help="Train batch size per device.")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=64, help="Eval batch size per device.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument(
        "--evaluation-strategy",
        type=str,
        choices=["no", "steps", "epoch"],
        default="epoch",
        help="When to run evaluation/save checkpoints.",
    )
    parser.add_argument("--eval-steps", type=int, default=500, help="Eval/save every N steps when using steps strategy.")
    parser.add_argument("--logging-steps", type=int, default=100, help="Log every N steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", type=Path, default=Path("baseline_submission_distilbert.csv"), help="Submission CSV path.")
    parser.add_argument("--output-dir", type=Path, default=Path("distilbert_outputs"), help="Checkpoint/logging directory.")
    parser.add_argument("--device", type=str, default="auto", help='Device string ("cuda", "cpu", or "auto").')
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience (epochs without improvement).")
    parser.add_argument("--metric", type=str, choices=["accuracy", "eval_loss"], default="accuracy", help="Metric for best model.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 mixed precision if supported.")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def read_labeled(path: Path, label: int, limit: int | None) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            cleaned = line.strip()
            if cleaned:
                texts.append(cleaned)
                labels.append(label)
    return texts, labels


def load_train_data(data_dir: Path, use_full: bool, limit_per_class: int | None) -> Tuple[List[str], List[int]]:
    pos_file = data_dir / ("train_pos_full.txt" if use_full else "train_pos.txt")
    neg_file = data_dir / ("train_neg_full.txt" if use_full else "train_neg.txt")
    pos_texts, pos_labels = read_labeled(pos_file, 1, limit_per_class)
    neg_texts, neg_labels = read_labeled(neg_file, 0, limit_per_class)
    texts = pos_texts + neg_texts
    labels = pos_labels + neg_labels
    print(f"Loaded {len(texts)} train tweets (pos: {len(pos_texts)}, neg: {len(neg_texts)}).")
    return texts, labels


def load_test_data(path: Path) -> Tuple[List[int], List[str]]:
    ids: List[int] = []
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            cleaned = line.rstrip("\n")
            if not cleaned:
                continue
            id_str, tweet = cleaned.split(",", 1)
            ids.append(int(id_str))
            texts.append(tweet.strip())
    print(f"Loaded {len(ids)} test tweets from {path}.")
    return ids, texts


def build_datasets(
    tokenizer: DistilBertTokenizerFast,
    texts: List[str],
    labels: List[int],
    val_size: float,
    max_length: int,
    seed: int,
):
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=val_size,
        stratify=labels,
        random_state=seed,
    )
    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )

    tokenized_train = train_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
    tokenized_val = val_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
    return tokenized_train, tokenized_val


def build_test_dataset(tokenizer: DistilBertTokenizerFast, texts: List[str], max_length: int):
    ds = Dataset.from_dict({"text": texts})

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )

    tokenized = ds.map(tokenize_batch, batched=True, remove_columns=["text"])
    return tokenized


def write_submission(ids: Iterable[int], preds: Iterable[int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Id,Prediction\n")
        for idx, pred in zip(ids, preds):
            mapped = -1 if int(pred) == 0 else 1
            handle.write(f"{idx},{mapped}\n")
    print(f"Wrote predictions to {output_path.resolve()}")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    set_seed(args.seed)
    random.seed(args.seed)
    print(f"Using device: {device}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name, use_fast=True)
    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )
    model.to(device)

    texts, labels = load_train_data(args.data_dir, args.use_full, args.limit_per_class)
    train_ds, val_ds = build_datasets(tokenizer, texts, labels, args.val_size, args.max_length, args.seed)

    test_ids, test_texts = load_test_data(args.data_dir / "test_data.txt")
    test_ds = build_test_dataset(tokenizer, test_texts, args.max_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels_arr = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = (preds == labels_arr).mean().item()
        return {"accuracy": accuracy}

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps if args.evaluation_strategy == "steps" else None,
        save_steps=args.eval_steps if args.evaluation_strategy == "steps" else None,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric,
        greater_is_better=(args.metric != "eval_loss"),
        save_total_limit=2,
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        report_to=[],
    )

    callbacks = []
    if args.patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()

    test_logits = trainer.predict(test_ds).predictions
    test_preds = np.argmax(test_logits, axis=-1)
    write_submission(test_ids, test_preds, args.output)


if __name__ == "__main__":
    main()
