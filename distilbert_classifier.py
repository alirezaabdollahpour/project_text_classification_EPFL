from __future__ import annotations

import argparse
import random
from pathlib import Path
import math
from typing import Iterable, List, Tuple

import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)

from ademamix import AdEMAMix, alpha_scheduler, beta3_scheduler

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
    parser.add_argument("--betas", type=float, nargs=3, default=(0.9, 0.999, 0.9999), help="AdEMAMix betas (beta1 beta2 beta3).")
    parser.add_argument("--alpha-mix", type=float, default=2.0, help="AdEMAMix alpha mixing coefficient.")
    parser.add_argument("--beta3-warmup", type=int, default=0, help="Warmup steps for beta3 (0 disables).")
    parser.add_argument("--alpha-warmup", type=int, default=0, help="Warmup steps for alpha mix (0 disables).")
    parser.add_argument("--per-device-train-batch-size", type=int, default=32, help="Train batch size per device.")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=64, help="Eval batch size per device.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--logging-steps", type=int, default=100, help="Log every N steps.")
    parser.add_argument("--use-cosine-scheduler", action="store_true", help="Use cosine annealing LR scheduler for AdEMAMix.")
    parser.add_argument("--eta-min", type=float, default=0.0, help="Minimum LR for cosine scheduler.")
    parser.add_argument("--use-swa", action="store_true", help="Enable stochastic weight averaging.")
    parser.add_argument("--swa-start-epoch", type=int, default=1, help="Epoch (1-indexed) to start SWA updates.")
    parser.add_argument("--swa-freq", type=int, default=100, help="Apply SWA update every N optimizer steps after start.")
    parser.add_argument("--swa-lr", type=float, default=None, help="SWA learning rate; defaults to base lr if not set.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping max norm (0 to disable).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", type=Path, default=Path("baseline_submission_distilbert.csv"), help="Submission CSV path.")
    parser.add_argument("--device", type=str, default="auto", help='Device string ("cuda", "cpu", or "auto").')
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience (epochs without improvement).")
    parser.add_argument("--metric", type=str, choices=["accuracy", "eval_loss"], default="accuracy", help="Metric for best model.")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum improvement to reset patience.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 mixed precision if supported.")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def preprocess_tweet(text: str) -> str:
    tokens = []
    for tok in text.split():
        if tok.startswith("@") and len(tok) > 1:
            tokens.append("@user")
        elif tok.startswith("http"):
            tokens.append("http")
        else:
            tokens.append(tok)
    return " ".join(tokens)


def read_labeled(path: Path, label: int, limit: int | None) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            cleaned = line.strip()
            if cleaned:
                texts.append(preprocess_tweet(cleaned))
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
            texts.append(preprocess_tweet(tweet.strip()))
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


def evaluate(model, loader: DataLoader, device: torch.device):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    with torch.no_grad():
        for batch in loader:
            label_key = "labels" if "labels" in batch else "label"
            labels = batch[label_key].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != label_key}
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)
            preds = torch.argmax(outputs.logits, dim=-1)
            total_correct += (preds == labels).sum().item()
            total_seen += labels.size(0)
            total_loss += loss.item() * labels.size(0)
    avg_loss = total_loss / max(total_seen, 1)
    acc = total_correct / max(total_seen, 1)
    return avg_loss, acc


def clean_state_dict(state_dict: dict) -> dict:
    """Remove SWA/AveragedModel prefixes and non-parameter entries."""
    clean = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_k = k[len("module.") :]
        else:
            new_k = k
        if new_k == "n_averaged":
            continue
        clean[new_k] = v.detach().cpu().clone()
    return clean


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    set_seed(args.seed)
    random.seed(args.seed)
    print(f"Using device: {device}")
    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    print(
        f"Effective train batch size (per device x grad accumulation): "
        f"{args.per_device_train_batch_size} x {args.gradient_accumulation_steps} = {effective_batch}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )
    model.to(device)

    texts, labels = load_train_data(args.data_dir, args.use_full, args.limit_per_class)
    train_ds, val_ds = build_datasets(tokenizer, texts, labels, args.val_size, args.max_length, args.seed)

    test_ids, test_texts = load_test_data(args.data_dir / "test_data.txt")
    test_ds = build_test_dataset(tokenizer, test_texts, args.max_length)

    # Prepare torch datasets/dataloaders
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = args.epochs * steps_per_epoch

    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    alpha_sched = alpha_scheduler(
        alpha_end=args.alpha_mix,
        alpha_start=0.0,
        warmup=args.alpha_warmup or 0,
    )
    beta3_sched = beta3_scheduler(
        beta_end=args.betas[2],
        beta_start=args.betas[0],
        warmup=args.beta3_warmup or 0,
    )

    optimizer = AdEMAMix(
        optimizer_grouped_parameters,
        lr=args.lr,
        betas=tuple(args.betas),
        alpha=args.alpha_mix,
        beta3_scheduler=beta3_sched,
        alpha_scheduler=alpha_sched,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.eta_min)
        if args.use_cosine_scheduler
        else None
    )

    use_amp = False  # AMP disabled per request
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    use_swa = args.use_swa
    swa_model = AveragedModel(model) if use_swa else None
    swa_start_step = max(0, args.swa_start_epoch - 1) * steps_per_epoch
    swa_scheduler = (
        SWALR(optimizer, swa_lr=args.swa_lr if args.swa_lr is not None else args.lr)
        if use_swa
        else None
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    best_metric = None
    best_state = None
    best_epoch = 0
    no_improve = 0
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            label_key = "labels" if "labels" in batch else "label"
            labels = batch[label_key].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != label_key}
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**inputs)
                loss = loss_fn(outputs.logits, labels)
                loss = loss / args.gradient_accumulation_steps

            if not torch.isfinite(loss):
                print(f"[epoch {epoch} step {global_step}] non-finite loss, skipping step.")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            if step % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm and args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if use_swa and global_step >= swa_start_step:
                    if (global_step - swa_start_step) % args.swa_freq == 0:
                        swa_model.update_parameters(model)
                elif scheduler is not None and (not use_swa or global_step < swa_start_step):
                    # Stop cosine decay after SWA starts to keep LR flatter during averaging.
                    scheduler.step()

            running_loss += loss.item() * labels.size(0)
            if global_step > 0 and global_step % args.logging_steps == 0:
                avg_loss = running_loss / (step * train_loader.batch_size)
                print(f"[epoch {epoch} step {global_step}] train loss {avg_loss:.4f}")

        # Choose model to evaluate (SWA if active)
        eval_model = swa_model if use_swa and global_step >= swa_start_step else model
        val_loss, val_acc = evaluate(eval_model, val_loader, device)
        print(f"[epoch {epoch}/{args.epochs}] val loss {val_loss:.4f} acc {val_acc:.4f}")

        if use_swa and global_step >= swa_start_step:
            swa_scheduler.step()

        current_metric = val_acc if args.metric == "accuracy" else val_loss
        improved = False
        if best_metric is None:
            improved = True
        elif args.metric == "accuracy":
            improved = current_metric > best_metric + args.min_delta
        else:
            improved = current_metric < best_metric - args.min_delta

        if improved:
            best_metric = current_metric
            best_epoch = epoch
            # save the evaluation model parameters
            best_state = clean_state_dict(eval_model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"Loaded best checkpoint from epoch {best_epoch} (metric={best_metric:.4f}).")
        if use_swa and global_step >= swa_start_step:
            print("Updating batch norm statistics for SWA weights...")
            update_bn(train_loader, model)

    # Predict on test set
    model.eval()
    test_preds: List[int] = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            test_preds.extend(preds.cpu().tolist())

    write_submission(test_ids, test_preds, args.output)


if __name__ == "__main__":
    main()
