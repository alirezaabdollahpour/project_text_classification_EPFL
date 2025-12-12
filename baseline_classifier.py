from __future__ import annotations

import argparse
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ademamix import AdEMAMix


DEFAULT_DATA_DIR = Path("data/twitter-datasets")


@dataclass
class DatasetConfig:
    data_dir: Path
    use_full_train: bool
    limit_per_class: int | None

    @property
    def pos_path(self) -> Path:
        name = "train_pos_full.txt" if self.use_full_train else "train_pos.txt"
        return self.data_dir / name

    @property
    def neg_path(self) -> Path:
        name = "train_neg_full.txt" if self.use_full_train else "train_neg.txt"
        return self.data_dir / name

    @property
    def test_path(self) -> Path:
        return self.data_dir / "test_data.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GPU tweet sentiment classifier.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the twitter datasets.",
    )
    parser.add_argument(
        "--use-full",
        action="store_true",
        help="Use the full 2.5M tweet training set instead of the 200k sample.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Optional cap on the number of positive/negative training tweets for quick experiments.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction of the training data to hold out for validation (0 disables validation).",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="Maximum n-gram size (1=unigram only, 2=unigram+bigram).",
    )
    parser.add_argument(
        "--num-features",
        type=int,
        default=2**18,
        help="Number of hash buckets for the feature space.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs for training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size for the DataLoader.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate for AdEMAMix.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Early stopping patience (number of epochs without improvement before stopping).",
    )
    parser.add_argument(
        "--early-metric",
        type=str,
        choices=["loss", "acc"],
        default="loss",
        help="Metric to monitor for early stopping.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum improvement required to reset early stopping patience.",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs=3,
        default=(0.9, 0.999, 0.9999),
        help="Beta coefficients for AdEMAMix (beta1 beta2 beta3).",
    )
    parser.add_argument(
        "--alpha-mix",
        type=float,
        default=2.0,
        help="Alpha mixing coefficient for AdEMAMix.",
    )
    parser.add_argument(
        "--beta3-warmup",
        type=int,
        default=0,
        help="Warmup steps for beta3 (0 disables).",
    )
    parser.add_argument(
        "--alpha-warmup",
        type=int,
        default=0,
        help="Warmup steps for alpha mix (0 disables).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-6,
        help="Weight decay for AdEMAMix.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("baseline_submission.csv"),
        help="Where to write the submission CSV.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=13,
        help="Random seed for train/val split and initialization.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Target device: "cuda", "cpu", or "auto" to pick if CUDA is available.',
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase input tweets before hashing tokens.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_lines(path: Path, limit: int | None = None) -> List[str]:
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            cleaned = line.strip()
            if cleaned:
                texts.append(cleaned)
    return texts


def load_labeled_data(config: DatasetConfig) -> Tuple[List[str], List[int]]:
    pos_texts = read_lines(config.pos_path, config.limit_per_class)
    neg_texts = read_lines(config.neg_path, config.limit_per_class)
    texts = pos_texts + neg_texts
    labels = [1] * len(pos_texts) + [0] * len(neg_texts)
    print(
        f"Loaded {len(texts)} tweets "
        f"(positive: {len(pos_texts)}, negative: {len(neg_texts)}) "
        f"from {config.data_dir}"
    )
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
    print(f"Loaded {len(ids)} test tweets from {path}")
    return ids, texts


def hash_token(token: str, num_features: int, seed: int) -> int:
    # Stable hash to avoid Python's hash randomization between runs.
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8, person=str(seed).encode("utf-8")).digest()
    return int.from_bytes(digest, byteorder="little") % num_features


def text_to_features(
    text: str,
    num_features: int,
    ngram_max: int,
    seed: int,
    lowercase: bool,
) -> List[int]:
    tokens = text.split()
    if lowercase:
        tokens = [tok.lower() for tok in tokens]

    features: List[int] = []
    for tok in tokens:
        features.append(hash_token(tok, num_features, seed))

    if ngram_max >= 2 and len(tokens) > 1:
        for i in range(len(tokens) - 1):
            bigram = tokens[i] + "_" + tokens[i + 1]
            features.append(hash_token(bigram, num_features, seed + 17))

    return features or [0]  # ensure at least one index for empty tweets


class HashedTweetDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int] | None,
        num_features: int,
        ngram_max: int,
        seed: int,
        lowercase: bool,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.num_features = num_features
        self.ngram_max = ngram_max
        self.seed = seed
        self.lowercase = lowercase

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        feats = text_to_features(
            self.texts[idx],
            self.num_features,
            self.ngram_max,
            self.seed,
            self.lowercase,
        )
        if self.labels is None:
            return feats
        return feats, float(self.labels[idx])


def collate_train(batch):
    feature_lists, labels = zip(*batch)
    offsets: List[int] = []
    token_ids: List[int] = []
    cursor = 0
    for feats in feature_lists:
        offsets.append(cursor)
        token_ids.extend(feats)
        cursor += len(feats)
    tokens = torch.tensor(token_ids, dtype=torch.long)
    offsets_tensor = torch.tensor(offsets, dtype=torch.long)
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    return tokens, offsets_tensor, label_tensor


def collate_infer(batch):
    feature_lists, ids = zip(*batch)
    offsets: List[int] = []
    token_ids: List[int] = []
    cursor = 0
    for feats in feature_lists:
        offsets.append(cursor)
        token_ids.extend(feats)
        cursor += len(feats)
    tokens = torch.tensor(token_ids, dtype=torch.long)
    offsets_tensor = torch.tensor(offsets, dtype=torch.long)
    id_tensor = torch.tensor(ids, dtype=torch.long)
    return tokens, offsets_tensor, id_tensor


class HashedLogisticModel(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(num_features, 1, mode="sum", sparse=False)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, tokens: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        logits = self.embedding(tokens, offsets).squeeze(1) + self.bias
        return logits


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for tokens, offsets, labels in loader:
        tokens = tokens.to(device)
        offsets = offsets.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens, offsets)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) >= 0.5).long()
            total_correct += (preds == labels.long()).sum().item()
            total_seen += labels.numel()
            total_loss += loss.item() * labels.size(0)
    avg_loss = total_loss / max(total_seen, 1)
    acc = total_correct / max(total_seen, 1)
    return avg_loss, acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    with torch.no_grad():
        for tokens, offsets, labels in loader:
            tokens = tokens.to(device)
            offsets = offsets.to(device)
            labels = labels.to(device)
            logits = model(tokens, offsets)
            loss = criterion(logits, labels)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            total_correct += (preds == labels.long()).sum().item()
            total_seen += labels.numel()
            total_loss += loss.item() * labels.size(0)
    avg_loss = total_loss / max(total_seen, 1)
    acc = total_correct / max(total_seen, 1)
    return avg_loss, acc


def fit_model(
    texts: Sequence[str],
    labels: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
) -> nn.Module:
    set_seed(args.random_state)
    if args.val_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            texts,
            labels,
            test_size=args.val_size,
            stratify=labels,
            random_state=args.random_state,
        )
        train_ds = HashedTweetDataset(
            X_train,
            y_train,
            args.num_features,
            args.ngram_max,
            args.random_state,
            args.lowercase,
        )
        val_ds = HashedTweetDataset(
            X_val,
            y_val,
            args.num_features,
            args.ngram_max,
            args.random_state,
            args.lowercase,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_train,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_train,
            num_workers=0,
        )
        model = HashedLogisticModel(args.num_features).to(device)
        optimizer = AdEMAMix(
            model.parameters(),
            lr=args.lr,
            betas=tuple(args.betas),
            alpha=args.alpha_mix,
            beta3_warmup=args.beta3_warmup or None,
            alpha_warmup=args.alpha_warmup or None,
            weight_decay=args.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        best_state = None
        best_metric = None
        best_epoch = 0
        epochs_no_improve = 0

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(
                f"[epoch {epoch}/{args.epochs}] "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f}"
            )
            current_metric = val_loss if args.early_metric == "loss" else val_acc
            improved = False
            if best_metric is None:
                improved = True
            elif args.early_metric == "loss":
                improved = current_metric < best_metric - args.min_delta
            else:
                improved = current_metric > best_metric + args.min_delta

            if improved:
                best_metric = current_metric
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve > args.patience:
                    print(f"Early stopping triggered at epoch {epoch}; best epoch was {best_epoch}.")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)
            print(f"Loaded best checkpoint from epoch {best_epoch} (metric={best_metric:.4f}).")
    else:
        model = HashedLogisticModel(args.num_features).to(device)
        optimizer = AdEMAMix(
            model.parameters(),
            lr=args.lr,
            betas=tuple(args.betas),
            alpha=args.alpha_mix,
            beta3_warmup=args.beta3_warmup or None,
            alpha_warmup=args.alpha_warmup or None,
            weight_decay=args.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()
        train_ds = HashedTweetDataset(
            texts,
            labels,
            args.num_features,
            args.ngram_max,
            args.random_state,
            args.lowercase,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_train,
            num_workers=0,
        )
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            print(f"[epoch {epoch}/{args.epochs}] train loss {train_loss:.4f} acc {train_acc:.4f}")
    return model


def write_submission(ids: Iterable[int], preds: Iterable[int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_map = {0: -1, 1: 1}
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Id,Prediction\n")
        for idx, pred in zip(ids, preds):
            mapped = label_map.get(int(pred), 1)
            handle.write(f"{idx},{mapped}\n")
    print(f"Wrote predictions to {output_path.resolve()}")


def predict_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> List[int]:
    model.eval()
    preds: List[int] = []
    with torch.no_grad():
        for tokens, offsets, _ids in loader:
            tokens = tokens.to(device)
            offsets = offsets.to(device)
            logits = model(tokens, offsets)
            batch_preds = (torch.sigmoid(logits) >= 0.5).long().cpu().tolist()
            preds.extend(batch_preds)
    return preds


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    print(f"Using device: {device}")

    config = DatasetConfig(
        data_dir=args.data_dir,
        use_full_train=args.use_full,
        limit_per_class=args.limit_per_class,
    )

    texts, labels = load_labeled_data(config)
    model = fit_model(texts, labels, args, device)

    test_ids, test_texts = load_test_data(config.test_path)
    test_ds = HashedTweetDataset(
        test_texts,
        labels=None,
        num_features=args.num_features,
        ngram_max=args.ngram_max,
        seed=args.random_state,
        lowercase=args.lowercase,
    )
    test_pairs = list(zip(test_ds, test_ids))  # keep ordering
    test_loader = DataLoader(
        test_pairs,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_infer([(feats, tid) for feats, tid in batch]),
        num_workers=0,
    )
    test_preds = predict_on_loader(model, test_loader, device)
    write_submission(test_ids, test_preds, args.output)


if __name__ == "__main__":
    main()
