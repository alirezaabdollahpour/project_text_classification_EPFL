from __future__ import annotations

import argparse
import hashlib
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
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
        "--representation",
        type=str,
        choices=["vocab", "hash"],
        default="vocab",
        help="Feature representation: pretrained vocab/embeddings or hashing-trick baseline.",
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=Path("vocab.pkl"),
        help="Path to vocab.pkl containing the token to index mapping for embeddings.",
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=Path("embeddings.npy"),
        help="Path to embeddings.npy aligned with vocab.pkl.",
    )
    parser.add_argument(
        "--embedding-trainable",
        action="store_true",
        help="Finetune the pretrained embeddings instead of freezing them.",
    )
    parser.add_argument(
        "--embedding-dropout",
        type=float,
        default=0.2,
        help="Dropout applied after pooling embeddings.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for the classifier MLP (0 keeps logistic regression head).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=60,
        help="Maximum number of tokens per tweet (applied before pooling).",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="Maximum n-gram size for hashed features (1=unigram only, 2=unigram+bigram).",
    )
    parser.add_argument(
        "--num-features",
        type=int,
        default=2**18,
        help="Number of hash buckets for the feature space (used when representation=hash).",
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
        help="Learning rate for AdEMAMix (used for all params unless overridden).",
    )
    parser.add_argument(
        "--lr-embed",
        type=float,
        default=None,
        help=(
            "Optional learning rate for the embedding matrix when finetuning embeddings "
            "(only used with --representation vocab --embedding-trainable). Defaults to --lr."
        ),
    )
    parser.add_argument(
        "--use-linear-scheduler",
        action="store_true",
        help="Use linear warmup then decay LR scheduler for AdEMAMix.",
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=None,
        help=(
            "Warmup steps for linear LR scheduler; if omitted uses a sensible default when "
            "--use-linear-scheduler is set (set to 0 to disable warmup explicitly)."
        ),
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
        default="acc",
        help="Metric to monitor for early stopping (default uses validation accuracy).",
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
        help="Lowercase input tweets before tokenization.",
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
    max_tokens: int | None,
) -> List[int]:
    tokens = text.split()
    if lowercase:
        tokens = [tok.lower() for tok in tokens]
    if max_tokens is not None and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

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
        max_tokens: int | None,
        ids: Sequence[int] | None = None,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.ids = ids
        self.num_features = num_features
        self.ngram_max = ngram_max
        self.seed = seed
        self.lowercase = lowercase
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        feats = text_to_features(
            self.texts[idx],
            self.num_features,
            self.ngram_max,
            self.seed,
            self.lowercase,
            self.max_tokens,
        )
        if self.labels is None:
            if self.ids is None:
                return feats
            return feats, self.ids[idx]
        return feats, float(self.labels[idx])


def load_vocab_and_embeddings(
    vocab_path: Path, embeddings_path: Path
) -> Tuple[Dict[str, int], torch.Tensor, int]:
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    with vocab_path.open("rb") as handle:
        vocab: Dict[str, int] = pickle.load(handle)

    embeddings_np = np.load(embeddings_path)
    if embeddings_np.ndim != 2:
        raise ValueError(f"Expected 2D embeddings matrix, got shape {embeddings_np.shape}.")
    if embeddings_np.shape[0] < len(vocab):
        raise ValueError(
            f"Embeddings rows ({embeddings_np.shape[0]}) smaller than vocab size ({len(vocab)})."
        )
    embedding_tensor = torch.tensor(embeddings_np[: len(vocab)], dtype=torch.float32)
    unk_vector = embedding_tensor.mean(dim=0, keepdim=True)
    embedding_tensor = torch.cat([embedding_tensor, unk_vector], dim=0)
    unk_idx = embedding_tensor.size(0) - 1
    print(
        f"Loaded vocab with {len(vocab)} entries and embeddings "
        f"dim={embedding_tensor.size(1)} (unk index {unk_idx})."
    )
    return vocab, embedding_tensor, unk_idx


def split_tokens(text: str, lowercase: bool, max_tokens: int | None) -> List[str]:
    tokens = text.split()
    if lowercase:
        tokens = [tok.lower() for tok in tokens]
    if max_tokens is not None and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokens


class VocabTweetDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int] | None,
        vocab: Dict[str, int],
        unk_idx: int,
        lowercase: bool,
        max_tokens: int | None,
        ids: Sequence[int] | None = None,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.ids = ids
        self.vocab = vocab
        self.unk_idx = unk_idx
        self.lowercase = lowercase
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        tokens = split_tokens(self.texts[idx], self.lowercase, self.max_tokens)
        feats = [self.vocab.get(tok, self.unk_idx) for tok in tokens]
        if not feats:
            feats = [self.unk_idx]
        if self.labels is None:
            if self.ids is None:
                return feats
            return feats, self.ids[idx]
        return feats, float(self.labels[idx])


def _merge_features(feature_lists: Sequence[Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    offsets: List[int] = []
    token_ids: List[int] = []
    cursor = 0
    for feats in feature_lists:
        offsets.append(cursor)
        token_ids.extend(feats)
        cursor += len(feats)
    tokens = torch.tensor(token_ids, dtype=torch.long)
    offsets_tensor = torch.tensor(offsets, dtype=torch.long)
    return tokens, offsets_tensor


def collate_train(batch):
    feature_lists, labels = zip(*batch)
    tokens, offsets_tensor = _merge_features(feature_lists)
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    return tokens, offsets_tensor, label_tensor


def collate_infer(batch):
    feature_lists, ids = zip(*batch)
    tokens, offsets_tensor = _merge_features(feature_lists)
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


class EmbeddingClassifier(nn.Module):
    def __init__(
        self,
        embedding_weight: torch.Tensor,
        trainable: bool,
        dropout: float,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        num_embeddings, emb_dim = embedding_weight.shape
        self.embedding = nn.EmbeddingBag(
            num_embeddings,
            emb_dim,
            mode="mean",
            sparse=False,
            _weight=embedding_weight,
        )
        self.embedding.weight.requires_grad = trainable

        layers: List[nn.Module] = [nn.LayerNorm(emb_dim), nn.Dropout(dropout)]
        if hidden_dim and hidden_dim > 0:
            layers.extend(
                [
                    nn.Linear(emb_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                ]
            )
        else:
            layers.append(nn.Linear(emb_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, tokens: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        pooled = self.embedding(tokens, offsets)
        logits = self.classifier(pooled).squeeze(1)
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
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
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
        if scheduler is not None:
            scheduler.step()

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
    vocab_info: Tuple[Dict[str, int], torch.Tensor, int] | None,
) -> nn.Module:
    set_seed(args.random_state)
    use_vocab = args.representation == "vocab"

    vocab: Dict[str, int] | None = None
    embedding_weight: torch.Tensor | None = None
    unk_idx: int | None = None
    if use_vocab:
        if vocab_info is None:
            raise ValueError("representation='vocab' requested but no vocab/embeddings were provided.")
        vocab, embedding_weight, unk_idx = vocab_info

    def make_dataset(text_slice: Sequence[str], label_slice: Sequence[int] | None):
        if use_vocab:
            assert vocab is not None and embedding_weight is not None and unk_idx is not None
            return VocabTweetDataset(
                text_slice,
                label_slice,
                ids=None,
                vocab=vocab,
                unk_idx=unk_idx,
                lowercase=args.lowercase,
                max_tokens=args.max_tokens,
            )
        return HashedTweetDataset(
            text_slice,
            label_slice,
            ids=None,
            num_features=args.num_features,
            ngram_max=args.ngram_max,
            seed=args.random_state,
            lowercase=args.lowercase,
            max_tokens=args.max_tokens,
        )

    def build_model() -> nn.Module:
        if use_vocab:
            assert embedding_weight is not None
            return EmbeddingClassifier(
                embedding_weight=embedding_weight,
                trainable=args.embedding_trainable,
                dropout=args.embedding_dropout,
                hidden_dim=args.hidden_dim,
            ).to(device)
        return HashedLogisticModel(args.num_features).to(device)

    def build_optimizer(model: nn.Module, lr_head: float, lr_embed: float | None) -> torch.optim.Optimizer:
        if use_vocab and args.embedding_trainable:
            embedding_params = [p for p in model.embedding.parameters() if p.requires_grad]
            head_params = [
                p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("embedding.")
            ]
            if not embedding_params:
                raise ValueError("No trainable embedding parameters found (did you set --embedding-trainable?).")
            if not head_params:
                raise ValueError("No trainable head parameters found.")
            embed_lr = lr_head if lr_embed is None else lr_embed
            param_groups = [
                {"params": embedding_params, "lr": embed_lr},
                {"params": head_params, "lr": lr_head},
            ]
            params = param_groups
        else:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            if not trainable_params:
                raise ValueError("No trainable parameters found.")
            params = trainable_params

        return AdEMAMix(
            params,
            lr=lr_head,
            betas=tuple(args.betas),
            alpha=args.alpha_mix,
            beta3_warmup=args.beta3_warmup or None,
            alpha_warmup=args.alpha_warmup or None,
            weight_decay=args.weight_decay,
        )

    def build_linear_scheduler(
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        warmup_steps = min(max(warmup_steps, 0), max(total_steps, 0))

        def lr_lambda(step: int) -> float:
            if total_steps <= 0:
                return 1.0
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 1.0 - progress)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def build_optimizer_and_scheduler(
        model: nn.Module,
        train_loader: DataLoader,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
        steps_per_epoch = max(len(train_loader), 1)
        total_steps = args.epochs * steps_per_epoch
        if args.use_linear_scheduler:
            if args.lr_warmup_steps is None:
                warmup_steps = min(2000, total_steps)
                if warmup_steps > 0:
                    print(
                        f"Using default lr warmup steps: {warmup_steps} "
                        f"(set --lr-warmup-steps to override)."
                    )
            else:
                warmup_steps = min(max(args.lr_warmup_steps, 0), total_steps)
        else:
            warmup_steps = 0

        lr_embed = args.lr_embed
        if lr_embed is not None and not (use_vocab and args.embedding_trainable):
            print("Warning: --lr-embed is set but embeddings are not trainable; ignoring.")
            lr_embed = None

        optimizer = build_optimizer(model, lr_head=args.lr, lr_embed=lr_embed)
        scheduler = (
            build_linear_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)
            if args.use_linear_scheduler
            else None
        )
        return optimizer, scheduler

    criterion = nn.BCEWithLogitsLoss()
    model = build_model()

    if args.val_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            texts,
            labels,
            test_size=args.val_size,
            stratify=labels,
            random_state=args.random_state,
        )
        train_ds = make_dataset(X_train, y_train)
        val_ds = make_dataset(X_val, y_val)
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

        optimizer, scheduler = build_optimizer_and_scheduler(model, train_loader)

        best_state = None
        best_metric = None
        best_epoch = 0
        epochs_no_improve = 0

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, scheduler=scheduler
            )
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
            metric_name = "val_acc" if args.early_metric == "acc" else "val_loss"
            print(f"Loaded best checkpoint from epoch {best_epoch} ({metric_name}={best_metric:.4f}).")
    else:
        train_ds = make_dataset(texts, labels)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_train,
            num_workers=0,
        )
        optimizer, scheduler = build_optimizer_and_scheduler(model, train_loader)

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, scheduler=scheduler
            )
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
    if args.max_tokens is not None and args.max_tokens <= 0:
        args.max_tokens = None
    device = choose_device(args.device)
    print(f"Using device: {device}")

    config = DatasetConfig(
        data_dir=args.data_dir,
        use_full_train=args.use_full,
        limit_per_class=args.limit_per_class,
    )

    vocab_info = None
    if args.representation == "vocab":
        vocab_info = load_vocab_and_embeddings(args.vocab_path, args.embeddings_path)

    texts, labels = load_labeled_data(config)
    model = fit_model(texts, labels, args, device, vocab_info)

    test_ids, test_texts = load_test_data(config.test_path)
    if args.representation == "vocab":
        assert vocab_info is not None
        vocab, _embedding_weight, unk_idx = vocab_info
        test_ds = VocabTweetDataset(
            test_texts,
            labels=None,
            ids=test_ids,
            vocab=vocab,
            unk_idx=unk_idx,
            lowercase=args.lowercase,
            max_tokens=args.max_tokens,
        )
    else:
        test_ds = HashedTweetDataset(
            test_texts,
            labels=None,
            ids=test_ids,
            num_features=args.num_features,
            ngram_max=args.ngram_max,
            seed=args.random_state,
            lowercase=args.lowercase,
            max_tokens=args.max_tokens,
        )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_infer,
        num_workers=0,
    )
    test_preds = predict_on_loader(model, test_loader, device)
    write_submission(test_ids, test_preds, args.output)


if __name__ == "__main__":
    main()
