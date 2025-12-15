from __future__ import annotations

import argparse
import random
import pickle
import html
import re
from collections import Counter
from pathlib import Path
import math
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.model_selection import train_test_split
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput

try:
    from peft import LoraConfig, TaskType, get_peft_model
    from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
except ImportError:  # Optional dependency; only needed when --use-lora is set.
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {}

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:
    Conv1D = None

try:
    import emoji
except ImportError:
    emoji = None

from ademamix import AdEMAMix, alpha_scheduler, beta3_scheduler

COMMON_LORA_CANDIDATES = [
    "q_lin",
    "k_lin",
    "v_lin",
    "out_lin",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "query",
    "key",
    "value",
    "dense",
    "c_attn",
    "in_proj",
    "Wqkv",
]

ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\u2063]")
REPEATED_ALPHA_RE = re.compile(r"([A-Za-z])\1{2,}")
REPEATED_PUNCT_RE = re.compile(r"([!?.,])\1{2,}")
HASHTAG_SPLIT_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+")
DEMOJI_RE = re.compile(r":([a-z0-9_+\-]+):")

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
    parser.add_argument(
        "--use-linear-scheduler",
        action="store_true",
        help="Use linear warmup then decay LR scheduler for AdEMAMix (disables cosine).",
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=0,
        help="Warmup steps for linear LR scheduler (ignored for cosine/no scheduler).",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=0,
        help="Run an interim validation every N optimizer steps (0 disables mid-epoch eval).",
    )
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
    parser.add_argument("--embedding-path", type=Path, default=None, help="Optional .npy word embedding matrix (from GloVe).")
    parser.add_argument("--embedding-vocab", type=Path, default=None, help="Vocabulary pickle aligned with the embedding matrix.")
    parser.add_argument(
        "--embedding-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to averaged embeddings (useful if magnitudes differ from transformer outputs).",
    )
    parser.add_argument(
        "--fusion-dropout",
        type=float,
        default=None,
        help="Dropout rate when mixing transformer features and averaged embeddings (defaults to model config).",
    )
    parser.add_argument(
        "--estimate-embedding-scale",
        action="store_true",
        help="Estimate a recommended embedding scale from a small sample and apply it before training.",
    )
    parser.add_argument(
        "--estimate-sample-size",
        type=int,
        default=2000,
        help="Number of training tweets to sample when estimating embedding scale.",
    )
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze transformer backbone parameters.")
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA adapters on the transformer backbone.")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha scaling.")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout.")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_lin,k_lin,v_lin,out_lin",
        help=(
            "Comma-separated list of module names to target with LoRA (e.g., query,key,value,dense/q_lin,k_lin,v_lin,out_lin); "
            "use 'auto' to pick sensible defaults for the chosen backbone."
        ),
    )
    parser.add_argument(
        "--min-clean-tokens",
        type=int,
        default=2,
        help="Drop tweets with fewer cleaned tokens than this (after preprocessing).",
    )
    parser.add_argument(
        "--max-clean-tokens",
        type=int,
        default=0,
        help="Truncate cleaned tokens to first/last N when above this length (0 disables).",
    )
    parser.add_argument(
        "--demojize-emojis",
        action="store_true",
        help="Convert emojis to textual tokens (requires emoji package) before tokenization.",
    )
    parser.add_argument(
        "--outlier-url-mention-ratio",
        type=float,
        default=0.65,
        help="Drop tweets when (@user+http) density exceeds this ratio (set <=0 to disable).",
    )
    parser.add_argument(
        "--outlier-allcaps-ratio",
        type=float,
        default=0.9,
        help="Drop short tweets with uppercase letter ratio above this threshold (set <=0 to disable).",
    )
    parser.add_argument(
        "--augment-prob",
        type=float,
        default=0.0,
        help="Probability to add a lightly augmented copy of each training tweet (0 disables).",
    )
    parser.add_argument(
        "--augment-max-per-class",
        type=int,
        default=0,
        help="Cap augmented samples per class; 0 means no cap when augmentation is enabled.",
    )
    parser.add_argument(
        "--imbalance-threshold",
        type=float,
        default=1.1,
        help="Apply class weights when the max/min class ratio exceeds this threshold after cleaning.",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def _targetable_module_names(model) -> List[str]:
    """Return names of modules that LoRA can wrap (linear/Conv1D)."""
    targetable = [nn.Linear]
    if Conv1D is not None:
        targetable.append(Conv1D)
    targetable_types = tuple(targetable)
    return [name for name, module in model.named_modules() if isinstance(module, targetable_types)]


def resolve_lora_targets(model, requested_targets: List[str] | None) -> List[str]:
    """
    Ensure target module names exist on the model; fall back to sensible defaults if not.
    This prevents PEFT from raising when users switch architectures without updating targets.
    """
    module_names = _targetable_module_names(model)

    def matches(target: str) -> bool:
        return any(
            name == target
            or name.endswith(f".{target}")
            or name.split(".")[-1] == target
            for name in module_names
        )

    normalized = [t.strip() for t in (requested_targets or []) if t and t.strip()]
    normalized = [t for t in normalized if t.lower() != "auto"]
    if normalized and any(matches(t) for t in normalized):
        return normalized

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    auto_candidates: List[str] = []
    if model_type:
        auto_candidates.extend(TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING.get(model_type, []))
    if model_type == "distilbert":
        auto_candidates.extend(["q_lin", "k_lin", "v_lin", "out_lin"])

    resolved = [t for t in auto_candidates if matches(t)]
    if resolved:
        if normalized:
            print(
                f"Requested LoRA targets {normalized} not found on model type {model_type}; "
                f"using detected targets {resolved}."
            )
        else:
            print(f"Using auto-detected LoRA targets for {model_type or 'model'}: {resolved}")
        return resolved

    for candidate in COMMON_LORA_CANDIDATES:
        if candidate not in auto_candidates:
            auto_candidates.append(candidate)

    resolved = [t for t in auto_candidates if matches(t)]
    if resolved:
        if normalized:
            print(
                f"Requested LoRA targets {normalized} not found on model type {model_type}; "
                f"using detected targets {resolved}."
            )
        else:
            print(f"Using auto-detected LoRA targets for {model_type or 'model'}: {resolved}")
        return resolved

    suffix_counts = Counter(name.split(".")[-1] for name in module_names)
    frequent = [name for name, count in suffix_counts.items() if count > 1]
    if frequent:
        fallback = frequent[:4]
        print(
            "No matching LoRA targets found; falling back to frequent module names: "
            f"{fallback}. Override with --lora-target-modules if needed."
        )
        return fallback

    raise ValueError(
        "Unable to determine LoRA target modules for this model. "
        f"Available modules include: {module_names[:10]}"
    )


def split_hashtag(token: str) -> List[str]:
    body = token[1:] if token.startswith("#") else token
    parts = HASHTAG_SPLIT_RE.findall(body)
    return parts if parts else [body]


def normalize_repetitions(token: str) -> str:
    token = REPEATED_ALPHA_RE.sub(r"\1\1", token)
    token = REPEATED_PUNCT_RE.sub(r"\1\1", token)
    return token


def demojize_text(text: str) -> str:
    if emoji is None:
        return text
    try:
        demojized = emoji.demojize(text, language="en")
    except TypeError:
        demojized = emoji.demojize(text)
    return DEMOJI_RE.sub(r" \1 ", demojized)


class TweetEmbeddingAverager:
    """Average pre-computed word embeddings (e.g., GloVe) for a tweet."""

    def __init__(self, embedding_matrix: np.ndarray, vocab: Dict[str, int], scale: float = 1.0) -> None:
        self.embedding_matrix = embedding_matrix.astype(np.float32)
        self.vocab = vocab
        self.scale = scale
        self.dim = self.embedding_matrix.shape[1]
        self.zero = np.zeros(self.dim, dtype=np.float32)

    def __call__(self, text: str) -> np.ndarray:
        indices = [
            idx
            for tok in text.split()
            if (idx := self.vocab.get(tok)) is not None and 0 <= idx < self.embedding_matrix.shape[0]
        ]
        if not indices:
            return self.zero.copy()
        vectors = self.embedding_matrix[indices]
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return (vectors.mean(axis=0) * self.scale).astype(np.float32)


def load_embedding_lookup(vocab_path: Path, embedding_path: Path, scale: float) -> TweetEmbeddingAverager:
    with vocab_path.open("rb") as handle:
        vocab = pickle.load(handle)
    embedding_matrix = np.load(embedding_path)
    if embedding_matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding matrix, got shape {embedding_matrix.shape}.")
    print(f"Loaded embeddings from {embedding_path} with shape {embedding_matrix.shape}")
    print(f"Loaded vocabulary with {len(vocab)} entries from {vocab_path}")
    return TweetEmbeddingAverager(embedding_matrix, vocab, scale=scale)


def estimate_embedding_scale(
    texts: List[str],
    tokenizer,
    transformer,
    embedding_lookup: TweetEmbeddingAverager,
    max_length: int,
    device: torch.device,
    sample_size: int = 2000,
    batch_size: int = 128,
) -> float:
    """Compute avg L2 norms for CLS vs averaged embeddings on a sample."""
    sample_texts = texts[: sample_size or len(texts)]
    if not sample_texts:
        raise ValueError("No texts available to estimate embedding scale.")

    emb_norms = [np.linalg.norm(embedding_lookup(t)) for t in sample_texts]
    avg_emb_norm = float(np.mean(emb_norms))

    cls_norms: List[float] = []
    transformer.eval()
    with torch.no_grad():
        for start in range(0, len(sample_texts), batch_size):
            chunk = sample_texts[start : start + batch_size]
            enc = tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            outputs = transformer(**enc)
            cls_vec = outputs.last_hidden_state[:, 0]
            cls_norms.extend(cls_vec.norm(dim=1).cpu().tolist())
    avg_cls_norm = float(np.mean(cls_norms))

    if avg_emb_norm == 0.0:
        raise ValueError("Average embedding norm is zero; cannot estimate scale.")

    suggested = avg_cls_norm / avg_emb_norm
    print(
        f"Estimated embedding scale: avg_cls_norm={avg_cls_norm:.4f}, "
        f"avg_emb_norm={avg_emb_norm:.4f}, suggested_scale={suggested:.4f}"
    )
    return suggested


def apply_lora_adapters(
    model,
    target_modules: List[str],
    r: int,
    alpha: float,
    dropout: float,
    task_type: str,
):
    if get_peft_model is None or LoraConfig is None or TaskType is None:
        raise ImportError("peft is required for --use-lora. Install with `pip install peft`.")
    task_enum = getattr(TaskType, task_type)
    resolved_targets = resolve_lora_targets(model, target_modules)
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=resolved_targets,
        bias="none",
        task_type=task_enum,
    )
    return get_peft_model(model, config)


def set_trainable_with_backbone(
    model,
    backbone_keywords: List[str],
    freeze_backbone: bool,
    use_lora: bool,
) -> None:
    """Freeze backbone params if requested; keep head and LoRA adapters trainable."""
    for name, param in model.named_parameters():
        in_backbone = any(key in name for key in backbone_keywords)
        if use_lora and "lora_" in name:
            param.requires_grad = True
        elif freeze_backbone and in_backbone:
            param.requires_grad = False
        else:
            param.requires_grad = True


def detect_backbone_keywords(model) -> List[str]:
    candidates = [
        "bert",
        "roberta",
        "distilbert",
        "camembert",
        "albert",
        "xlm_roberta",
        "electra",
        "deberta",
        "deberta_v2",
        "longformer",
        "xlnet",
        "mpnet",
        "transformer",
    ]
    found: List[str] = []
    for attr in candidates:
        if hasattr(model, attr):
            found.append(attr)
    if not found:
        found.append("transformer")
    return found


class HybridCollator:
    """Pad token inputs and stack auxiliary averaged embeddings when provided."""

    def __init__(self, tokenizer) -> None:
        self.base_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features):
        has_extra = "extra_features" in features[0]
        token_features = [{k: v for k, v in feat.items() if k != "extra_features"} for feat in features]
        batch = self.base_collator(token_features)
        if has_extra:
            extras = [torch.as_tensor(feat["extra_features"], dtype=torch.float32) for feat in features]
            batch["extra_features"] = torch.stack(extras, dim=0)
        return batch


class HybridTransformerClassifier(nn.Module):
    """Transformer encoder with optional averaged word embeddings fused into the classifier."""

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        extra_dim: int = 0,
        dropout: float | None = None,
        transformer_model=None,
    ) -> None:
        super().__init__()
        self.transformer = transformer_model if transformer_model is not None else AutoModel.from_pretrained(model_name)
        self.extra_dim = extra_dim

        hidden_size = self.transformer.config.hidden_size
        drop_rate = (
            dropout
            if dropout is not None
            else getattr(self.transformer.config, "seq_classif_dropout", getattr(self.transformer.config, "dropout", 0.1))
        )
        self.dropout = nn.Dropout(drop_rate)
        if extra_dim > 0:
            self.extra_proj = nn.Sequential(
                nn.LayerNorm(extra_dim),
                nn.Linear(extra_dim, hidden_size),
                nn.GELU(),
            )
            fused_dim = hidden_size * 2
        else:
            self.extra_proj = None
            fused_dim = hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        extra_features=None,
        labels=None,
    ) -> SequenceClassifierOutput:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]

        if self.extra_proj is not None and extra_features is not None:
            extra = self.extra_proj(extra_features.to(dtype=pooled.dtype))
            pooled = torch.cat([pooled, extra], dim=1)

        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return SequenceClassifierOutput(logits=logits, loss=loss)


def preprocess_tweet(
    text: str,
    *,
    min_tokens: int = 0,
    max_tokens: int = 0,
    demojize_emojis: bool = False,
    url_mention_ratio: float = 0.0,
    allcaps_ratio: float = 0.0,
) -> str | None:
    """
    Normalize tweet text with placeholder mapping, hashtag splitting, emoji text conversion,
    repetition collapse, HTML unescape, and light outlier filtering.
    Returns cleaned text or None if filtered out.
    """
    cleaned = html.unescape(text)
    cleaned = ZERO_WIDTH_RE.sub("", cleaned).replace("\ufeff", "").strip()
    cleaned = re.sub(r"^rt\s+", "", cleaned, flags=re.IGNORECASE)
    if demojize_emojis:
        cleaned = demojize_text(cleaned)

    tokens: List[str] = []
    for raw_tok in cleaned.split():
        parts = split_hashtag(raw_tok) if raw_tok.startswith("#") else [raw_tok]
        for part in parts:
            lowered = normalize_repetitions(part.lower().strip())
            lowered = lowered.replace("_", " ")
            for tok in lowered.split():
                if not tok:
                    continue
                if tok.startswith("@") and len(tok) > 1:
                    tokens.append("@user")
                elif tok.startswith("http"):
                    tokens.append("http")
                else:
                    tokens.append(tok)

    if not tokens:
        return None

    num_alpha = sum(ch.isalpha() for ch in text)
    upper_ratio = (sum(ch.isupper() for ch in text) / num_alpha) if num_alpha else 0.0
    if allcaps_ratio > 0 and upper_ratio >= allcaps_ratio and len(tokens) <= 4:
        return None

    url_mentions = sum(1 for t in tokens if t in {"http", "@user"})
    density = url_mentions / len(tokens)
    if url_mention_ratio > 0 and density >= url_mention_ratio and len(tokens) > 4:
        return None

    if min_tokens and len(tokens) < min_tokens:
        return None

    if max_tokens and len(tokens) > max_tokens:
        keep_first = max_tokens // 2
        keep_last = max_tokens - keep_first
        tokens = tokens[:keep_first] + tokens[-keep_last:]

    return " ".join(tokens)


def read_labeled(
    path: Path,
    label: int,
    limit: int | None,
    preprocess_fn: Callable[[str], str | None],
) -> Tuple[List[str], List[int], int]:
    texts: List[str] = []
    labels: List[int] = []
    dropped = 0
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            cleaned = line.strip()
            if not cleaned:
                continue
            processed = preprocess_fn(cleaned)
            if processed is None:
                dropped += 1
                continue
            texts.append(processed)
            labels.append(label)
    return texts, labels, dropped


def load_train_data(
    data_dir: Path,
    use_full: bool,
    limit_per_class: int | None,
    preprocess_fn: Callable[[str], str | None],
) -> Tuple[List[str], List[int], dict]:
    pos_file = data_dir / ("train_pos_full.txt" if use_full else "train_pos.txt")
    neg_file = data_dir / ("train_neg_full.txt" if use_full else "train_neg.txt")
    pos_texts, pos_labels, pos_dropped = read_labeled(pos_file, 1, limit_per_class, preprocess_fn)
    neg_texts, neg_labels, neg_dropped = read_labeled(neg_file, 0, limit_per_class, preprocess_fn)
    texts = pos_texts + neg_texts
    labels = pos_labels + neg_labels
    dropped = {"pos": pos_dropped, "neg": neg_dropped}
    print(
        f"Loaded {len(texts)} train tweets (pos: {len(pos_texts)}, neg: {len(neg_texts)}); "
        f"dropped after cleaning (pos: {pos_dropped}, neg: {neg_dropped})."
    )
    return texts, labels, dropped


def load_test_data(
    path: Path,
    preprocess_fn: Callable[[str], str | None],
) -> Tuple[List[int], List[str]]:
    ids: List[int] = []
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            cleaned = line.rstrip("\n")
            if not cleaned:
                continue
            id_str, tweet = cleaned.split(",", 1)
            ids.append(int(id_str))
            processed = preprocess_fn(tweet.strip())
            if processed is None:
                processed = ""
            texts.append(processed)
    print(f"Loaded {len(ids)} test tweets from {path}.")
    return ids, texts


def synonym_augment(text: str) -> str | None:
    """Lightweight synonym replacement on a few sentiment-bearing words."""
    synonym_bank = {
        "good": ["great", "nice", "positive", "pleasant"],
        "great": ["excellent", "wonderful"],
        "love": ["enjoy", "like", "adore"],
        "amazing": ["fantastic", "awesome"],
        "bad": ["terrible", "awful", "poor"],
        "sad": ["unhappy", "down"],
        "hate": ["dislike", "detest"],
        "angry": ["mad", "upset"],
        "happy": ["glad", "cheerful"],
        "disappointed": ["let down", "dissatisfied"],
    }
    tokens = text.split()
    new_tokens: List[str] = []
    changed = False
    for tok in tokens:
        base = tok.lower()
        if base in synonym_bank and random.random() < 0.3:
            replacement = random.choice(synonym_bank[base])
            new_tokens.append(replacement)
            changed = True
        else:
            new_tokens.append(tok)
    if not changed:
        return None
    return " ".join(new_tokens)


def augment_dataset(
    texts: List[str],
    labels: List[int],
    prob: float,
    max_per_class: int,
) -> Tuple[List[str], List[int], int]:
    if prob <= 0.0:
        return texts, labels, 0

    per_class_added = Counter()
    augmented = 0
    new_texts = list(texts)
    new_labels = list(labels)
    for text, label in zip(texts, labels):
        if random.random() >= prob:
            continue
        if max_per_class and per_class_added[label] >= max_per_class:
            continue
        augmented_text = synonym_augment(text)
        if augmented_text is None:
            continue
        new_texts.append(augmented_text)
        new_labels.append(label)
        per_class_added[label] += 1
        augmented += 1
    return new_texts, new_labels, augmented


def build_datasets(
    tokenizer: DistilBertTokenizerFast,
    texts: List[str],
    labels: List[int],
    val_size: float,
    max_length: int,
    seed: int,
    embedding_lookup: TweetEmbeddingAverager | None = None,
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
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        if embedding_lookup is not None:
            tokenized["extra_features"] = [embedding_lookup(text) for text in batch["text"]]
        return tokenized

    tokenized_train = train_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
    tokenized_val = val_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
    return tokenized_train, tokenized_val


def build_test_dataset(
    tokenizer: DistilBertTokenizerFast,
    texts: List[str],
    max_length: int,
    embedding_lookup: TweetEmbeddingAverager | None = None,
):
    ds = Dataset.from_dict({"text": texts})

    def tokenize_batch(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        if embedding_lookup is not None:
            tokenized["extra_features"] = [embedding_lookup(text) for text in batch["text"]]
        return tokenized

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


def evaluate(model, loader: DataLoader, device: torch.device, loss_fn=None):
    model.eval()
    loss_fn = loss_fn if loss_fn is not None else torch.nn.CrossEntropyLoss()
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

    if (args.embedding_path is None) != (args.embedding_vocab is None):
        raise ValueError("Both --embedding-path and --embedding-vocab are required to enable averaged embeddings.")

    embedding_lookup = (
        load_embedding_lookup(args.embedding_vocab, args.embedding_path, args.embedding_scale)
        if args.embedding_path is not None and args.embedding_vocab is not None
        else None
    )
    if embedding_lookup is not None:
        print(f"Enabling averaged word embeddings (dim={embedding_lookup.dim}) for fusion with transformer outputs.")

    preprocess_fn = lambda txt: preprocess_tweet(
        txt,
        min_tokens=args.min_clean_tokens,
        max_tokens=args.max_clean_tokens,
        demojize_emojis=args.demojize_emojis,
        url_mention_ratio=args.outlier_url_mention_ratio,
        allcaps_ratio=args.outlier_allcaps_ratio,
    )

    texts, labels, dropped = load_train_data(args.data_dir, args.use_full, args.limit_per_class, preprocess_fn)

    if args.augment_prob > 0.0:
        texts, labels, added = augment_dataset(texts, labels, args.augment_prob, args.augment_max_per_class)
        if added:
            print(f"Augmented training set with {added} synthetic samples (prob={args.augment_prob}).")

    label_counts = Counter(labels)
    if len(label_counts) < 2:
        raise ValueError("Training data must contain at least two classes after preprocessing.")
    imbalance_ratio = max(label_counts.values()) / max(1, min(label_counts.values()))
    class_weights = None
    if imbalance_ratio >= args.imbalance_threshold:
        total = sum(label_counts.values())
        num_classes = len(label_counts)
        weight_vals = [total / (num_classes * label_counts[i]) for i in sorted(label_counts)]
        class_weights = torch.tensor(weight_vals, dtype=torch.float32)
        print(f"Applying class weights due to imbalance ratio {imbalance_ratio:.3f}: {weight_vals}")
    else:
        print(f"Class distribution after cleaning/augment: {dict(label_counts)} (imbalance ratio {imbalance_ratio:.3f})")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    lora_targets = [t.strip() for t in args.lora_target_modules.split(",") if t.strip()]
    base_transformer = None
    if embedding_lookup is not None:
        base_transformer = AutoModel.from_pretrained(args.model_name)
        if args.use_lora:
            base_transformer = apply_lora_adapters(
                base_transformer,
                lora_targets,
                r=args.lora_r,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                task_type="FEATURE_EXTRACTION",
            )
        base_transformer.to(device)

    if args.estimate_embedding_scale:
        if embedding_lookup is None:
            raise ValueError("--estimate-embedding-scale requires --embedding-path and --embedding-vocab.")
        print(
            f"Estimating embedding scale using {min(len(texts), args.estimate_sample_size)} samples "
            f"(max_length={args.max_length})..."
        )
        suggested = estimate_embedding_scale(
            texts,
            tokenizer,
            base_transformer,
            embedding_lookup,
            args.max_length,
            device,
            sample_size=args.estimate_sample_size,
        )
        args.embedding_scale = suggested
        embedding_lookup.scale = suggested
        print(f"Applied suggested embedding scale: {suggested:.4f}")

    if embedding_lookup is not None:
        model = HybridTransformerClassifier(
            args.model_name,
            num_labels=2,
            extra_dim=embedding_lookup.dim,
            dropout=args.fusion_dropout,
            transformer_model=base_transformer,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2,
        )
        if args.use_lora:
            model = apply_lora_adapters(
                model,
                lora_targets,
                r=args.lora_r,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                task_type="SEQ_CLS",
            )
    model.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)

    backbone_keys = ["transformer"] if isinstance(model, HybridTransformerClassifier) else detect_backbone_keywords(model)
    set_trainable_with_backbone(
        model,
        backbone_keys,
        freeze_backbone=args.freeze_encoder or args.use_lora,
        use_lora=args.use_lora,
    )

    train_ds, val_ds = build_datasets(
        tokenizer,
        texts,
        labels,
        args.val_size,
        args.max_length,
        args.seed,
        embedding_lookup=embedding_lookup,
    )

    test_ids, test_texts = load_test_data(args.data_dir / "test_data.txt", preprocess_fn)
    test_ds = build_test_dataset(tokenizer, test_texts, args.max_length, embedding_lookup=embedding_lookup)

    # Prepare torch datasets/dataloaders
    extra_columns = ["extra_features"] if embedding_lookup is not None else []
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"] + extra_columns)
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"] + extra_columns)
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask"] + extra_columns)

    data_collator = HybridCollator(tokenizer) if embedding_lookup is not None else DataCollatorWithPadding(tokenizer=tokenizer)
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
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
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
        if args.use_cosine_scheduler and not args.use_linear_scheduler
        else None
    )
    if args.use_linear_scheduler and args.use_cosine_scheduler:
        raise ValueError("Choose either cosine or linear LR scheduler, not both.")
    if args.use_linear_scheduler:
        warmup_steps = max(args.lr_warmup_steps, 0)

        def lr_lambda(step: int):
            if total_steps <= 0:
                return 1.0
            if step < warmup_steps and warmup_steps > 0:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 1.0 - progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

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

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
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

            if args.eval_steps and global_step > 0 and global_step % args.eval_steps == 0:
                eval_model = swa_model if use_swa and global_step >= swa_start_step else model
                val_loss, val_acc = evaluate(eval_model, val_loader, device, loss_fn=loss_fn)
                print(f"[epoch {epoch} step {global_step}] interim val loss {val_loss:.4f} acc {val_acc:.4f}")
                model.train()

        # Choose model to evaluate (SWA if active)
        eval_model = swa_model if use_swa and global_step >= swa_start_step else model
        val_loss, val_acc = evaluate(eval_model, val_loader, device, loss_fn=loss_fn)
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
