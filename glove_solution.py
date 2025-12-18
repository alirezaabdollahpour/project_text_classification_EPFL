#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def parse_args() -> argparse.Namespace:
    class _ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description="Train GloVe embeddings and evaluate linear models with multiple feature sets.",
        formatter_class=_ArgFormatter,
        epilog=(
            "Examples:\n"
            "  # Small run with full diagnostics\n"
            "  python glove_solution.py --limit-per-class 5000 --val-size 0.2 --diagnostics \\\n"
            "    --tune-thresholds --svm-loss hinge --svm-c 0.1 --svm-max-iter 20000 --feature-sets all\n"
            "\n"
            "  # GloVe only (fast)\n"
            "  python glove_solution.py --limit-per-class 50000 --val-size 0.2 --feature-sets glove\n"
        ),
    )
    parser.add_argument(
        "--cooc-path",
        type=Path,
        default=Path("cooc.pkl"),
        help="Path to co-occurrence matrix pickle (COO sparse matrix).",
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=Path("vocab.pkl"),
        help="Path to vocab.pkl mapping tokens to row indices.",
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=Path("embeddings.npy"),
        help="Where to load/save the embedding matrix (NumPy .npy).",
    )
    parser.add_argument(
        "--train-embeddings",
        action="store_true",
        help="Force training embeddings even if embeddings.npy already exists.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=20,
        help="Embedding dimension (only used when training embeddings).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of SGD epochs for GloVe training (only used when training embeddings).",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.001,
        help="Learning rate for GloVe SGD (only used when training embeddings).",
    )
    parser.add_argument(
        "--nmax",
        type=int,
        default=100,
        help="GloVe weighting cutoff nmax (only used when training embeddings).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.75,
        help="GloVe weighting exponent alpha (only used when training embeddings).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/twitter-datasets"),
        help="Directory containing train_pos.txt/train_neg.txt (or *_full.txt).",
    )
    parser.add_argument(
        "--use-full",
        action="store_true",
        help="Use train_pos_full.txt/train_neg_full.txt instead of train_pos.txt/train_neg.txt.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Optional cap on the number of positive/negative tweets (for quicker experiments).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction of the dataset to hold out for validation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=13,
        help="Random seed for train/validation split.",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase tweets before token lookup (off by default; dataset is already tokenized).",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print extra sanity-check diagnostics for features and model predictions.",
    )
    parser.add_argument(
        "--everything",
        action="store_true",
        help="Shortcut for: --diagnostics --tune-thresholds --feature-sets all",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["mean", "mean_max"],
        default="mean",
        help="Tweet embedding pooling: mean, or mean+max (concatenation).",
    )
    parser.add_argument(
        "--tune-thresholds",
        action="store_true",
        help=(
            "Tune decision thresholds on the training split to maximize training accuracy "
            "(applies to linear regression outputs and LinearSVC decision_function)."
        ),
    )
    parser.add_argument(
        "--lr-threshold",
        type=float,
        default=0.5,
        help="Decision threshold for linear regression outputs (used unless --tune-thresholds).",
    )
    parser.add_argument(
        "--svm-threshold",
        type=float,
        default=0.0,
        help="Decision threshold for LinearSVC decision_function (used unless --tune-thresholds).",
    )
    parser.add_argument(
        "--svm-c",
        type=float,
        default=1.0,
        help="LinearSVC regularization strength C.",
    )
    parser.add_argument(
        "--svm-loss",
        type=str,
        choices=["squared_hinge", "hinge"],
        default="squared_hinge",
        help="LinearSVC loss (hinge often works better but may need more iterations).",
    )
    parser.add_argument(
        "--svm-max-iter",
        type=int,
        default=10_000,
        help="Max iterations for LinearSVC.",
    )
    parser.add_argument(
        "--svm-tol",
        type=float,
        default=1e-4,
        help="Tolerance for LinearSVC.",
    )
    parser.add_argument(
        "--no-word-ngrams",
        action="store_true",
        help="Disable word n-gram bag-of-words features.",
    )
    parser.add_argument(
        "--no-char-ngrams",
        action="store_true",
        help="Disable character n-gram features.",
    )
    parser.add_argument(
        "--word-ngram-range",
        type=int,
        nargs=2,
        default=(1, 2),
        metavar=("MIN_N", "MAX_N"),
        help="Word n-gram range for bag-of-words features.",
    )
    parser.add_argument(
        "--char-ngram-range",
        type=int,
        nargs=2,
        default=(3, 5),
        metavar=("MIN_N", "MAX_N"),
        help="Character n-gram range for char features.",
    )
    parser.add_argument(
        "--word-hash-features",
        type=int,
        default=2**18,
        help="Number of hashing features for word n-grams.",
    )
    parser.add_argument(
        "--char-hash-features",
        type=int,
        default=2**17,
        help="Number of hashing features for char n-grams.",
    )
    parser.add_argument(
        "--no-sublinear-tf",
        action="store_true",
        help="Disable sublinear TF scaling in TF-IDF (uses raw term frequency).",
    )
    parser.add_argument(
        "--sgd-alpha",
        type=float,
        default=1e-4,
        help="SGDRegressor L2 regularization strength (only used for sparse features).",
    )
    parser.add_argument(
        "--sgd-max-iter",
        type=int,
        default=1000,
        help="SGDRegressor max iterations (only used for sparse features).",
    )
    parser.add_argument(
        "--sgd-tol",
        type=float,
        default=1e-3,
        help="SGDRegressor tolerance (only used for sparse features).",
    )
    parser.add_argument(
        "--feature-sets",
        type=str,
        nargs="+",
        default=None,
        choices=[
            "glove",
            "bow",
            "char",
            "bow+char",
            "glove+bow",
            "glove+char",
            "glove+bow+char",
            "all",
        ],
        help=(
            "Which feature sets to evaluate.\n"
            "If omitted, runs: glove + the strongest available combo based on enabled n-grams.\n"
            "Use 'all' to evaluate all valid combinations."
        ),
    )
    return parser.parse_args()


@dataclass(frozen=True)
class FeatureEncodingStats:
    total_tweets: int
    total_tokens: int
    in_vocab_tokens: int
    tweets_with_no_in_vocab_tokens: int


@dataclass(frozen=True)
class EvalResult:
    feature_set: str
    model: str
    train_time_s: float
    train_acc: float
    val_acc: float
    threshold: float
    details: dict[str, Any]


def train_glove_embeddings(
    cooc_path: Path,
    embeddings_path: Path,
    embedding_dim: int,
    epochs: int,
    eta: float,
    nmax: int,
    alpha: float,
) -> np.ndarray:
    print(f"Loading co-occurrence matrix from {cooc_path}")
    with open(cooc_path, "rb") as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    start = time.perf_counter()
    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x
    elapsed = time.perf_counter() - start

    np.save(embeddings_path, xs)
    print(f"Saved embeddings to {embeddings_path} (shape={xs.shape}, time={elapsed:.1f}s)")
    return xs


def load_vocab(vocab_path: Path) -> dict[str, int]:
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    if not isinstance(vocab, dict):
        raise TypeError(f"Expected vocab.pkl to contain a dict, got {type(vocab)}.")
    return vocab


def read_lines(path: Path, limit: int | None = None) -> list[str]:
    lines: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.rstrip("\n"))
            if limit is not None and len(lines) >= limit:
                break
    return lines


def build_average_embedding_features(
    tweets: list[str],
    vocab: dict[str, int],
    embeddings: np.ndarray,
    *,
    lowercase: bool,
    pooling: str = "mean",
    print_every: int = 20_000,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, FeatureEncodingStats]:
    dim = int(embeddings.shape[1])
    out_dim = dim if pooling == "mean" else dim * 2
    features = np.zeros((len(tweets), out_dim), dtype=np.float32)
    emb = embeddings.astype(np.float32, copy=False)

    total_tokens = 0
    in_vocab_tokens = 0
    tweets_with_no_in_vocab_tokens = 0

    for i, tweet in enumerate(tweets):
        tokens = tweet.split()
        total_tokens += len(tokens)
        if lowercase:
            tokens = [t.lower() for t in tokens]
        indices: list[int] = []
        for token in tokens:
            idx = vocab.get(token)
            if idx is not None:
                indices.append(idx)
        in_vocab_tokens += len(indices)
        if indices:
            vecs = emb[indices]
            if pooling == "mean":
                features[i] = vecs.mean(axis=0)
            elif pooling == "mean_max":
                features[i, :dim] = vecs.mean(axis=0)
                features[i, dim:] = vecs.max(axis=0)
            else:
                raise ValueError(f"Unknown pooling mode: {pooling}")
        else:
            tweets_with_no_in_vocab_tokens += 1
        if print_every and (i + 1) % print_every == 0:
            print(f"  Encoded {i + 1}/{len(tweets)} tweets")

    if not return_stats:
        return features
    stats = FeatureEncodingStats(
        total_tweets=len(tweets),
        total_tokens=total_tokens,
        in_vocab_tokens=in_vocab_tokens,
        tweets_with_no_in_vocab_tokens=tweets_with_no_in_vocab_tokens,
    )
    return features, stats


def _format_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, str]:
    total = int(y_true.shape[0])
    correct = int(np.count_nonzero(y_true == y_pred))
    acc = correct / total if total else float("nan")
    return acc, f"{acc:.6f} ({correct}/{total})"


def _best_threshold_from_scores(scores: np.ndarray, y_true_binary: np.ndarray) -> tuple[float, float]:
    scores = np.asarray(scores).reshape(-1)
    y_true_binary = np.asarray(y_true_binary).reshape(-1).astype(np.int8, copy=False)
    if scores.shape != y_true_binary.shape:
        raise ValueError(f"scores shape {scores.shape} != y_true shape {y_true_binary.shape}")
    if scores.size == 0:
        raise ValueError("Cannot tune threshold on empty arrays.")

    order = np.argsort(scores, kind="mergesort")
    scores_sorted = scores[order]
    y_sorted = y_true_binary[order]

    cum_pos = np.cumsum(y_sorted, dtype=np.int64)
    total_pos = int(cum_pos[-1])
    n = int(scores_sorted.shape[0])
    pos_below = np.concatenate(([0], cum_pos))
    k = np.arange(n + 1, dtype=np.int64)
    acc = (k + total_pos - 2 * pos_below) / n

    uniq_start = np.concatenate(([0], np.flatnonzero(scores_sorted[1:] != scores_sorted[:-1]) + 1))
    candidate_k = np.concatenate((uniq_start, [n]))
    best_k = int(candidate_k[np.argmax(acc[candidate_k])])
    best_acc = float(acc[best_k])

    if best_k >= n:
        threshold = float(np.nextafter(scores_sorted[-1], np.inf))
    else:
        threshold = float(scores_sorted[best_k])
    return threshold, best_acc


def _matrix_info(X: Any) -> str:
    if sparse.issparse(X):
        size = int(X.shape[0]) * int(X.shape[1])
        density = (float(X.nnz) / float(size)) if size else 0.0
        return (
            f"shape={X.shape}, nnz={X.nnz}, density={density:.6e}, "
            f"dtype={X.dtype}, format={X.getformat()}"
        )
    return f"shape={X.shape}, dtype={X.dtype}"


def build_hashed_tfidf_features(
    train_texts: list[str],
    val_texts: list[str],
    *,
    analyzer: str,
    ngram_range: tuple[int, int],
    n_features: int,
    lowercase: bool,
    sublinear_tf: bool,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    if analyzer == "word":
        vectorizer = HashingVectorizer(
            analyzer="word",
            tokenizer=str.split,
            token_pattern=None,
            ngram_range=ngram_range,
            n_features=n_features,
            alternate_sign=False,
            norm=None,
            lowercase=lowercase,
            dtype=np.float32,
        )
    elif analyzer == "char":
        vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            n_features=n_features,
            alternate_sign=False,
            norm=None,
            lowercase=lowercase,
            dtype=np.float32,
        )
    else:
        raise ValueError(f"Unsupported analyzer: {analyzer}")

    X_train_counts = vectorizer.transform(train_texts)
    X_val_counts = vectorizer.transform(val_texts)

    tfidf = TfidfTransformer(
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=sublinear_tf,
    )
    X_train = tfidf.fit_transform(X_train_counts)
    X_val = tfidf.transform(X_val_counts)
    return X_train.tocsr(), X_val.tocsr()


def _train_and_eval_regression(
    feature_set: str,
    X_train: Any,
    y_train: np.ndarray,
    X_val: Any,
    y_val: np.ndarray,
    *,
    args: argparse.Namespace,
) -> tuple[EvalResult, np.ndarray]:
    use_sparse = sparse.issparse(X_train)
    if use_sparse:
        model = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=args.sgd_alpha,
            max_iter=args.sgd_max_iter,
            tol=args.sgd_tol,
            random_state=args.random_state,
        )
        model_name = "Linear Regression (SGD)"
    else:
        model = LinearRegression(n_jobs=-1)
        model_name = "Linear Regression"

    print(f"\nTraining {model_name} on [{feature_set}] (threshold @ {args.lr_threshold})")
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    y_train_scores = model.predict(X_train)
    y_val_scores = model.predict(X_val)

    threshold = float(args.lr_threshold)
    details: dict[str, Any] = {"estimator": type(model).__name__}
    if args.tune_thresholds:
        threshold, best_train_acc = _best_threshold_from_scores(y_train_scores, y_train)
        details["best_train_acc_at_threshold"] = best_train_acc
        print(f"  Tuned threshold={threshold:.6f} (best train acc={best_train_acc:.6f})")

    y_train_pred = (y_train_scores >= threshold).astype(np.int64)
    y_val_pred = (y_val_scores >= threshold).astype(np.int64)
    train_acc, train_acc_str = _format_accuracy(y_train, y_train_pred)
    val_acc, val_acc_str = _format_accuracy(y_val, y_val_pred)
    print(f"{model_name} done in {train_time:.1f}s | train acc={train_acc_str} | val acc={val_acc_str}")

    if use_sparse:
        details.update({"alpha": args.sgd_alpha, "max_iter": args.sgd_max_iter, "tol": args.sgd_tol})

    return (
        EvalResult(
            feature_set=feature_set,
            model=model_name,
            train_time_s=train_time,
            train_acc=train_acc,
            val_acc=val_acc,
            threshold=threshold,
            details=details,
        ),
        y_val_pred,
    )


def _train_and_eval_svm(
    feature_set: str,
    X_train: Any,
    y_train: np.ndarray,
    X_val: Any,
    y_val: np.ndarray,
    *,
    args: argparse.Namespace,
) -> tuple[EvalResult, np.ndarray]:
    model = LinearSVC(
        dual="auto",
        max_iter=args.svm_max_iter,
        tol=args.svm_tol,
        C=args.svm_c,
        loss=args.svm_loss,
    )
    model_name = "Linear SVM (LinearSVC)"

    print(f"\nTraining {model_name} on [{feature_set}] (threshold @ {args.svm_threshold})")
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    y_train_scores = model.decision_function(X_train)
    y_val_scores = model.decision_function(X_val)

    neg_label = int(model.classes_[0])
    pos_label = int(model.classes_[1])

    threshold = float(args.svm_threshold)
    details: dict[str, Any] = {
        "C": args.svm_c,
        "loss": args.svm_loss,
        "max_iter": args.svm_max_iter,
        "tol": args.svm_tol,
    }
    n_iter = getattr(model, "n_iter_", None)
    if n_iter is not None:
        details["n_iter_"] = n_iter
        n_iter_max = int(np.max(n_iter)) if np.size(n_iter) else 0
        if n_iter_max >= int(args.svm_max_iter):
            print(
                f"  Note: LinearSVC reached max_iter={args.svm_max_iter} "
                "(consider increasing --svm-max-iter or loosening --svm-tol)."
            )

    if args.tune_thresholds:
        threshold, best_train_acc = _best_threshold_from_scores(
            y_train_scores, (y_train == pos_label).astype(np.int8)
        )
        details["best_train_acc_at_threshold"] = best_train_acc
        print(f"  Tuned threshold={threshold:.6f} (best train acc={best_train_acc:.6f})")

    y_train_pred = np.where(y_train_scores >= threshold, pos_label, neg_label).astype(np.int64)
    y_val_pred = np.where(y_val_scores >= threshold, pos_label, neg_label).astype(np.int64)
    train_acc, train_acc_str = _format_accuracy(y_train, y_train_pred)
    val_acc, val_acc_str = _format_accuracy(y_val, y_val_pred)
    print(f"{model_name} done in {train_time:.1f}s | train acc={train_acc_str} | val acc={val_acc_str}")

    return (
        EvalResult(
            feature_set=feature_set,
            model=model_name,
            train_time_s=train_time,
            train_acc=train_acc,
            val_acc=val_acc,
            threshold=threshold,
            details=details,
        ),
        y_val_pred,
    )


def _evaluate_feature_set(
    feature_set: str,
    X_train: Any,
    y_train: np.ndarray,
    X_val: Any,
    y_val: np.ndarray,
    *,
    args: argparse.Namespace,
) -> list[EvalResult]:
    print(f"\n=== Feature set: {feature_set} ===")
    print(f"X_train: {_matrix_info(X_train)}")
    print(f"X_val:   {_matrix_info(X_val)}")

    lr_result, lr_val_pred = _train_and_eval_regression(feature_set, X_train, y_train, X_val, y_val, args=args)
    svm_result, svm_val_pred = _train_and_eval_svm(feature_set, X_train, y_train, X_val, y_val, args=args)

    feature_meta: dict[str, Any] = {"n_features": int(X_train.shape[1])}
    if sparse.issparse(X_train):
        feature_meta.update(
            {
                "train_nnz": int(X_train.nnz),
                "val_nnz": int(X_val.nnz),
            }
        )
    lr_result.details.update(feature_meta)
    svm_result.details.update(feature_meta)

    if args.diagnostics:
        lr_counts = np.bincount(lr_val_pred, minlength=2)
        svm_counts = np.bincount(svm_val_pred, minlength=2)
        disagree = int(np.count_nonzero(lr_val_pred != svm_val_pred))
        agree = 1.0 - (disagree / int(y_val.shape[0]))
        print(
            "\nDiagnostics: val prediction balance "
            f"(LR 0/1={int(lr_counts[0])}/{int(lr_counts[1])}, "
            f"SVM 0/1={int(svm_counts[0])}/{int(svm_counts[1])})"
        )
        print(f"Diagnostics: LR vs SVM val agreement={agree:.6f} (disagreements={disagree})")

    return [lr_result, svm_result]


def main() -> None:
    args = parse_args()
    if args.everything:
        args.diagnostics = True
        args.tune_thresholds = True
        args.feature_sets = ["all"]

    if args.train_embeddings or not args.embeddings_path.exists():
        embeddings = train_glove_embeddings(
            args.cooc_path,
            args.embeddings_path,
            embedding_dim=args.embedding_dim,
            epochs=args.epochs,
            eta=args.eta,
            nmax=args.nmax,
            alpha=args.alpha,
        )
    else:
        embeddings = np.load(args.embeddings_path)
        print(f"Loaded embeddings from {args.embeddings_path} (shape={embeddings.shape})")

    vocab = load_vocab(args.vocab_path)
    if embeddings.shape[0] < len(vocab):
        raise ValueError(
            f"Embeddings rows ({embeddings.shape[0]}) smaller than vocab size ({len(vocab)})."
        )

    pos_name = "train_pos_full.txt" if args.use_full else "train_pos.txt"
    neg_name = "train_neg_full.txt" if args.use_full else "train_neg.txt"
    pos_path = args.data_dir / pos_name
    neg_path = args.data_dir / neg_name

    print(f"Loading tweets from {pos_path} and {neg_path}")
    pos_tweets = read_lines(pos_path, limit=args.limit_per_class)
    neg_tweets = read_lines(neg_path, limit=args.limit_per_class)
    print(f"Loaded {len(pos_tweets)} positive and {len(neg_tweets)} negative tweets")

    tweets = pos_tweets + neg_tweets
    labels = np.array([1] * len(pos_tweets) + [0] * len(neg_tweets), dtype=np.int64)

    pos = int(labels.sum())
    neg = int(labels.shape[0] - pos)
    print(
        f"Dataset: {labels.shape[0]} examples | "
        f"pos={pos} ({pos/labels.shape[0]:.4f}) | "
        f"neg={neg} ({neg/labels.shape[0]:.4f})"
    )

    tweets_train, tweets_val, y_train, y_val = train_test_split(
        tweets,
        labels,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=labels,
    )
    del tweets
    del labels
    print(f"Train split: {len(tweets_train)} examples | Val split: {len(tweets_val)} examples")

    use_word = not args.no_word_ngrams
    use_char = not args.no_char_ngrams

    all_feature_sets = [
        "glove",
        "bow",
        "char",
        "bow+char",
        "glove+bow",
        "glove+char",
        "glove+bow+char",
    ]
    default_feature_sets = ["glove"]
    if use_word and use_char:
        default_feature_sets.append("glove+bow+char")
    elif use_word:
        default_feature_sets.append("glove+bow")
    elif use_char:
        default_feature_sets.append("glove+char")

    requested_feature_sets = args.feature_sets or default_feature_sets
    if "all" in requested_feature_sets:
        requested_feature_sets = [
            fs
            for fs in all_feature_sets
            if ("bow" not in fs or use_word) and ("char" not in fs or use_char)
        ]
    else:
        requested_feature_sets = [fs for fs in requested_feature_sets if fs != "all"]

    feature_sets: list[str] = []
    seen: set[str] = set()
    for fs in requested_feature_sets:
        if fs not in seen:
            feature_sets.append(fs)
            seen.add(fs)

    invalid: list[str] = []
    for fs in feature_sets:
        if "bow" in fs and not use_word:
            invalid.append(f"{fs} (word n-grams disabled via --no-word-ngrams)")
        if "char" in fs and not use_char:
            invalid.append(f"{fs} (char n-grams disabled via --no-char-ngrams)")
    if invalid:
        raise SystemExit("Invalid --feature-sets requested: " + ", ".join(invalid))

    print("\nWill evaluate feature sets:")
    for fs in feature_sets:
        print(f"  - {fs}")

    glove_needed = any("glove" in fs for fs in feature_sets)
    bow_needed = any("bow" in fs for fs in feature_sets)
    char_needed = any("char" in fs for fs in feature_sets)
    glove_sparse_needed = any(("glove" in fs) and (fs != "glove") for fs in feature_sets)

    X_train_glove: np.ndarray | None = None
    X_val_glove: np.ndarray | None = None
    X_train_glove_csr: sparse.csr_matrix | None = None
    X_val_glove_csr: sparse.csr_matrix | None = None

    if glove_needed:
        print("\nBuilding GloVe pooled features")
        train_stats: FeatureEncodingStats | None = None
        val_stats: FeatureEncodingStats | None = None
        if args.diagnostics:
            X_train_glove, train_stats = build_average_embedding_features(
                tweets_train,
                vocab,
                embeddings,
                lowercase=args.lowercase,
                pooling=args.pooling,
                print_every=20_000,
                return_stats=True,
            )
            X_val_glove, val_stats = build_average_embedding_features(
                tweets_val,
                vocab,
                embeddings,
                lowercase=args.lowercase,
                pooling=args.pooling,
                print_every=20_000,
                return_stats=True,
            )
        else:
            X_train_glove = build_average_embedding_features(
                tweets_train,
                vocab,
                embeddings,
                lowercase=args.lowercase,
                pooling=args.pooling,
                print_every=20_000,
            )
            X_val_glove = build_average_embedding_features(
                tweets_val,
                vocab,
                embeddings,
                lowercase=args.lowercase,
                pooling=args.pooling,
                print_every=20_000,
            )

        if args.diagnostics and train_stats is not None and val_stats is not None:
            for split_name, stats in [("train", train_stats), ("val", val_stats)]:
                coverage = (stats.in_vocab_tokens / stats.total_tokens) if stats.total_tokens else 0.0
                avg_tokens = stats.total_tokens / stats.total_tweets if stats.total_tweets else 0.0
                avg_in_vocab = stats.in_vocab_tokens / stats.total_tweets if stats.total_tweets else 0.0
                print(
                    f"GloVe encoding stats ({split_name}): "
                    f"avg_tokens/tweet={avg_tokens:.2f}, "
                    f"avg_in_vocab_tokens/tweet={avg_in_vocab:.2f}, "
                    f"token_coverage={coverage:.4f}, "
                    f"zero_tweets={stats.tweets_with_no_in_vocab_tokens}"
                )

        assert X_train_glove is not None and X_val_glove is not None
        print(
            f"GloVe features (before scaling): train {_matrix_info(X_train_glove)} | val {_matrix_info(X_val_glove)}"
        )
        scaler = StandardScaler(copy=False)
        X_train_glove = scaler.fit_transform(X_train_glove)
        X_val_glove = scaler.transform(X_val_glove)

        if glove_sparse_needed:
            print("Converting scaled GloVe features to CSR for stacking")
            start = time.perf_counter()
            X_train_glove_csr = sparse.csr_matrix(X_train_glove)
            X_val_glove_csr = sparse.csr_matrix(X_val_glove)
            elapsed = time.perf_counter() - start
            print(f"GloVe CSR conversion done in {elapsed:.1f}s")

    X_train_word: sparse.csr_matrix | None = None
    X_val_word: sparse.csr_matrix | None = None
    if bow_needed:
        word_range = (int(args.word_ngram_range[0]), int(args.word_ngram_range[1]))
        sublinear_tf = not args.no_sublinear_tf
        print(
            "\nBuilding word n-gram TF-IDF features (hashing) "
            f"(ngram_range={word_range}, n_features={args.word_hash_features})"
        )
        start = time.perf_counter()
        X_train_word, X_val_word = build_hashed_tfidf_features(
            tweets_train,
            tweets_val,
            analyzer="word",
            ngram_range=word_range,
            n_features=int(args.word_hash_features),
            lowercase=args.lowercase,
            sublinear_tf=sublinear_tf,
        )
        elapsed = time.perf_counter() - start
        print(
            f"Word TF-IDF built in {elapsed:.1f}s | "
            f"train {_matrix_info(X_train_word)} | val {_matrix_info(X_val_word)}"
        )

    X_train_char: sparse.csr_matrix | None = None
    X_val_char: sparse.csr_matrix | None = None
    if char_needed:
        char_range = (int(args.char_ngram_range[0]), int(args.char_ngram_range[1]))
        sublinear_tf = not args.no_sublinear_tf
        print(
            "\nBuilding char n-gram TF-IDF features (hashing) "
            f"(ngram_range={char_range}, n_features={args.char_hash_features})"
        )
        start = time.perf_counter()
        X_train_char, X_val_char = build_hashed_tfidf_features(
            tweets_train,
            tweets_val,
            analyzer="char",
            ngram_range=char_range,
            n_features=int(args.char_hash_features),
            lowercase=args.lowercase,
            sublinear_tf=sublinear_tf,
        )
        elapsed = time.perf_counter() - start
        print(
            f"Char TF-IDF built in {elapsed:.1f}s | "
            f"train {_matrix_info(X_train_char)} | val {_matrix_info(X_val_char)}"
        )

    del tweets_train
    del tweets_val

    results: list[EvalResult] = []

    for feature_set in feature_sets:
        X_train_fs: Any
        X_val_fs: Any
        built_here = False

        if feature_set == "glove":
            if X_train_glove is None or X_val_glove is None:
                raise SystemExit("Feature set 'glove' requested but GloVe features were not built.")
            X_train_fs = X_train_glove
            X_val_fs = X_val_glove
        elif feature_set == "bow":
            if X_train_word is None or X_val_word is None:
                raise SystemExit("Feature set 'bow' requested but word n-gram features were not built.")
            X_train_fs = X_train_word
            X_val_fs = X_val_word
        elif feature_set == "char":
            if X_train_char is None or X_val_char is None:
                raise SystemExit("Feature set 'char' requested but char n-gram features were not built.")
            X_train_fs = X_train_char
            X_val_fs = X_val_char
        elif feature_set == "bow+char":
            if X_train_word is None or X_val_word is None or X_train_char is None or X_val_char is None:
                raise SystemExit("Feature set 'bow+char' requested but required features were not built.")
            start = time.perf_counter()
            X_train_fs = sparse.hstack([X_train_word, X_train_char], format="csr")
            X_val_fs = sparse.hstack([X_val_word, X_val_char], format="csr")
            elapsed = time.perf_counter() - start
            print(f"\nBuilt [{feature_set}] combined matrix in {elapsed:.1f}s")
            built_here = True
        elif feature_set == "glove+bow":
            if X_train_glove_csr is None or X_val_glove_csr is None or X_train_word is None or X_val_word is None:
                raise SystemExit("Feature set 'glove+bow' requested but required features were not built.")
            start = time.perf_counter()
            X_train_fs = sparse.hstack([X_train_glove_csr, X_train_word], format="csr")
            X_val_fs = sparse.hstack([X_val_glove_csr, X_val_word], format="csr")
            elapsed = time.perf_counter() - start
            print(f"\nBuilt [{feature_set}] combined matrix in {elapsed:.1f}s")
            built_here = True
        elif feature_set == "glove+char":
            if X_train_glove_csr is None or X_val_glove_csr is None or X_train_char is None or X_val_char is None:
                raise SystemExit("Feature set 'glove+char' requested but required features were not built.")
            start = time.perf_counter()
            X_train_fs = sparse.hstack([X_train_glove_csr, X_train_char], format="csr")
            X_val_fs = sparse.hstack([X_val_glove_csr, X_val_char], format="csr")
            elapsed = time.perf_counter() - start
            print(f"\nBuilt [{feature_set}] combined matrix in {elapsed:.1f}s")
            built_here = True
        elif feature_set == "glove+bow+char":
            if (
                X_train_glove_csr is None
                or X_val_glove_csr is None
                or X_train_word is None
                or X_val_word is None
                or X_train_char is None
                or X_val_char is None
            ):
                raise SystemExit("Feature set 'glove+bow+char' requested but required features were not built.")
            start = time.perf_counter()
            X_train_fs = sparse.hstack([X_train_glove_csr, X_train_word, X_train_char], format="csr")
            X_val_fs = sparse.hstack([X_val_glove_csr, X_val_word, X_val_char], format="csr")
            elapsed = time.perf_counter() - start
            print(f"\nBuilt [{feature_set}] combined matrix in {elapsed:.1f}s")
            built_here = True
        else:
            raise SystemExit(f"Unknown feature set: {feature_set}")

        results.extend(_evaluate_feature_set(feature_set, X_train_fs, y_train, X_val_fs, y_val, args=args))
        if built_here:
            del X_train_fs
            del X_val_fs

    print("\nFinal comparison (validation accuracy)")
    results_sorted = sorted(results, key=lambda r: r.val_acc, reverse=True)
    for r in results_sorted:
        n_features = r.details.get("n_features")
        train_nnz = r.details.get("train_nnz")
        feat_meta = f"n_features={n_features}"
        if train_nnz is not None:
            feat_meta += f", train_nnz={train_nnz}"
        print(
            f"  [{r.feature_set}] {r.model}: "
            f"val_acc={r.val_acc:.6f} | train_acc={r.train_acc:.6f} | "
            f"time={r.train_time_s:.1f}s | threshold={r.threshold:.6f} | {feat_meta}"
        )
    best = results_sorted[0]
    print(f"Best overall: {best.model} on [{best.feature_set}] (val_acc={best.val_acc:.6f})")


if __name__ == "__main__":
    main()
