from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, Sequence, Set, Tuple

import numpy as np


def jaccard_similarity(pred: Set[str], gold: Set[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    intersection = len(pred & gold)
    union = len(pred | gold)
    return intersection / union


def set_based_metrics(pred: Set[str], gold: Set[str]) -> dict[str, float]:
    if not pred and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    intersection = len(pred & gold)
    precision = intersection / len(pred)
    recall = intersection / len(gold)
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def micro_f1(predictions: Sequence[Set[str]], golds: Sequence[Set[str]]) -> float:
    total_intersection = 0
    total_pred = 0
    total_gold = 0
    for p, g in zip(predictions, golds):
        total_intersection += len(p & g)
        total_pred += len(p)
        total_gold += len(g)
    if total_pred == 0 or total_gold == 0:
        return 0.0
    precision = total_intersection / total_pred
    recall = total_intersection / total_gold
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match_rate(
    predictions: Sequence[Set[str]], golds: Sequence[Set[str]]
) -> float:
    matches = sum(1 for p, g in zip(predictions, golds) if p == g)
    return matches / len(predictions) if predictions else 0.0


def length_distribution_analysis(
    pred_lengths: Iterable[int], gold_lengths: Iterable[int]
) -> dict[str, float]:
    pred_arr = np.array(list(pred_lengths))
    gold_arr = np.array(list(gold_lengths))
    if len(pred_arr) == 0 or len(gold_arr) == 0:
        return {
            "pred_mean": 0.0,
            "pred_std": 0.0,
            "gold_mean": 0.0,
            "gold_std": 0.0,
            "correlation": 0.0,
        }
    return {
        "pred_mean": float(np.mean(pred_arr)),
        "pred_std": float(np.std(pred_arr)),
        "gold_mean": float(np.mean(gold_arr)),
        "gold_std": float(np.std(gold_arr)),
        "correlation": float(np.corrcoef(pred_arr, gold_arr)[0, 1])
        if len(pred_arr) > 1 and len(gold_arr) > 1
        else 0.0,
    }


def diversity_ratio(
    predictions: Sequence[str], top_n_emojis: Set[str]
) -> dict[str, float]:
    """Top-N絵文字以外の出力割合を計算する（多様性指標）。

    Args:
        predictions: 予測結果のリスト（スペース区切りの絵文字文字列）
        top_n_emojis: Top-N絵文字のセット

    Returns:
        dict with:
        - non_top_n_ratio: Top-N以外の絵文字の割合
        - unique_emojis: ユニークな絵文字数
        - total_emojis: 総絵文字数
        - top_n_count: Top-N絵文字の出現数
        - non_top_n_count: Top-N以外の絵文字の出現数
    """
    total = 0
    non_top_n = 0
    all_emojis: list[str] = []

    for pred in predictions:
        emojis = pred.split() if pred else []
        all_emojis.extend(emojis)
        total += len(emojis)
        non_top_n += sum(1 for e in emojis if e not in top_n_emojis)

    unique_emojis = len(set(all_emojis))
    ratio = non_top_n / total if total > 0 else 0.0

    return {
        "non_top_n_ratio": ratio,
        "unique_emojis": unique_emojis,
        "total_emojis": total,
        "top_n_count": total - non_top_n,
        "non_top_n_count": non_top_n,
    }


def compute_emoji_stats(
    samples: Sequence[Dict[str, Any]],
    emoji_key: str = "emoji_string",
) -> Tuple[Counter[str], int, int]:
    """サンプルリストから絵文字統計を計算する。

    Args:
        samples: サンプルリスト（各サンプルはemoji_keyを含む辞書）
        emoji_key: 絵文字文字列が格納されているキー名

    Returns:
        Tuple of:
        - emoji_counts: 絵文字ごとの出現回数（Counter）
        - total_count: 総絵文字数
        - unique_count: ユニークな絵文字数
    """
    all_emojis: list[str] = []
    for sample in samples:
        emoji_str = sample.get(emoji_key, "")
        if emoji_str:
            emojis = emoji_str.split()
            all_emojis.extend(emojis)

    counts: Counter[str] = Counter(all_emojis)
    return counts, len(all_emojis), len(counts)


def emoji_distribution(predictions: Sequence[str]) -> dict[str, int]:
    """予測結果の絵文字分布を計算する。

    Args:
        predictions: 予測結果のリスト（スペース区切りの絵文字文字列）

    Returns:
        絵文字ごとの出現回数（降順でソート）
    """
    all_emojis: list[str] = []
    for pred in predictions:
        emojis = pred.split() if pred else []
        all_emojis.extend(emojis)

    counts: Counter[str] = Counter(all_emojis)
    return dict(counts.most_common())


__all__ = [
    "jaccard_similarity",
    "set_based_metrics",
    "micro_f1",
    "exact_match_rate",
    "length_distribution_analysis",
    "diversity_ratio",
    "compute_emoji_stats",
    "emoji_distribution",
]
