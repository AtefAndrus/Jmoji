from __future__ import annotations

from typing import Iterable, Sequence, Set

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


__all__ = [
    "jaccard_similarity",
    "set_based_metrics",
    "micro_f1",
    "exact_match_rate",
    "length_distribution_analysis",
]
