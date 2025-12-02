from src.evaluation.metrics import (
    exact_match_rate,
    jaccard_similarity,
    length_distribution_analysis,
    micro_f1,
    set_based_metrics,
)


def test_jaccard_and_set_metrics():
    pred = {"😊", "🎉"}
    gold = {"😊", "✨"}
    assert jaccard_similarity(pred, gold) == 1 / 3
    metrics = set_based_metrics(pred, gold)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_micro_f1_and_exact_match():
    preds = [{"a"}, {"b"}]
    golds = [{"a"}, {"c"}]
    assert micro_f1(preds, golds) == 0.5
    assert exact_match_rate(preds, golds) == 0.5


def test_length_distribution_analysis():
    stats = length_distribution_analysis([1, 2, 3], [3, 2, 1])
    assert stats["pred_mean"] == 2.0
    assert stats["gold_mean"] == 2.0
    assert "correlation" in stats


# エッジケーステスト
def test_jaccard_both_empty():
    """両方空の場合は1.0を返す"""
    assert jaccard_similarity(set(), set()) == 1.0


def test_jaccard_one_empty():
    """片方だけ空の場合は0.0を返す"""
    assert jaccard_similarity({"a"}, set()) == 0.0
    assert jaccard_similarity(set(), {"a"}) == 0.0


def test_set_based_metrics_both_empty():
    """両方空の場合は全て1.0を返す"""
    metrics = set_based_metrics(set(), set())
    assert metrics == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_set_based_metrics_pred_empty():
    """predだけ空の場合は全て0.0を返す"""
    metrics = set_based_metrics(set(), {"a"})
    assert metrics == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_set_based_metrics_gold_empty():
    """goldだけ空の場合は全て0.0を返す"""
    metrics = set_based_metrics({"a"}, set())
    assert metrics == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_micro_f1_empty_sets():
    """空の集合リストの場合は0.0を返す"""
    assert micro_f1([set(), set()], [set(), set()]) == 0.0


def test_exact_match_rate_empty_list():
    """空のリストの場合は0.0を返す"""
    assert exact_match_rate([], []) == 0.0


def test_length_distribution_analysis_empty():
    """空の配列の場合はデフォルト値を返す"""
    stats = length_distribution_analysis([], [])
    assert stats["pred_mean"] == 0.0
    assert stats["gold_mean"] == 0.0
    assert stats["correlation"] == 0.0
