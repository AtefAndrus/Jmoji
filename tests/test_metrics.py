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
