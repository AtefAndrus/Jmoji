from src.evaluation.metrics import (
    compute_emoji_stats,
    diversity_ratio,
    emoji_distribution,
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


# diversity_ratio テスト
class TestDiversityRatio:
    """diversity_ratio関数のテスト"""

    def test_basic_diversity(self):
        """基本的な多様性計算"""
        predictions = ["😊 🎉", "😊 ✨"]
        top_n_emojis = {"😊", "🎉"}  # ✨はTop-Nに含まれない

        result = diversity_ratio(predictions, top_n_emojis)

        assert result["total_emojis"] == 4
        assert result["top_n_count"] == 3  # 😊, 🎉, 😊
        assert result["non_top_n_count"] == 1  # ✨
        assert result["non_top_n_ratio"] == 0.25
        assert result["unique_emojis"] == 3

    def test_all_in_top_n(self):
        """すべてTop-Nに含まれる場合"""
        predictions = ["😊 🎉", "😊"]
        top_n_emojis = {"😊", "🎉", "✨"}

        result = diversity_ratio(predictions, top_n_emojis)

        assert result["non_top_n_ratio"] == 0.0
        assert result["non_top_n_count"] == 0

    def test_none_in_top_n(self):
        """すべてTop-Nに含まれない場合"""
        predictions = ["😊 🎉"]
        top_n_emojis = {"✨", "🔥"}

        result = diversity_ratio(predictions, top_n_emojis)

        assert result["non_top_n_ratio"] == 1.0
        assert result["top_n_count"] == 0

    def test_empty_predictions(self):
        """空の予測リスト"""
        result = diversity_ratio([], {"😊"})

        assert result["total_emojis"] == 0
        assert result["non_top_n_ratio"] == 0.0

    def test_empty_prediction_strings(self):
        """空文字列を含む予測"""
        predictions = ["", "😊", ""]
        top_n_emojis = {"😊"}

        result = diversity_ratio(predictions, top_n_emojis)

        assert result["total_emojis"] == 1
        assert result["unique_emojis"] == 1


# compute_emoji_stats テスト
class TestComputeEmojiStats:
    """compute_emoji_stats関数のテスト"""

    def test_basic_stats(self):
        """基本的な統計計算"""
        samples = [
            {"emoji_string": "😊 🎉"},
            {"emoji_string": "😊 😊"},
            {"emoji_string": "✨"},
        ]

        counts, total, unique = compute_emoji_stats(samples)

        assert counts["😊"] == 3
        assert counts["🎉"] == 1
        assert counts["✨"] == 1
        assert total == 5
        assert unique == 3

    def test_empty_samples(self):
        """空のサンプルリスト"""
        counts, total, unique = compute_emoji_stats([])

        assert len(counts) == 0
        assert total == 0
        assert unique == 0

    def test_custom_emoji_key(self):
        """カスタムemoji_keyを使用"""
        samples = [
            {"emojis": "😊 🎉"},
            {"emojis": "😊"},
        ]

        counts, total, unique = compute_emoji_stats(samples, emoji_key="emojis")

        assert counts["😊"] == 2
        assert counts["🎉"] == 1
        assert total == 3
        assert unique == 2

    def test_missing_emoji_key(self):
        """emoji_keyが存在しないサンプル"""
        samples = [
            {"text": "a"},
            {"emoji_string": "😊"},
        ]

        counts, total, unique = compute_emoji_stats(samples)

        assert counts["😊"] == 1
        assert total == 1
        assert unique == 1

    def test_empty_emoji_string(self):
        """空のemoji_stringを持つサンプル"""
        samples = [
            {"emoji_string": ""},
            {"emoji_string": "😊"},
        ]

        counts, total, unique = compute_emoji_stats(samples)

        assert total == 1
        assert unique == 1


# emoji_distribution テスト
class TestEmojiDistribution:
    """emoji_distribution関数のテスト"""

    def test_basic_distribution(self):
        """基本的な分布計算"""
        predictions = ["😊 🎉", "😊 😊", "✨"]

        dist = emoji_distribution(predictions)

        assert dist["😊"] == 3
        assert dist["🎉"] == 1
        assert dist["✨"] == 1
        # 降順でソートされているか確認
        keys = list(dist.keys())
        assert keys[0] == "😊"  # 最も多い

    def test_empty_predictions(self):
        """空の予測リスト"""
        dist = emoji_distribution([])

        assert dist == {}

    def test_empty_prediction_strings(self):
        """空文字列を含む予測"""
        predictions = ["", "😊", ""]

        dist = emoji_distribution(predictions)

        assert dist == {"😊": 1}

    def test_single_emoji_type(self):
        """1種類の絵文字のみ"""
        predictions = ["😊 😊", "😊"]

        dist = emoji_distribution(predictions)

        assert dist == {"😊": 3}
