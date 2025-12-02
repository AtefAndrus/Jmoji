from pathlib import Path

from src.generation.dataset_generator import (
    DataSample,
    generate_dataset,
    load_dataset,
    validate_sample,
)


class FakeClient:
    def __init__(self):
        self.calls = 0

    def complete(self, prompt: str) -> str:
        # 偶数回: SNS変換、奇数回: 絵文字生成
        if self.calls % 2 == 0:
            result = "SNS文" + str(self.calls // 2)
        else:
            result = "😊 🎉"
        self.calls += 1
        return result


def test_generate_dataset_creates_jsonl(tmp_path: Path):
    client = FakeClient()
    sentences = ["今日は楽しい", "明日は晴れる"]
    out = tmp_path / "dataset.jsonl"

    samples = generate_dataset(client, sentences, output_path=out, request_delay=0)
    assert len(samples) == 2
    assert out.exists()

    loaded = load_dataset(out)
    assert isinstance(loaded[0], DataSample)
    assert loaded[0].emojis == ["😊", "🎉"]


def test_validate_sample_valid():
    """有効なサンプルはTrueを返す"""
    sample = DataSample(
        original_text="元の文",
        sns_text="SNS文",
        emojis=["😊", "🎉"],
        emoji_string="😊 🎉",
    )
    assert validate_sample(sample) is True


def test_validate_sample_invalid_empty_sns_text():
    """SNSテキストが空の場合はFalseを返す"""
    sample = DataSample(
        original_text="元の文",
        sns_text="",
        emojis=["😊"],
        emoji_string="😊",
    )
    assert validate_sample(sample) is False


def test_validate_sample_invalid_emoji_count_zero():
    """絵文字が0個の場合はFalseを返す"""
    sample = DataSample(
        original_text="元の文",
        sns_text="SNS文",
        emojis=[],
        emoji_string="",
    )
    assert validate_sample(sample, min_count=1) is False


def test_validate_sample_invalid_emoji_count_exceeds_max():
    """絵文字が最大数を超える場合はFalseを返す"""
    sample = DataSample(
        original_text="元の文",
        sns_text="SNS文",
        emojis=["😊", "🎉", "✨", "💕", "🔥", "⭐"],
        emoji_string="😊 🎉 ✨ 💕 🔥 ⭐",
    )
    assert validate_sample(sample, max_count=5) is False


def test_validate_sample_invalid_non_emoji():
    """絵文字でない文字が含まれる場合はFalseを返す"""
    sample = DataSample(
        original_text="元の文",
        sns_text="SNS文",
        emojis=["😊", "abc"],
        emoji_string="😊 abc",
    )
    assert validate_sample(sample) is False
