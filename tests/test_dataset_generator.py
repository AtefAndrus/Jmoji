from pathlib import Path

from src.generation.dataset_generator import DataSample, generate_dataset, load_dataset


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
