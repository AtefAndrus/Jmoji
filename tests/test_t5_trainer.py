from pathlib import Path

import torch

from src.models.t5_trainer import EmojiDataset, load_jsonl, split_dataset


class FakeTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        text,
        max_length=10,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ):
        length = min(len(text), max_length)
        ids = list(range(length)) + [0] * (max_length - length)
        attn = [1] * length + [0] * (max_length - length)
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.tensor([attn]),
        }


def test_emoji_dataset_shapes():
    samples = [
        {"sns_text": "今日は楽しい", "emoji_string": "😊 🎉"},
        {"sns_text": "明日は晴れ", "emoji_string": "☀️"},
    ]
    tok = FakeTokenizer()
    ds = EmojiDataset(samples, tok, max_input_length=8, max_output_length=4)
    item = ds[0]
    assert item["input_ids"].shape == (8,)
    assert item["attention_mask"].shape == (8,)
    assert item["labels"].shape == (4,)


def test_split_dataset():
    samples = [{"id": i} for i in range(10)]
    train, val, test = split_dataset(samples, 0.6, 0.2)
    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2


def test_split_dataset_shuffle_with_seed():
    """同じseedで同じ結果、異なるseedで異なる結果になることを確認"""
    samples = [{"id": i} for i in range(100)]

    train1, _, _ = split_dataset(samples, 0.8, 0.1, seed=42)
    train2, _, _ = split_dataset(samples, 0.8, 0.1, seed=42)
    train3, _, _ = split_dataset(samples, 0.8, 0.1, seed=123)

    assert train1 == train2  # 同じseedなら同じ結果
    assert train1 != train3  # 異なるseedなら異なる結果


def test_split_dataset_no_shuffle():
    """shuffle=Falseで順序が保持されることを確認"""
    samples = [{"id": i} for i in range(10)]
    train, val, test = split_dataset(samples, 0.6, 0.2, shuffle=False)

    assert [s["id"] for s in train] == [0, 1, 2, 3, 4, 5]
    assert [s["id"] for s in val] == [6, 7]
    assert [s["id"] for s in test] == [8, 9]


def test_load_jsonl(tmp_path: Path):
    """JSONLファイルを正しく読み込めることを確認"""
    jsonl_file = tmp_path / "test.jsonl"
    jsonl_file.write_text(
        '{"sns_text": "テスト1", "emoji_string": "😊"}\n'
        '{"sns_text": "テスト2", "emoji_string": "🎉"}\n',
        encoding="utf-8",
    )

    data = load_jsonl(jsonl_file)
    assert len(data) == 2
    assert data[0]["sns_text"] == "テスト1"
    assert data[1]["emoji_string"] == "🎉"


def test_load_jsonl_empty(tmp_path: Path):
    """空のJSONLファイルは空リストを返す"""
    jsonl_file = tmp_path / "empty.jsonl"
    jsonl_file.write_text("", encoding="utf-8")

    data = load_jsonl(jsonl_file)
    assert data == []
