import torch

from src.models.t5_trainer import EmojiDataset, split_dataset


class FakeTokenizer:
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
    samples = list(range(10))
    train, val, test = split_dataset(samples, 0.6, 0.2)
    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2
