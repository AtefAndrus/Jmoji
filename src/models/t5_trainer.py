from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
)

from src.data.emoji_utils import get_all_emojis


class EmojiDataset(Dataset):
    """JSONLから読み込んだサンプルをT5用テンソルに変換するDataset。

    samples: List[dict] を受け取り、"sns_text" -> 入力、"emoji_string" -> 出力
    """

    def __init__(
        self,
        samples: Sequence[Dict[str, Any]],
        tokenizer: Any,
        max_input_length: int = 128,
        max_output_length: int = 32,
    ) -> None:
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        sample = self.samples[idx]
        input_text = sample["sns_text"]
        output_text = sample["emoji_string"]

        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        output_encoding = self.tokenizer(
            output_text,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "labels": output_encoding["input_ids"].squeeze(0),
        }


@dataclass
class TrainConfig:
    model_name: str
    output_dir: str
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    logging_steps: int = 100
    warmup_steps: int = 500
    fp16: bool = True


def setup_model_with_emoji_tokens(
    model_name: str = "sonoisa/t5-base-japanese",
) -> Tuple[T5Tokenizer, T5ForConditionalGeneration]:
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    emoji_tokens = list(get_all_emojis())
    num_added = tokenizer.add_tokens(emoji_tokens)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


def build_trainer(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    cfg: TrainConfig,
) -> Trainer:
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        logging_steps=cfg.logging_steps,
        warmup_steps=cfg.warmup_steps,
        fp16=cfg.fp16,
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )


def split_dataset(
    samples: Sequence[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    *,
    shuffle: bool = True,
    seed: Optional[int] = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    data = list(samples)
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(data)
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]
    return train, val, test


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


__all__ = [
    "EmojiDataset",
    "TrainConfig",
    "setup_model_with_emoji_tokens",
    "build_trainer",
    "split_dataset",
    "load_jsonl",
]
