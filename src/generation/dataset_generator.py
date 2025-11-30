from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

from src.data.emoji_utils import extract_emojis, is_valid_emoji
from src.generation import prompts


@dataclass
class DataSample:
    original_text: str
    sns_text: str
    emojis: List[str]
    emoji_string: str


def validate_sample(sample: DataSample, min_count: int = 1, max_count: int = 5) -> bool:
    if not (min_count <= len(sample.emojis) <= max_count):
        return False
    if not sample.sns_text.strip():
        return False
    for e in sample.emojis:
        if not is_valid_emoji(e):
            return False
    return True


def save_dataset(samples: Sequence[DataSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")


def load_dataset(path: Path) -> List[DataSample]:
    data: List[DataSample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append(
                DataSample(
                    original_text=obj["original_text"],
                    sns_text=obj["sns_text"],
                    emojis=list(obj["emojis"]),
                    emoji_string=obj["emoji_string"],
                )
            )
    return data


def generate_dataset(
    client,
    sentences: Sequence[str],
    *,
    output_path: Path,
    request_delay: float = 0.5,
    min_emoji_count: int = 1,
    max_emoji_count: int = 5,
) -> List[DataSample]:
    samples: List[DataSample] = []
    for idx, sentence in enumerate(sentences):
        try:
            sns_text = client.complete(
                prompts.SNS_CONVERSION_PROMPT.format(text=sentence)
            ).strip()
            emoji_output = client.complete(
                prompts.EMOJI_GENERATION_PROMPT.format(text=sns_text)
            ).strip()
            emojis = extract_emojis(emoji_output, max_count=max_emoji_count)
            sample = DataSample(
                original_text=sentence,
                sns_text=sns_text,
                emojis=emojis,
                emoji_string=" ".join(emojis),
            )
            if validate_sample(
                sample, min_count=min_emoji_count, max_count=max_emoji_count
            ):
                samples.append(sample)
        except Exception:
            continue
        if request_delay:
            time.sleep(request_delay)
        # 定期保存は簡略化、最後にまとめて保存
    save_dataset(samples, output_path)
    return samples


__all__ = [
    "DataSample",
    "generate_dataset",
    "validate_sample",
    "save_dataset",
    "load_dataset",
]
