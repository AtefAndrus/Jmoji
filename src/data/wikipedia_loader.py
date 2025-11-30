from __future__ import annotations

from typing import List

from datasets import load_dataset  # type: ignore[import-untyped]

from src.data.text_preprocessor import extract_sentences, normalize_text


def load_wikipedia_sentences(
    *,
    max_samples: int,
    subset: str = "20231101.ja",
    dataset_name: str = "wikimedia/wikipedia",
    seed: int = 42,
    min_len: int = 10,
    max_len_text: int = 100,
) -> List[str]:
    """Wikipediaデータセットから文を抽出して返す。

    - `datasets` ライブラリでデータをストリーミングし、文を抽出。
    - 長さフィルタと日本語文字チェックは `extract_sentences` に委譲。
    """

    ds = load_dataset(dataset_name, subset, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    sentences: List[str] = []
    for item in ds:
        text = normalize_text(item.get("text", ""))
        sents = extract_sentences(text, min_len=min_len, max_len=max_len_text)
        for s in sents:
            sentences.append(s)
            if len(sentences) >= max_samples:
                return sentences
    return sentences


__all__ = ["load_wikipedia_sentences"]
