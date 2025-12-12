from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Sequence, Set, Tuple

import emoji

SKIN_TONE_PATTERN = re.compile(r"[\U0001F3FB-\U0001F3FF]")


def get_all_emojis() -> Set[str]:
    return set(emoji.EMOJI_DATA.keys())


def normalize_skin_tone(text: str) -> str:
    return SKIN_TONE_PATTERN.sub("", text)


def extract_emojis(text: str, max_count: int = 5) -> List[str]:
    items = emoji.emoji_list(text)
    emojis = [normalize_skin_tone(item["emoji"]) for item in items]
    return emojis[:max_count]


def is_valid_emoji(e: str) -> bool:
    return emoji.is_emoji(e)


def filter_samples_by_top_emojis(
    samples: Sequence[Dict[str, Any]],
    top_n: int = 100,
    emoji_key: str = "emoji_string",
) -> Tuple[List[Dict[str, Any]], Counter[str], Set[str]]:
    """Top-N絵文字のみを含むサンプルにフィルタリングする。

    Args:
        samples: サンプルリスト（各サンプルはemoji_keyを含む辞書）
        top_n: 使用するTop絵文字の数
        emoji_key: 絵文字文字列が格納されているキー名

    Returns:
        Tuple of:
        - filtered_samples: フィルタ後のサンプルリスト
        - emoji_counts: フィルタ後の絵文字カウント（Counter）
        - top_emojis: Top-N絵文字のセット
    """
    # 全サンプルから絵文字を集計
    all_emojis: List[str] = []
    for sample in samples:
        emoji_str = sample.get(emoji_key, "")
        if emoji_str:
            all_emojis.extend(emoji_str.split())

    original_counts: Counter[str] = Counter(all_emojis)
    top_emojis = set(e for e, _ in original_counts.most_common(top_n))

    # フィルタリング（全絵文字がTop-Nに含まれるサンプルのみ）
    filtered: List[Dict[str, Any]] = []
    for sample in samples:
        emoji_str = sample.get(emoji_key, "")
        if emoji_str:
            sample_emojis = emoji_str.split()
            if all(e in top_emojis for e in sample_emojis):
                filtered.append(sample)

    # フィルタ後の統計を再計算
    filtered_emojis: List[str] = []
    for sample in filtered:
        emoji_str = sample.get(emoji_key, "")
        if emoji_str:
            filtered_emojis.extend(emoji_str.split())

    filtered_counts: Counter[str] = Counter(filtered_emojis)

    return filtered, filtered_counts, top_emojis


__all__ = [
    "get_all_emojis",
    "normalize_skin_tone",
    "extract_emojis",
    "is_valid_emoji",
    "filter_samples_by_top_emojis",
    "SKIN_TONE_PATTERN",
]
