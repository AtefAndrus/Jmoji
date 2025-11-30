from __future__ import annotations

import re
from typing import List, Set

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


__all__ = [
    "get_all_emojis",
    "normalize_skin_tone",
    "extract_emojis",
    "is_valid_emoji",
    "SKIN_TONE_PATTERN",
]
