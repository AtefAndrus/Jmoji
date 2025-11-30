from __future__ import annotations

import re
import unicodedata
from typing import List

_JP_CHAR_PATTERN = re.compile(r"[ぁ-んァ-ン一-龥]")
_SENT_SPLIT = re.compile(r"(?<=[。！？!?])")


def normalize_text(text: str) -> str:
    """NFKC正規化と空白整理を行う。"""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_sentences(text: str, min_len: int = 10, max_len: int = 100) -> List[str]:
    """日本語文を簡易分割し、長さと日本語文字の有無でフィルタ。"""
    candidates = _SENT_SPLIT.split(text)
    results: List[str] = []
    for s in candidates:
        s = s.strip()
        if not s:
            continue
        if not (min_len <= len(s) <= max_len):
            continue
        if not _JP_CHAR_PATTERN.search(s):
            continue
        results.append(s)
    return results


__all__ = ["normalize_text", "extract_sentences"]
