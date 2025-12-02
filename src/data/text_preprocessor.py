from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Set

_JP_CHAR_PATTERN = re.compile(r"[ぁ-んァ-ン一-龥]")
_SENT_SPLIT = re.compile(r"(?<=[。！？!?])")

# デフォルトのNSFWキーワード（設定ファイルで上書き可能）
DEFAULT_NSFW_KEYWORDS: Set[str] = {
    "性行為",
    "性交",
    "ポルノ",
    "アダルト",
    "風俗",
    "売春",
    "淫行",
    "殺人",
    "虐殺",
    "拷問",
    "処刑",
    "惨殺",
}


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


def is_safe_sentence(
    text: str,
    keywords: Optional[Set[str]] = None,
) -> bool:
    """NSFWキーワードを含まないかチェック。

    Args:
        text: チェック対象のテキスト
        keywords: NSFWキーワードのセット（Noneの場合はデフォルトを使用）

    Returns:
        NSFWキーワードを含まない場合True
    """
    if keywords is None:
        keywords = DEFAULT_NSFW_KEYWORDS
    return not any(kw in text for kw in keywords)


def filter_safe_sentences(
    sentences: List[str],
    keywords: Optional[Set[str]] = None,
) -> List[str]:
    """NSFWキーワードを含む文をフィルタリング。

    Args:
        sentences: フィルタ対象の文のリスト
        keywords: NSFWキーワードのセット（Noneの場合はデフォルトを使用）

    Returns:
        安全な文のリスト
    """
    if keywords is None:
        keywords = DEFAULT_NSFW_KEYWORDS
    return [s for s in sentences if is_safe_sentence(s, keywords)]


__all__ = [
    "normalize_text",
    "extract_sentences",
    "is_safe_sentence",
    "filter_safe_sentences",
    "DEFAULT_NSFW_KEYWORDS",
]
