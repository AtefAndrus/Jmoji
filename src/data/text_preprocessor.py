from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Set

import emoji

_JP_CHAR_PATTERN = re.compile(r"[ぁ-んァ-ン一-龥]")
_SENT_SPLIT = re.compile(r"(?<=[。！？!?])")

# 半端な文を検出するパターン
_META_SECTION_PATTERN = re.compile(r"^(関連項目|参考文献|外部リンク|脚注|出典|注釈)")
_TRUNCATED_END_PATTERN = re.compile(r"[（(「『][^）)」』]{0,30}$")
_ORPHAN_CLOSE_PATTERN = re.compile(r"^[」』）)]")
_VALID_ENDING_PATTERN = re.compile(r"[。！？!?」』)]$")

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


def is_complete_sentence(text: str) -> tuple[bool, str]:
    """文として完全かどうかを判定。

    Args:
        text: 判定対象のテキスト

    Returns:
        (完全かどうか, 不完全な場合の理由)
    """
    # メタセクション（Wikipediaの構造情報）
    if _META_SECTION_PATTERN.match(text):
        return False, "meta_section"

    # 途中切れ（開き括弧で終わる）
    if _TRUNCATED_END_PATTERN.search(text):
        return False, "truncated"

    # 閉じ括弧で始まる（前の文脈がない）
    if _ORPHAN_CLOSE_PATTERN.match(text):
        return False, "orphan_close"

    # 句読点なし
    if not _VALID_ENDING_PATTERN.search(text):
        return False, "no_ending"

    return True, ""


def filter_complete_sentences(
    sentences: List[str],
) -> tuple[List[str], List[tuple[str, str]]]:
    """完全な文のみをフィルタリング。

    Args:
        sentences: フィルタ対象の文のリスト

    Returns:
        (完全な文のリスト, [(除外された文, 理由), ...])
    """
    complete = []
    filtered = []
    for s in sentences:
        is_complete, reason = is_complete_sentence(s)
        if is_complete:
            complete.append(s)
        else:
            filtered.append((s, reason))
    return complete, filtered


def remove_emojis(text: str) -> str:
    """テキストから絵文字を除去。

    Args:
        text: 絵文字を除去するテキスト

    Returns:
        絵文字を除去したテキスト
    """
    return emoji.replace_emoji(text, replace="")


__all__ = [
    "normalize_text",
    "extract_sentences",
    "is_safe_sentence",
    "filter_safe_sentences",
    "is_complete_sentence",
    "filter_complete_sentences",
    "remove_emojis",
    "DEFAULT_NSFW_KEYWORDS",
]
