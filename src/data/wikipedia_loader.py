from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Set

from datasets import load_dataset  # type: ignore[import-untyped]

from src.data.text_preprocessor import (
    DEFAULT_NSFW_KEYWORDS,
    extract_sentences,
    is_complete_sentence,
    is_safe_sentence,
    normalize_text,
)

logger = logging.getLogger(__name__)


class FilterLog:
    """フィルタで除外された文をログに記録するクラス。"""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path
        self._file = None
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = log_path.open("w", encoding="utf-8")

    def log(self, text: str, reason: str, detail: str = "") -> None:
        """除外された文をログに記録。"""
        if self._file:
            entry = {"reason": reason, "detail": detail, "text": text}
            self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def load_wikipedia_sentences(
    *,
    max_samples: int,
    subset: str = "20231101.ja",
    dataset_name: str = "wikimedia/wikipedia",
    seed: int = 42,
    min_len: int = 10,
    max_len_text: int = 100,
    buffer_ratio: float = 1.3,
    nsfw_keywords: Optional[Set[str]] = None,
    filter_log_path: Optional[Path] = None,
    apply_complete_filter: bool = True,
    apply_nsfw_filter: bool = True,
) -> List[str]:
    """Wikipediaデータセットから文を抽出して返す。

    - `datasets` ライブラリでデータをストリーミングし、文を抽出。
    - 長さフィルタと日本語文字チェックは `extract_sentences` に委譲。
    - 完全な文かどうかのフィルタを適用（オプション）。
    - NSFWフィルタを適用（オプション）。
    - フィルタで除外された文はログに記録。

    Args:
        max_samples: 取得する最大サンプル数
        subset: Wikipediaのサブセット
        dataset_name: データセット名
        seed: シャッフル用シード
        min_len: 最小文字数
        max_len_text: 最大文字数
        buffer_ratio: 件数保証のためのバッファ率（1.3 = 30%多く取得）
        nsfw_keywords: NSFWキーワード（Noneでデフォルト使用）
        filter_log_path: フィルタログの出力パス（Noneでログ出力しない）
        apply_complete_filter: 完全な文フィルタを適用するか
        apply_nsfw_filter: NSFWフィルタを適用するか

    Returns:
        フィルタ適用済みの文のリスト
    """
    if nsfw_keywords is None:
        nsfw_keywords = DEFAULT_NSFW_KEYWORDS

    # 件数保証のため多めに取得
    target_count = int(max_samples * buffer_ratio)

    ds = load_dataset(dataset_name, subset, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    sentences: List[str] = []
    stats = {"total": 0, "nsfw": 0, "incomplete": 0, "accepted": 0}

    with FilterLog(filter_log_path) as flog:
        for item in ds:
            text = normalize_text(item.get("text", ""))
            sents = extract_sentences(text, min_len=min_len, max_len=max_len_text)

            for s in sents:
                stats["total"] += 1

                # NSFWフィルタ
                if apply_nsfw_filter and not is_safe_sentence(s, nsfw_keywords):
                    stats["nsfw"] += 1
                    # どのキーワードにマッチしたか特定
                    matched = [kw for kw in nsfw_keywords if kw in s]
                    flog.log(s, "nsfw", ",".join(matched))
                    continue

                # 完全な文フィルタ
                if apply_complete_filter:
                    is_complete, reason = is_complete_sentence(s)
                    if not is_complete:
                        stats["incomplete"] += 1
                        flog.log(s, "incomplete", reason)
                        continue

                sentences.append(s)
                stats["accepted"] += 1

                if len(sentences) >= target_count:
                    break

            if len(sentences) >= target_count:
                break

    logger.info(
        f"Wikipedia loader stats: total={stats['total']}, "
        f"nsfw={stats['nsfw']}, incomplete={stats['incomplete']}, "
        f"accepted={stats['accepted']}"
    )

    return sentences


__all__ = ["load_wikipedia_sentences", "FilterLog"]
