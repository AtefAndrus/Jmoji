from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import httpx
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from src.data.emoji_utils import extract_emojis, is_valid_emoji
from src.data.text_preprocessor import remove_emojis
from src.generation import prompts

logger = logging.getLogger(__name__)


@dataclass
class DataSample:
    original_text: str
    sns_text: str
    emojis: List[str]
    emoji_string: str


@dataclass
class GenerationStats:
    """生成統計情報"""

    total: int = 0
    success: int = 0
    skipped: int = 0
    errors: int = 0
    content_rejections: int = 0  # APIコンテンツポリシー拒否数

    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.success / self.total * 100


def is_content_policy_error(error: Exception) -> bool:
    """APIコンテンツポリシー拒否かどうかを判定。

    OpenRouter経由の場合、コンテンツモデレーション違反は403で返される。
    error.metadataにreasonsが含まれる。
    """
    if isinstance(error, httpx.HTTPStatusError):
        if error.response.status_code == 403:
            return True
        # レスポンスボディにmoderationやcontent関連のキーワードがあるか確認
        try:
            body = error.response.text.lower()
            if any(kw in body for kw in ["moderation", "content", "flagged", "policy"]):
                return True
        except Exception:
            pass
    return False


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


def append_sample(sample: DataSample, path: Path) -> None:
    """1件のサンプルを追記保存"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


def load_dataset(path: Path) -> List[DataSample]:
    data: List[DataSample] = []
    if not path.exists():
        return data
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
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


def count_existing_samples(path: Path) -> int:
    """既存ファイルのサンプル数をカウント"""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def generate_dataset(
    client,
    sentences: Sequence[str],
    *,
    output_path: Path,
    request_delay: float = 0.5,
    min_emoji_count: int = 1,
    max_emoji_count: int = 5,
    checkpoint_interval: int = 100,
    resume: bool = True,
    preview_interval: int = 50,
    show_progress: bool = True,
    target_count: Optional[int] = None,
) -> List[DataSample]:
    """データセット生成（進捗表示・途中保存対応）

    Args:
        client: OpenRouterClient インスタンス
        sentences: 入力文のリスト
        output_path: 出力JSONLパス
        request_delay: リクエスト間の待機秒数
        min_emoji_count: 最小絵文字数
        max_emoji_count: 最大絵文字数
        checkpoint_interval: 何件ごとにチェックポイント表示するか
        resume: 既存ファイルがあれば続きから再開するか
        preview_interval: 何件ごとにサンプルプレビューを表示するか
        show_progress: プログレスバーを表示するか
        target_count: 目標サンプル数（Noneの場合は全文を処理）

    Returns:
        生成されたDataSampleのリスト
    """
    stats = GenerationStats()
    samples: List[DataSample] = []

    # 再開処理
    start_idx = 0
    if resume and output_path.exists():
        existing_count = count_existing_samples(output_path)
        if existing_count > 0:
            start_idx = existing_count
            samples = load_dataset(output_path)
            logger.info(f"Resuming from index {start_idx} ({existing_count} samples found)")
            if show_progress:
                print(f"[Resume] {existing_count} samples found, continuing from index {start_idx}")

    # 新規開始の場合はファイルをクリア
    if start_idx == 0 and output_path.exists():
        output_path.unlink()

    sentences_to_process = sentences[start_idx:]
    total = len(sentences_to_process)

    if total == 0:
        logger.info("No sentences to process")
        return samples

    # プログレスバー設定
    pbar: Optional[tqdm] = None
    if show_progress:
        pbar = tqdm(
            total=total,
            desc="Generating",
            unit="sample",
            dynamic_ncols=True,
        )

    # 件数保証: 既存サンプル数を考慮
    existing_success = len(samples)

    for idx, sentence in enumerate(sentences_to_process):
        stats.total += 1
        global_idx = start_idx + idx

        try:
            sns_text = client.complete(
                prompts.SNS_CONVERSION_PROMPT.format(text=sentence)
            ).strip()
            # SNS変換結果から絵文字を除去
            sns_text = remove_emojis(sns_text).strip()
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

            if validate_sample(sample, min_count=min_emoji_count, max_count=max_emoji_count):
                samples.append(sample)
                append_sample(sample, output_path)
                stats.success += 1

                # サンプルプレビュー
                if show_progress and stats.success % preview_interval == 0:
                    _print_preview(sample, stats)

                # 件数保証: 目標に達したら終了
                if target_count and (existing_success + stats.success) >= target_count:
                    logger.info(f"Target count {target_count} reached, stopping")
                    break
            else:
                stats.skipped += 1
                logger.debug(f"Sample {global_idx} skipped (validation failed)")

        except Exception as e:
            if is_content_policy_error(e):
                stats.content_rejections += 1
                logger.warning(
                    f"Content policy rejection at {global_idx}: "
                    f"{sentence[:50]}..."
                )
            else:
                stats.errors += 1
                logger.warning(f"Error at index {global_idx}: {e}")

        # プログレスバー更新
        if pbar:
            pbar.set_postfix(
                success=stats.success,
                skip=stats.skipped,
                err=stats.errors,
                rej=stats.content_rejections,
                rate=f"{stats.success_rate():.1f}%",
            )
            pbar.update(1)

        # チェックポイント表示
        if stats.total % checkpoint_interval == 0:
            logger.info(
                f"Checkpoint: {stats.total}/{total} processed, "
                f"{stats.success} success, {stats.skipped} skipped, {stats.errors} errors"
            )

        if request_delay:
            time.sleep(request_delay)

    if pbar:
        pbar.close()

    # 最終統計
    if show_progress:
        _print_final_stats(stats)

    return samples


def _print_preview(sample: DataSample, stats: GenerationStats) -> None:
    """サンプルプレビューを表示"""
    tqdm.write("")
    tqdm.write(f"--- Preview (#{stats.success}) ---")
    tqdm.write(f"  Input:  {sample.original_text[:50]}...")
    tqdm.write(f"  SNS:    {sample.sns_text[:50]}...")
    tqdm.write(f"  Emojis: {sample.emoji_string}")
    tqdm.write("")


def _print_final_stats(stats: GenerationStats) -> None:
    """最終統計を表示"""
    print("\n" + "=" * 50)
    print("Generation Complete")
    print("=" * 50)
    print(f"  Total processed:      {stats.total}")
    print(f"  Success:              {stats.success} ({stats.success_rate():.1f}%)")
    print(f"  Skipped:              {stats.skipped}")
    print(f"  Errors:               {stats.errors}")
    print(f"  Content rejections:   {stats.content_rejections}")
    print("=" * 50)


async def _process_single_async(
    client,
    sentence: str,
    idx: int,
    min_emoji_count: int,
    max_emoji_count: int,
) -> Tuple[int, Optional[DataSample], str]:
    """1件の非同期処理

    Returns:
        (idx, sample or None, status): status is "success", "skipped", "error", or "content_rejection"
    """
    try:
        sns_text = (
            await client.complete(prompts.SNS_CONVERSION_PROMPT.format(text=sentence))
        ).strip()
        # SNS変換結果から絵文字を除去
        sns_text = remove_emojis(sns_text).strip()
        emoji_output = (
            await client.complete(prompts.EMOJI_GENERATION_PROMPT.format(text=sns_text))
        ).strip()
        emojis = extract_emojis(emoji_output, max_count=max_emoji_count)
        sample = DataSample(
            original_text=sentence,
            sns_text=sns_text,
            emojis=emojis,
            emoji_string=" ".join(emojis),
        )

        if validate_sample(sample, min_count=min_emoji_count, max_count=max_emoji_count):
            return (idx, sample, "success")
        else:
            return (idx, None, "skipped")

    except Exception as e:
        if is_content_policy_error(e):
            logger.warning(f"Content policy rejection at {idx}: {sentence[:50]}...")
            return (idx, None, "content_rejection")
        logger.warning(f"Error at index {idx}: {e}")
        return (idx, None, "error")


async def generate_dataset_async(
    client,
    sentences: Sequence[str],
    *,
    output_path: Path,
    min_emoji_count: int = 1,
    max_emoji_count: int = 5,
    preview_interval: int = 50,
    show_progress: bool = True,
    resume: bool = True,
    target_count: Optional[int] = None,
) -> List[DataSample]:
    """非同期版データセット生成（並列リクエスト対応）

    Args:
        client: AsyncOpenRouterClient インスタンス
        sentences: 入力文のリスト
        output_path: 出力JSONLパス
        min_emoji_count: 最小絵文字数
        max_emoji_count: 最大絵文字数
        preview_interval: 何件ごとにサンプルプレビューを表示するか
        show_progress: プログレスバーを表示するか
        resume: 既存ファイルがあれば続きから再開するか
        target_count: 目標サンプル数（Noneの場合は全文を処理）

    Returns:
        生成されたDataSampleのリスト
    """
    stats = GenerationStats()
    samples: List[DataSample] = []

    # 再開処理
    start_idx = 0
    if resume and output_path.exists():
        existing_count = count_existing_samples(output_path)
        if existing_count > 0:
            start_idx = existing_count
            samples = load_dataset(output_path)
            logger.info(f"Resuming from index {start_idx} ({existing_count} samples found)")
            if show_progress:
                print(f"[Resume] {existing_count} samples found, continuing from index {start_idx}")

    # 新規開始の場合はファイルをクリア
    if start_idx == 0 and output_path.exists():
        output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    sentences_to_process = sentences[start_idx:]
    total = len(sentences_to_process)

    if total == 0:
        logger.info("No sentences to process")
        return samples

    # タスク作成
    tasks = [
        _process_single_async(client, sentence, start_idx + idx, min_emoji_count, max_emoji_count)
        for idx, sentence in enumerate(sentences_to_process)
    ]

    # 非同期実行（プログレスバー付き）
    if show_progress:
        results = []
        async for result in atqdm(
            asyncio.as_completed(tasks),
            total=total,
            desc="Generating (async)",
            unit="sample",
        ):
            results.append(await result)
            stats.total += 1

            # プログレスバー更新用に統計を計算
            _, sample, status = results[-1]
            if status == "success":
                stats.success += 1
            elif status == "skipped":
                stats.skipped += 1
            elif status == "content_rejection":
                stats.content_rejections += 1
            else:
                stats.errors += 1
    else:
        results = await asyncio.gather(*tasks)
        for _, sample, status in results:
            stats.total += 1
            if status == "success":
                stats.success += 1
            elif status == "skipped":
                stats.skipped += 1
            elif status == "content_rejection":
                stats.content_rejections += 1
            else:
                stats.errors += 1

    # 結果をインデックス順にソートして保存
    results.sort(key=lambda x: x[0])

    for idx, sample, status in results:
        if sample is not None:
            samples.append(sample)
            append_sample(sample, output_path)

            # サンプルプレビュー
            if show_progress and len(samples) % preview_interval == 0:
                _print_preview(sample, stats)

            # 件数保証: 目標に達したら保存を終了
            if target_count and len(samples) >= target_count:
                logger.info(f"Target count {target_count} reached, stopping")
                break

    # 最終統計
    if show_progress:
        _print_final_stats(stats)

    return samples


__all__ = [
    "DataSample",
    "GenerationStats",
    "generate_dataset",
    "generate_dataset_async",
    "validate_sample",
    "save_dataset",
    "append_sample",
    "load_dataset",
    "count_existing_samples",
    "is_content_policy_error",
]
