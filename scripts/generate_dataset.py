from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Optional, Set

from dotenv import load_dotenv

from src.config import load_config
from src.data.wikipedia_loader import load_wikipedia_sentences
from src.generation.dataset_generator import generate_dataset, generate_dataset_async
from src.generation.openrouter_client import AsyncOpenRouterClient, OpenRouterClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# httpxのログを抑制（リクエストごとのログが邪魔なため）
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def _get_nsfw_keywords(data_cfg: dict) -> Optional[Set[str]]:
    """設定からNSFWキーワードを取得"""
    nsfw_cfg = data_cfg.get("nsfw_filter", {})
    if not nsfw_cfg.get("enabled", True):
        return set()  # 空セット = フィルタ無効
    keywords = nsfw_cfg.get("keywords", [])
    if keywords:
        return set(keywords)
    return None  # デフォルトキーワードを使用


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate emoji dataset from Wikipedia")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--async", dest="use_async", action="store_true", help="Use async mode")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Don't resume from existing file")
    parser.add_argument("--no-nsfw-filter", dest="nsfw_filter", action="store_false", help="Disable NSFW filter")
    parser.add_argument("--no-complete-filter", dest="complete_filter", action="store_false", help="Disable incomplete sentence filter")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    teacher_cfg = cfg.get("teacher", {})
    emoji_cfg = cfg.get("emoji", {})

    # 設定から非同期モードを取得（コマンドライン引数が優先）
    use_async = args.use_async or teacher_cfg.get("use_async", False)

    # 出力パス設定
    output_dir = Path(data_cfg.get("output_dir", "data/outputs"))
    output_file = data_cfg.get("output_filename", "dataset_v3.jsonl")
    output_path = output_dir / output_file

    # フィルタログパス
    filter_log_filename = data_cfg.get("filter_log_filename", "filtered_sentences.jsonl")
    filter_log_path = output_dir / filter_log_filename

    # フィルタ設定
    apply_nsfw = args.nsfw_filter and data_cfg.get("nsfw_filter", {}).get("enabled", True)
    apply_complete = args.complete_filter and data_cfg.get("complete_sentence_filter", True)
    nsfw_keywords = _get_nsfw_keywords(data_cfg) if apply_nsfw else set()

    # 目標サンプル数
    target_count = int(data_cfg.get("max_samples", 5000))

    logger.info("Loading Wikipedia sentences with pre-filtering...")
    sentences = load_wikipedia_sentences(
        max_samples=target_count,
        subset=str(data_cfg.get("wikipedia_subset", "20231101.ja")),
        dataset_name=str(data_cfg.get("wikipedia_dataset", "wikimedia/wikipedia")),
        seed=int(data_cfg.get("random_seed", 42)),
        min_len=int(data_cfg.get("min_text_length", 10)),
        max_len_text=int(data_cfg.get("max_text_length", 100)),
        buffer_ratio=float(data_cfg.get("buffer_ratio", 1.3)),
        nsfw_keywords=nsfw_keywords,
        filter_log_path=filter_log_path,
        apply_complete_filter=apply_complete,
        apply_nsfw_filter=apply_nsfw,
    )
    logger.info(f"Loaded {len(sentences)} sentences (target: {target_count})")

    if use_async:
        logger.info("Using async mode")
        asyncio.run(
            _run_async(
                sentences=sentences,
                output_path=output_path,
                teacher_cfg=teacher_cfg,
                emoji_cfg=emoji_cfg,
                data_cfg=data_cfg,
                resume=args.resume,
                target_count=target_count,
            )
        )
    else:
        logger.info("Using sync mode")
        _run_sync(
            sentences=sentences,
            output_path=output_path,
            teacher_cfg=teacher_cfg,
            emoji_cfg=emoji_cfg,
            data_cfg=data_cfg,
            resume=args.resume,
            target_count=target_count,
        )


def _run_sync(
    sentences: list[str],
    output_path: Path,
    teacher_cfg: dict,
    emoji_cfg: dict,
    data_cfg: dict,
    resume: bool,
    target_count: int,
) -> None:
    """同期モードでデータセット生成"""
    client = OpenRouterClient(
        model=teacher_cfg.get("model", "anthropic/claude-haiku-4.5"),
        base_url=teacher_cfg.get("base_url", "https://openrouter.ai/api/v1"),
        timeout=float(teacher_cfg.get("timeout", 60)),
        max_retries=int(teacher_cfg.get("max_retries", 3)),
    )

    generate_dataset(
        client,
        sentences,
        output_path=output_path,
        request_delay=float(teacher_cfg.get("request_delay", 0.5)),
        min_emoji_count=int(emoji_cfg.get("min_count", 1)),
        max_emoji_count=int(emoji_cfg.get("max_count", 5)),
        checkpoint_interval=int(data_cfg.get("checkpoint_interval", 100)),
        resume=resume,
        preview_interval=int(data_cfg.get("preview_interval", 50)),
        target_count=target_count,
    )

    client.close()


async def _run_async(
    sentences: list[str],
    output_path: Path,
    teacher_cfg: dict,
    emoji_cfg: dict,
    data_cfg: dict,
    resume: bool,
    target_count: int,
) -> None:
    """非同期モードでデータセット生成"""
    async with AsyncOpenRouterClient(
        model=teacher_cfg.get("model", "anthropic/claude-haiku-4.5"),
        base_url=teacher_cfg.get("base_url", "https://openrouter.ai/api/v1"),
        timeout=float(teacher_cfg.get("timeout", 60)),
        max_retries=int(teacher_cfg.get("max_retries", 3)),
        max_concurrent=int(teacher_cfg.get("max_concurrent", 5)),
    ) as client:
        await generate_dataset_async(
            client,
            sentences,
            output_path=output_path,
            min_emoji_count=int(emoji_cfg.get("min_count", 1)),
            max_emoji_count=int(emoji_cfg.get("max_count", 5)),
            preview_interval=int(data_cfg.get("preview_interval", 50)),
            resume=resume,
            target_count=target_count,
        )


if __name__ == "__main__":
    main()
