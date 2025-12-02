from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv

from src.config import load_config
from src.data.text_preprocessor import normalize_text
from src.data.wikipedia_loader import load_wikipedia_sentences
from src.generation.dataset_generator import generate_dataset, generate_dataset_async
from src.generation.openrouter_client import AsyncOpenRouterClient, OpenRouterClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate emoji dataset from Wikipedia")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--async", dest="use_async", action="store_true", help="Use async mode")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Don't resume from existing file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    teacher_cfg = cfg.get("teacher", {})
    emoji_cfg = cfg.get("emoji", {})

    # 設定から非同期モードを取得（コマンドライン引数が優先）
    use_async = args.use_async or teacher_cfg.get("use_async", False)

    logger.info("Loading Wikipedia sentences...")
    sentences = load_wikipedia_sentences(
        max_samples=int(data_cfg.get("max_samples", 1000)),
        subset=str(data_cfg.get("wikipedia_subset", "20231101.ja")),
        dataset_name=str(data_cfg.get("wikipedia_dataset", "wikimedia/wikipedia")),
        seed=int(data_cfg.get("random_seed", 42)),
        min_len=int(data_cfg.get("min_text_length", 10)),
        max_len_text=int(data_cfg.get("max_text_length", 100)),
    )
    sentences = [normalize_text(s) for s in sentences]
    logger.info(f"Loaded {len(sentences)} sentences")

    output_dir = Path(data_cfg.get("output_dir", "data/outputs"))
    output_file = data_cfg.get("output_filename", "dataset_v1.jsonl")
    output_path = output_dir / output_file

    if use_async:
        logger.info("Using async mode")
        asyncio.run(
            _run_async(
                sentences=sentences,
                output_path=output_path,
                teacher_cfg=teacher_cfg,
                emoji_cfg=emoji_cfg,
                data_cfg=data_cfg,
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
        )


def _run_sync(
    sentences: list[str],
    output_path: Path,
    teacher_cfg: dict,
    emoji_cfg: dict,
    data_cfg: dict,
    resume: bool,
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
    )

    client.close()


async def _run_async(
    sentences: list[str],
    output_path: Path,
    teacher_cfg: dict,
    emoji_cfg: dict,
    data_cfg: dict,
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
        )


if __name__ == "__main__":
    main()
