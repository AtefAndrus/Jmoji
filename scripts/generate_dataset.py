from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.config import load_config
from src.data.text_preprocessor import normalize_text
from src.data.wikipedia_loader import load_wikipedia_sentences
from src.generation.dataset_generator import generate_dataset
from src.generation.openrouter_client import OpenRouterClient

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    teacher_cfg = cfg.get("teacher", {})
    emoji_cfg = cfg.get("emoji", {})

    sentences = load_wikipedia_sentences(
        max_samples=int(data_cfg.get("max_samples", 1000)),
        subset=str(data_cfg.get("wikipedia_subset", "20231101.ja")),
        dataset_name=str(data_cfg.get("wikipedia_dataset", "wikimedia/wikipedia")),
        seed=int(data_cfg.get("random_seed", 42)),
        min_len=int(data_cfg.get("min_text_length", 10)),
        max_len_text=int(data_cfg.get("max_text_length", 100)),
    )
    sentences = [normalize_text(s) for s in sentences]

    client = OpenRouterClient(
        model=teacher_cfg.get("model", "anthropic/claude-haiku-4.5"),
        base_url=teacher_cfg.get("base_url", "https://openrouter.ai/api/v1"),
        timeout=float(teacher_cfg.get("timeout", 60)),
        max_retries=int(teacher_cfg.get("max_retries", 3)),
    )

    output_dir = Path(data_cfg.get("output_dir", "data/outputs"))
    output_file = data_cfg.get("output_filename", "dataset_v1.jsonl")
    output_path = output_dir / output_file

    generate_dataset(
        client,
        sentences,
        output_path=output_path,
        request_delay=float(teacher_cfg.get("request_delay", 0.5)),
        min_emoji_count=int(emoji_cfg.get("min_count", 1)),
        max_emoji_count=int(emoji_cfg.get("max_count", 5)),
    )


if __name__ == "__main__":
    main()
