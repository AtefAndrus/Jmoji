from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.data.text_preprocessor import normalize_text
from src.generation.dataset_generator import generate_dataset
from src.generation.openrouter_client import OpenRouterClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    teacher_cfg = cfg.get("teacher", {})

    # 入力文はここでは簡略化し、max_samplesぶん空のリストを用意
    # 実際にはWikipediaなどから取得してfillする
    max_samples = int(data_cfg.get("max_samples", 100))
    sentences = [f"ダミー文{idx}" for idx in range(max_samples)]

    client = OpenRouterClient(
        model=teacher_cfg.get("model", "anthropic/claude-haiku-4.5"),
        base_url=teacher_cfg.get("base_url", "https://openrouter.ai/api/v1"),
        timeout=float(teacher_cfg.get("timeout", 60)),
        max_retries=int(teacher_cfg.get("max_retries", 3)),
    )

    normalized = [normalize_text(s) for s in sentences]
    # 実際には extract_sentences で記事から抽出する
    output_dir = Path(data_cfg.get("output_dir", "data/outputs"))
    output_file = data_cfg.get("output_filename", "dataset_v1.jsonl")
    output_path = output_dir / output_file

    generate_dataset(
        client,
        normalized,
        output_path=output_path,
        request_delay=float(teacher_cfg.get("request_delay", 0.5)),
    )


if __name__ == "__main__":
    main()
