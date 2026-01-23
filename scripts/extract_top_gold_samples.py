#!/usr/bin/env python3
"""Goldスコア上位サンプルを抽出するスクリプト.

使用方法:
    # デフォルト実行（JSON + Markdown出力）
    uv run scripts/extract_top_gold_samples.py

    # すべての形式で出力
    uv run scripts/extract_top_gold_samples.py \
        --format json,markdown,csv \
        --output outputs/llm_multi_eval/top_gold_samples

    # 上位10件に変更
    uv run scripts/extract_top_gold_samples.py --top-n 10
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_all_responses(response_dir: Path) -> list[dict]:
    """全評価結果を読み込み."""
    all_responses = []

    if not response_dir.exists():
        print(f"Directory not found: {response_dir}", file=sys.stderr)
        return []

    for filepath in response_dir.glob("*.jsonl"):
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        all_responses.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error parsing {filepath}: {e}", file=sys.stderr)

    return all_responses


def load_samples_csv(csv_path: Path) -> dict[int, dict]:
    """samples.csvからメタデータをロード."""
    samples = {}

    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        return {}

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sample_id = int(row["ID"])
                samples[sample_id] = {
                    "input_text": row["入力文"],
                    "gold_emojis": row["教師出力（Gold）"],
                }
            except (KeyError, ValueError) as e:
                print(f"Error parsing CSV row: {e}", file=sys.stderr)

    return samples


def aggregate_gold_scores(responses: list[dict]) -> dict[int, dict]:
    """sample_idごとにGoldスコアを集計."""
    scores_by_sample: dict[int, dict[str, list]] = defaultdict(
        lambda: {"semantic": [], "naturalness": []}
    )

    for r in responses:
        sample_id = r.get("sample_id")
        gold = r.get("gold", {})

        if sample_id is not None and gold:
            if gold.get("semantic") is not None:
                scores_by_sample[sample_id]["semantic"].append(gold["semantic"])
            if gold.get("naturalness") is not None:
                scores_by_sample[sample_id]["naturalness"].append(gold["naturalness"])

    aggregated = {}
    for sample_id, scores in scores_by_sample.items():
        sem_values = scores["semantic"]
        nat_values = scores["naturalness"]

        if sem_values and nat_values:
            avg_semantic = float(np.mean(sem_values))
            avg_naturalness = float(np.mean(nat_values))
            total_score = avg_semantic + avg_naturalness

            aggregated[sample_id] = {
                "avg_semantic": avg_semantic,
                "avg_naturalness": avg_naturalness,
                "total_score": total_score,
                "n_evaluators": len(sem_values),
            }

            # 評価者数が5未満なら警告
            if len(sem_values) < 5:
                print(
                    f"Warning: Sample {sample_id} has only {len(sem_values)} evaluators",
                    file=sys.stderr,
                )

    return aggregated


def extract_top_samples(
    aggregated: dict[int, dict], metadata: dict[int, dict], top_n: int
) -> list[dict]:
    """上位N件を抽出してマージ."""
    sorted_samples = sorted(
        aggregated.items(), key=lambda x: x[1]["total_score"], reverse=True
    )

    top_samples = []
    for rank, (sample_id, scores) in enumerate(sorted_samples[:top_n], start=1):
        meta = metadata.get(sample_id, {"input_text": "N/A", "gold_emojis": "N/A"})
        top_samples.append(
            {
                "rank": rank,
                "sample_id": sample_id,
                "total_score": round(scores["total_score"], 3),
                "avg_semantic": round(scores["avg_semantic"], 3),
                "avg_naturalness": round(scores["avg_naturalness"], 3),
                "n_evaluators": scores["n_evaluators"],
                "input_text": meta["input_text"],
                "gold_emojis": meta["gold_emojis"],
            }
        )

    return top_samples


def format_json(samples: list[dict]) -> str:
    """JSON形式で出力."""
    return json.dumps(samples, ensure_ascii=False, indent=2)


def format_markdown(samples: list[dict]) -> str:
    """Markdown形式で出力."""
    lines = ["# Goldスコア上位サンプル", ""]

    for s in samples:
        lines.extend(
            [
                f"## {s['rank']}位: Sample {s['sample_id']} (総合スコア: {s['total_score']})",
                f"- **入力文**: {s['input_text']}",
                f"- **Gold絵文字**: {s['gold_emojis']}",
                f"- **意味的一致度**: {s['avg_semantic']}/4.0",
                f"- **自然さ**: {s['avg_naturalness']}/4.0",
                f"- **評価者数**: {s['n_evaluators']}人",
                "",
            ]
        )

    return "\n".join(lines)


def format_csv(samples: list[dict]) -> str:
    """CSV形式で出力."""
    import io

    output = io.StringIO()
    if samples:
        writer = csv.DictWriter(output, fieldnames=samples[0].keys())
        writer.writeheader()
        writer.writerows(samples)

    return output.getvalue()


def main() -> None:
    """メイン処理."""
    parser = argparse.ArgumentParser(description="Goldスコア上位サンプルを抽出")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/llm_multi_eval"),
        help="評価結果ディレクトリ（デフォルト: outputs/llm_multi_eval）",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="上位N件を抽出（デフォルト: 5）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="出力ファイルのベースパス（拡張子は自動付与、デフォルト: {input_dir}/top_gold_samples）",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="json,markdown",
        help="出力形式（json/markdown/csv、カンマ区切りで複数指定可、デフォルト: json,markdown）",
    )
    args = parser.parse_args()

    # パス設定
    response_dir = args.input_dir / "responses"
    samples_csv = args.input_dir / "samples.csv"

    if args.output is None:
        output_base = args.input_dir / "top_gold_samples"
    else:
        output_base = args.output

    # データロード
    print(f"Loading responses from {response_dir}...", file=sys.stderr)
    responses = load_all_responses(response_dir)

    if not responses:
        print("No responses found. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(responses)} responses", file=sys.stderr)

    print(f"Loading samples metadata from {samples_csv}...", file=sys.stderr)
    metadata = load_samples_csv(samples_csv)

    if not metadata:
        print("No metadata found. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded metadata for {len(metadata)} samples", file=sys.stderr)

    # スコア集計
    print("Aggregating Gold scores...", file=sys.stderr)
    aggregated = aggregate_gold_scores(responses)

    if not aggregated:
        print("No Gold scores found. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Aggregated scores for {len(aggregated)} samples", file=sys.stderr)

    # 上位抽出
    print(f"Extracting top {args.top_n} samples...", file=sys.stderr)
    top_samples = extract_top_samples(aggregated, metadata, args.top_n)

    # 出力形式処理
    formats = [f.strip().lower() for f in args.format.split(",")]

    for fmt in formats:
        if fmt == "json":
            output_path = output_base.with_suffix(".json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(format_json(top_samples))
            print(f"Saved JSON to {output_path}")

        elif fmt == "markdown":
            output_path = output_base.with_suffix(".md")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(format_markdown(top_samples))
            print(f"Saved Markdown to {output_path}")

        elif fmt == "csv":
            output_path = output_base.with_suffix(".csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8-sig") as f:  # BOM for Excel
                f.write(format_csv(top_samples))
            print(f"Saved CSV to {output_path}")

        else:
            print(f"Unknown format: {fmt}", file=sys.stderr)

    # コンソール出力
    print("\n" + format_markdown(top_samples))


if __name__ == "__main__":
    main()
