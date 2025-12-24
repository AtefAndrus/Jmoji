#!/usr/bin/env python3
"""人手評価用サンプルを準備するスクリプト.

既存の予測サンプルをマージして、人手評価用フォーマットに変換する。

使用方法:
    uv run scripts/prepare_human_eval.py

出力:
    outputs/human_eval/samples.jsonl  - 評価サンプル（JSONL形式）
    outputs/human_eval/samples.csv    - Googleフォーム用（CSV形式）
    outputs/human_eval/samples.md     - 確認用（Markdown形式）
"""

import argparse
import csv
import json
import random
from pathlib import Path


def load_predictions(path: Path) -> list[dict]:
    """予測サンプルを読み込む."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def merge_predictions(
    model_a_path: Path,
    model_b_path: Path,
    model_a_name: str,
    model_b_name: str,
) -> list[dict]:
    """2つのモデルの予測をマージする."""
    samples_a = load_predictions(model_a_path)
    samples_b = load_predictions(model_b_path)

    if len(samples_a) != len(samples_b):
        raise ValueError(f"Sample count mismatch: {len(samples_a)} vs {len(samples_b)}")

    merged = []
    for i, (a, b) in enumerate(zip(samples_a, samples_b)):
        # 入力文が一致することを確認
        if a["text"] != b["text"]:
            raise ValueError(f"Text mismatch at index {i}")

        merged.append(
            {
                "id": i + 1,
                "text": a["text"],
                "gold": a["gold"],
                f"pred_{model_a_name}": a["pred"],
                f"pred_{model_b_name}": b["pred"],
                f"jaccard_{model_a_name}": a["jaccard"],
                f"jaccard_{model_b_name}": b["jaccard"],
            }
        )

    return merged


def save_jsonl(samples: list[dict], path: Path) -> None:
    """JSONL形式で保存."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved {len(samples)} samples to {path}")


def save_csv(samples: list[dict], path: Path) -> None:
    """CSV形式で保存（Googleフォーム用）."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # フォーム用に整形
    rows = []
    for s in samples:
        rows.append(
            {
                "ID": s["id"],
                "入力文": s["text"],
                "教師出力（Gold）": s["gold"],
                "モデルA出力": s["pred_focal_top50"],
                "モデルB出力": s["pred_top50"],
            }
        )

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV to {path}")


def save_markdown(samples: list[dict], path: Path) -> None:
    """Markdown形式で保存（確認用）."""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# 人手評価サンプル",
        "",
        f"サンプル数: {len(samples)}件",
        "",
        "---",
        "",
    ]

    for s in samples:
        lines.extend(
            [
                f"## サンプル #{s['id']}",
                "",
                f"**入力文**: {s['text']}",
                "",
                f"**教師出力（Gold）**: {s['gold']}",
                "",
                f"**モデルA（focal_top50）**: {s['pred_focal_top50']} "
                f"(Jaccard: {s['jaccard_focal_top50']:.3f})",
                "",
                f"**モデルB（top50）**: {s['pred_top50']} "
                f"(Jaccard: {s['jaccard_top50']:.3f})",
                "",
                "---",
                "",
            ]
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved Markdown to {path}")


def main() -> None:
    """メイン処理."""
    parser = argparse.ArgumentParser(description="人手評価用サンプルを準備する")
    parser.add_argument(
        "--model-a",
        type=Path,
        default=Path(
            "outputs/experiments/v4_focal_top50_20251224/predictions_sample.jsonl"
        ),
        help="モデルAの予測ファイル（精度重視）",
    )
    parser.add_argument(
        "--model-b",
        type=Path,
        default=Path("outputs/experiments/v4_top50_20251224/predictions_sample.jsonl"),
        help="モデルBの予測ファイル（バランス型）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/human_eval"),
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大サンプル数（指定しない場合は全件）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="ランダムシード（サンプリング用）",
    )
    args = parser.parse_args()

    # 予測をマージ
    print("Loading predictions from:")
    print(f"  Model A (focal_top50): {args.model_a}")
    print(f"  Model B (top50): {args.model_b}")

    samples = merge_predictions(
        args.model_a,
        args.model_b,
        model_a_name="focal_top50",
        model_b_name="top50",
    )
    print(f"Merged {len(samples)} samples")

    # サンプリング（指定された場合）
    if args.max_samples and args.max_samples < len(samples):
        random.seed(args.seed)
        samples = random.sample(samples, args.max_samples)
        # IDを振り直し
        for i, s in enumerate(samples):
            s["id"] = i + 1
        print(f"Sampled {len(samples)} samples (seed={args.seed})")

    # 保存
    save_jsonl(samples, args.output_dir / "samples.jsonl")
    save_csv(samples, args.output_dir / "samples.csv")
    save_markdown(samples, args.output_dir / "samples.md")

    # 統計情報を表示
    print("\n=== 統計情報 ===")
    jaccard_a = [s["jaccard_focal_top50"] for s in samples]
    jaccard_b = [s["jaccard_top50"] for s in samples]
    print(f"Model A (focal_top50) 平均Jaccard: {sum(jaccard_a)/len(jaccard_a):.3f}")
    print(f"Model B (top50) 平均Jaccard: {sum(jaccard_b)/len(jaccard_b):.3f}")

    # モデル比較
    a_wins = sum(1 for a, b in zip(jaccard_a, jaccard_b) if a > b)
    b_wins = sum(1 for a, b in zip(jaccard_a, jaccard_b) if b > a)
    ties = sum(1 for a, b in zip(jaccard_a, jaccard_b) if a == b)
    print("\nJaccard比較:")
    print(f"  Model A wins: {a_wins}")
    print(f"  Model B wins: {b_wins}")
    print(f"  Ties: {ties}")


if __name__ == "__main__":
    main()
