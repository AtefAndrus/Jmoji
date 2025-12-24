#!/usr/bin/env python3
"""人手評価用サンプルを準備するスクリプト.

既存の予測サンプルをマージ、またはHuggingFace Hubからモデルをロードして推論を実行し、
人手評価用フォーマットに変換する。

使用方法（既存の予測ファイルを使用）:
    uv run scripts/prepare_human_eval.py

使用方法（HuggingFace Hubから推論）:
    uv run scripts/prepare_human_eval.py \
        --model-a-repo AtefAndrus/jmoji-t5-v4_focal_top50_20251224 \
        --model-b-repo AtefAndrus/jmoji-t5-v4_top50_20251224 \
        --input-file data/test.jsonl \
        --max-samples 50

出力:
    outputs/human_eval/samples.jsonl  - 評価サンプル（JSONL形式）
    outputs/human_eval/samples.csv    - Googleフォーム用（CSV形式）
    outputs/human_eval/samples.md     - 確認用（Markdown形式）

環境変数:
    HF_TOKEN: HuggingFace Hub認証トークン（プライベートリポジトリ用）
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from src.evaluation.metrics import jaccard_similarity
from src.models.t5_trainer import generate_emoji, load_model_from_hub


def load_predictions(path: Path) -> list[dict]:
    """予測サンプルを読み込む."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def load_test_samples(path: Path) -> list[dict]:
    """テストセットを読み込む."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                samples.append(
                    {
                        "text": data["sns_text"],
                        "gold": data["emoji_string"],
                    }
                )
    return samples


def run_inference_for_samples(
    repo_id: str,
    samples: list[dict],
    device: str | None = None,
) -> list[dict]:
    """モデルをロードしてサンプルに対して推論を実行する."""
    print(f"Loading model from {repo_id}...", file=sys.stderr)
    try:
        tokenizer, model = load_model_from_hub(repo_id=repo_id, device=device)
    except OSError as e:
        if "401" in str(e) or "403" in str(e):
            print(
                "Error: 認証エラー。環境変数 HF_TOKEN を設定してください。",
                file=sys.stderr,
            )
            sys.exit(1)
        raise

    device_str = str(next(model.parameters()).device)
    print(f"Model loaded on {device_str}", file=sys.stderr)

    results = []
    for i, sample in enumerate(samples):
        pred = generate_emoji(
            model,
            tokenizer,
            sample["text"],
            use_sampling=True,
            device=device_str,
        )

        gold_set = set(sample["gold"].split())
        pred_set = set(pred.split())
        jacc = jaccard_similarity(pred_set, gold_set)

        results.append(
            {
                "text": sample["text"],
                "gold": sample["gold"],
                "pred": pred,
                "jaccard": jacc,
            }
        )

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(samples)} samples...", file=sys.stderr)

    print(f"Inference completed for {len(results)} samples", file=sys.stderr)
    return results


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


def merge_inference_results(
    results_a: list[dict],
    results_b: list[dict],
    model_a_name: str,
    model_b_name: str,
) -> list[dict]:
    """2つのモデルの推論結果をマージする."""
    if len(results_a) != len(results_b):
        raise ValueError(f"Sample count mismatch: {len(results_a)} vs {len(results_b)}")

    merged = []
    for i, (a, b) in enumerate(zip(results_a, results_b)):
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


def save_csv(
    samples: list[dict], path: Path, model_a_name: str, model_b_name: str
) -> None:
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
                "モデルA出力": s[f"pred_{model_a_name}"],
                "モデルB出力": s[f"pred_{model_b_name}"],
            }
        )

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV to {path}")


def save_markdown(
    samples: list[dict], path: Path, model_a_name: str, model_b_name: str
) -> None:
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
                f"**モデルA（{model_a_name}）**: {s[f'pred_{model_a_name}']} "
                f"(Jaccard: {s[f'jaccard_{model_a_name}']:.3f})",
                "",
                f"**モデルB（{model_b_name}）**: {s[f'pred_{model_b_name}']} "
                f"(Jaccard: {s[f'jaccard_{model_b_name}']:.3f})",
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
    parser = argparse.ArgumentParser(
        description="人手評価用サンプルを準備する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # 既存の予測ファイルを使用するオプション
    parser.add_argument(
        "--model-a",
        type=Path,
        default=None,
        help="モデルAの予測ファイル（既存の predictions_sample.jsonl）",
    )
    parser.add_argument(
        "--model-b",
        type=Path,
        default=None,
        help="モデルBの予測ファイル（既存の predictions_sample.jsonl）",
    )

    # HuggingFace Hubから推論するオプション
    parser.add_argument(
        "--model-a-repo",
        type=str,
        default=None,
        help="モデルAのHuggingFace Hubリポジトリ ID",
    )
    parser.add_argument(
        "--model-b-repo",
        type=str,
        default=None,
        help="モデルBのHuggingFace Hubリポジトリ ID",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="入力ファイル（JSONL形式、sns_textとemoji_stringを含む）",
    )

    # 共通オプション
    parser.add_argument(
        "--model-a-name",
        type=str,
        default="focal_top50",
        help="モデルAの名前（出力に使用）",
    )
    parser.add_argument(
        "--model-b-name",
        type=str,
        default="top50",
        help="モデルBの名前（出力に使用）",
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
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="デバイス（cuda/cpu、デフォルト: 自動検出）",
    )
    args = parser.parse_args()

    # モード判定
    use_hub = args.model_a_repo is not None or args.model_b_repo is not None

    if use_hub:
        # HuggingFace Hubから推論モード
        if not args.model_a_repo or not args.model_b_repo:
            parser.error("--model-a-repo と --model-b-repo の両方を指定してください")
        if not args.input_file:
            parser.error("--input-file を指定してください")
        if not args.input_file.exists():
            parser.error(f"入力ファイルが見つかりません: {args.input_file}")

        print("=== HuggingFace Hub推論モード ===")
        print(f"Model A: {args.model_a_repo}")
        print(f"Model B: {args.model_b_repo}")
        print(f"Input: {args.input_file}")

        # テストセットを読み込み
        test_samples = load_test_samples(args.input_file)
        print(f"Loaded {len(test_samples)} samples from {args.input_file}")

        # サンプリング（推論前に行う）
        if args.max_samples and args.max_samples < len(test_samples):
            random.seed(args.seed)
            test_samples = random.sample(test_samples, args.max_samples)
            print(f"Sampled {len(test_samples)} samples (seed={args.seed})")

        # 両モデルで推論を実行
        print("\n--- Model A 推論 ---")
        results_a = run_inference_for_samples(
            args.model_a_repo, test_samples, args.device
        )

        print("\n--- Model B 推論 ---")
        results_b = run_inference_for_samples(
            args.model_b_repo, test_samples, args.device
        )

        # 結果をマージ
        samples = merge_inference_results(
            results_a, results_b, args.model_a_name, args.model_b_name
        )
    else:
        # 既存の予測ファイルを使用するモード（互換性維持）
        model_a_path = args.model_a or Path(
            "outputs/experiments/v4_focal_top50_20251224/predictions_sample.jsonl"
        )
        model_b_path = args.model_b or Path(
            "outputs/experiments/v4_top50_20251224/predictions_sample.jsonl"
        )

        print("=== 既存予測ファイルモード ===")
        print(f"Model A ({args.model_a_name}): {model_a_path}")
        print(f"Model B ({args.model_b_name}): {model_b_path}")

        samples = merge_predictions(
            model_a_path, model_b_path, args.model_a_name, args.model_b_name
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
    save_csv(
        samples, args.output_dir / "samples.csv", args.model_a_name, args.model_b_name
    )
    save_markdown(
        samples, args.output_dir / "samples.md", args.model_a_name, args.model_b_name
    )

    # 統計情報を表示
    print("\n=== 統計情報 ===")
    jaccard_a = [s[f"jaccard_{args.model_a_name}"] for s in samples]
    jaccard_b = [s[f"jaccard_{args.model_b_name}"] for s in samples]
    print(
        f"Model A ({args.model_a_name}) 平均Jaccard: {sum(jaccard_a) / len(jaccard_a):.3f}"
    )
    print(
        f"Model B ({args.model_b_name}) 平均Jaccard: {sum(jaccard_b) / len(jaccard_b):.3f}"
    )

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
