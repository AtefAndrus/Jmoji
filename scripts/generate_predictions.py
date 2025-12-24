#!/usr/bin/env python3
"""HuggingFace Hubからモデルをロードして絵文字予測を行うスクリプト.

使用例:
    # テキストファイルから推論
    uv run scripts/generate_predictions.py \
        --model AtefAndrus/jmoji-t5-v4_focal_top50_20251224 \
        --input texts.txt \
        --output predictions.jsonl

    # 標準入力から
    echo "今日は楽しかった" | uv run scripts/generate_predictions.py \
        --model AtefAndrus/jmoji-t5-v4_focal_top50_20251224

    # JSONLファイル入力
    uv run scripts/generate_predictions.py \
        --model AtefAndrus/jmoji-t5-v4_focal_top50_20251224 \
        --input data/test.jsonl \
        --input-format jsonl \
        --text-key sns_text

環境変数:
    HF_TOKEN: HuggingFace Hub認証トークン（プライベートリポジトリ用）
"""

import argparse
import json
import sys
from pathlib import Path
from typing import IO, Iterator

from src.models.t5_trainer import generate_emoji, load_model_from_hub


def read_texts_from_file(
    path: Path,
    input_format: str,
    text_key: str,
) -> Iterator[dict]:
    """ファイルからテキストを読み込む.

    Args:
        path: 入力ファイルパス
        input_format: 入力形式（"text" または "jsonl"）
        text_key: JSONLの場合のテキストキー名

    Yields:
        dict: {"text": str, "gold": Optional[str]} の形式
    """
    with open(path, encoding="utf-8") as f:
        if input_format == "jsonl":
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    yield {
                        "text": data[text_key],
                        "gold": data.get("emoji_string"),
                    }
        else:  # text format
            for line in f:
                text = line.strip()
                if text:
                    yield {"text": text, "gold": None}


def read_texts_from_stdin(input_format: str, text_key: str) -> Iterator[dict]:
    """標準入力からテキストを読み込む."""
    if input_format == "jsonl":
        for line in sys.stdin:
            if line.strip():
                data = json.loads(line)
                yield {
                    "text": data[text_key],
                    "gold": data.get("emoji_string"),
                }
    else:  # text format
        for line in sys.stdin:
            text = line.strip()
            if text:
                yield {"text": text, "gold": None}


def main() -> None:
    """メイン処理."""
    parser = argparse.ArgumentParser(
        description="HuggingFace Hubからモデルをロードして絵文字予測を行う",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace Hub リポジトリ ID（例: AtefAndrus/jmoji-t5-v4_focal_top50_20251224）",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="入力ファイルパス（省略時は標準入力）",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        choices=["text", "jsonl"],
        default="text",
        help="入力形式（デフォルト: text）",
    )
    parser.add_argument(
        "--text-key",
        type=str,
        default="sns_text",
        help="JSONLの場合のテキストキー名（デフォルト: sns_text）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="出力ファイルパス（省略時は標準出力）",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["jsonl", "text"],
        default="jsonl",
        help="出力形式（デフォルト: jsonl）",
    )
    parser.add_argument(
        "--use-sampling",
        action="store_true",
        default=True,
        help="サンプリングを使用（デフォルト: True）",
    )
    parser.add_argument(
        "--no-sampling",
        action="store_true",
        help="Beam Searchを使用（サンプリングを無効化）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="サンプリング温度（デフォルト: 1.0）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="デバイス（cuda/cpu、デフォルト: 自動検出）",
    )
    args = parser.parse_args()

    # サンプリング設定
    use_sampling = not args.no_sampling

    # モデルロード
    print(f"Loading model from {args.model}...", file=sys.stderr)
    try:
        tokenizer, model = load_model_from_hub(
            repo_id=args.model,
            device=args.device,
        )
    except OSError as e:
        if "401" in str(e) or "403" in str(e):
            print(
                "Error: 認証エラー。環境変数 HF_TOKEN を設定してください。",
                file=sys.stderr,
            )
            print("  export HF_TOKEN='hf_...'", file=sys.stderr)
        else:
            print(f"Error: モデルのロードに失敗しました: {e}", file=sys.stderr)
        sys.exit(1)

    device = next(model.parameters()).device
    print(f"Model loaded on {device}", file=sys.stderr)

    # 入力読み込み
    if args.input:
        if not args.input.exists():
            print(f"Error: 入力ファイルが見つかりません: {args.input}", file=sys.stderr)
            sys.exit(1)
        texts = read_texts_from_file(args.input, args.input_format, args.text_key)
    else:
        texts = read_texts_from_stdin(args.input_format, args.text_key)

    # 出力先
    out_file: IO[str]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_file = open(args.output, "w", encoding="utf-8")
    else:
        out_file = sys.stdout

    # 推論実行
    try:
        count = 0
        for item in texts:
            text = item["text"]
            pred = generate_emoji(
                model,
                tokenizer,
                text,
                use_sampling=use_sampling,
                temperature=args.temperature,
                device=str(device),
            )

            if args.output_format == "jsonl":
                result = {"text": text, "pred": pred}
                if item["gold"]:
                    result["gold"] = item["gold"]
                out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            else:  # text format
                out_file.write(pred + "\n")

            count += 1
            if count % 10 == 0:
                print(f"Processed {count} samples...", file=sys.stderr)

        print(f"Done. Processed {count} samples.", file=sys.stderr)
    finally:
        if args.output:
            out_file.close()


if __name__ == "__main__":
    main()
