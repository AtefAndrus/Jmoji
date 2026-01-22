#!/usr/bin/env python3
"""Claude Code subagentsを使った多人数LLM評価スクリプト.

5つの異なるペルソナでモデル出力を評価し、人手評価との比較を可能にする。

使用方法:
    # ステップ1: サンプル準備
    uv run scripts/prepare_human_eval.py \\
        --model-a-repo AtefAndrus/jmoji-t5-v4_focal_top50_20251224 \\
        --model-b-repo AtefAndrus/jmoji-t5-v4_top50_20251224 \\
        --input-file data/outputs/dataset_v4.jsonl \\
        --max-samples 50 \\
        --output-dir outputs/llm_multi_eval

    # ステップ2: プロンプト生成
    uv run scripts/llm_multi_eval.py prepare \\
        --samples outputs/llm_multi_eval/samples.jsonl \\
        --output-dir outputs/llm_multi_eval

    # ステップ3: Claude Codeでsubagents実行（手動）
    # 各ペルソナのプロンプトをClaude Codeに渡してsubagentsを並列実行

    # ステップ4: 結果を収集・検証
    uv run scripts/llm_multi_eval.py collect \\
        --output-dir outputs/llm_multi_eval

    # ステップ5: 結果を集計
    uv run scripts/analyze_human_eval.py \\
        --local-dir outputs/llm_multi_eval/responses \\
        --output outputs/llm_multi_eval/results.json \\
        --report outputs/llm_multi_eval/report.md
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from src.evaluation.llm_evaluator import LLMEvaluator, get_personas


def load_samples(samples_path: Path) -> list[dict]:
    """評価サンプルを読み込む.

    Args:
        samples_path: samples.jsonlのパス

    Returns:
        サンプルリスト
    """
    samples = []
    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def prepare_prompts(samples_path: Path, output_dir: Path) -> None:
    """各ペルソナの評価プロンプトを生成.

    Args:
        samples_path: samples.jsonlのパス
        output_dir: 出力ディレクトリ
    """
    # サンプルロード
    samples = load_samples(samples_path)
    print(f"Loaded {len(samples)} samples from {samples_path}", file=sys.stderr)

    # 出力ディレクトリ作成
    prompts_dir = output_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    responses_dir = output_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)

    # 各ペルソナのプロンプトを生成
    personas = get_personas()
    print(f"\nGenerating prompts for {len(personas)} personas:", file=sys.stderr)

    for persona_id, persona_config in personas.items():
        evaluator = LLMEvaluator(persona_config)
        prompt = evaluator.build_prompt(samples)

        # プロンプトをファイルに保存
        prompt_file = prompts_dir / f"{persona_id}.txt"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt)

        print(f"  [{persona_id}] {prompt_file}", file=sys.stderr)

    print(f"\n✅ Prompts generated in {prompts_dir}", file=sys.stderr)
    print(
        "\n次のステップ: Claude Codeで各プロンプトをsubagentsに渡して並列実行してください。",
        file=sys.stderr,
    )
    print(
        "  例: 'Evaluate samples using these 5 persona prompts in parallel'",
        file=sys.stderr,
    )


def collect_results(output_dir: Path, samples_path: Path | None = None) -> None:
    """subagentsの結果を収集・検証・保存.

    Args:
        output_dir: 出力ディレクトリ
        samples_path: samples.jsonlのパス（オプション、検証用）
    """
    prompts_dir = output_dir / "prompts"
    responses_dir = output_dir / "responses"

    # プロンプトディレクトリ内の応答ファイルを検索
    # ユーザーがClaude Codeで実行した結果を各ペルソナのプロンプトファイルと同じ場所に
    # {persona_id}_response.jsonまたは{persona_id}.jsonとして保存することを想定

    personas = get_personas()
    collected = 0
    errors = []

    print(f"Collecting results from {prompts_dir}...", file=sys.stderr)

    for persona_id, persona_config in personas.items():
        # 応答ファイルを検索
        possible_response_files = [
            prompts_dir / f"{persona_id}_response.json",
            prompts_dir / f"{persona_id}.json",
            responses_dir / f"{persona_id}.json",
        ]

        response_file = None
        for candidate in possible_response_files:
            if candidate.exists():
                response_file = candidate
                break

        if not response_file:
            errors.append(f"❌ {persona_id}: Response file not found")
            continue

        # 応答を読み込み
        try:
            with open(response_file, encoding="utf-8") as f:
                response_text = f.read()

            # 評価器で解析
            evaluator = LLMEvaluator(persona_config)
            evaluations = evaluator.parse_response(response_text)

            # タイムスタンプとevaluator_idを付与してJSONLに保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = responses_dir / f"{persona_id}.jsonl"

            with open(output_file, "w", encoding="utf-8") as f:
                for eval_item in evaluations:
                    # evaluator_idとtimestampを追加
                    eval_item_with_meta = {
                        "evaluator_id": persona_id,
                        "timestamp": timestamp,
                        **eval_item,
                    }
                    f.write(json.dumps(eval_item_with_meta, ensure_ascii=False) + "\n")

            print(
                f"  ✅ {persona_id}: {len(evaluations)} evaluations → {output_file}",
                file=sys.stderr,
            )
            collected += 1

        except Exception as e:
            errors.append(f"❌ {persona_id}: {e}")
            continue

    # 結果サマリー
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Collected: {collected}/{len(personas)} personas", file=sys.stderr)

    if errors:
        print("\nErrors:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)

    if collected == len(personas):
        print("\n✅ All results collected successfully!", file=sys.stderr)
        print(
            "\n次のステップ: 結果を集計してください",
            file=sys.stderr,
        )
        print(
            "  uv run scripts/analyze_human_eval.py \\",
            file=sys.stderr,
        )
        print(
            f"      --local-dir {responses_dir} \\",
            file=sys.stderr,
        )
        print(
            f"      --output {output_dir}/results.json \\",
            file=sys.stderr,
        )
        print(
            f"      --report {output_dir}/report.md",
            file=sys.stderr,
        )


def main():
    """メイン関数."""
    parser = argparse.ArgumentParser(
        description="LLM multi-evaluator with Claude Code subagents"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare コマンド
    prepare_parser = subparsers.add_parser(
        "prepare", help="Generate evaluation prompts for each persona"
    )
    prepare_parser.add_argument(
        "--samples",
        type=Path,
        required=True,
        help="Path to samples.jsonl",
    )
    prepare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/llm_multi_eval"),
        help="Output directory (default: outputs/llm_multi_eval)",
    )

    # collect コマンド
    collect_parser = subparsers.add_parser(
        "collect", help="Collect and validate subagent results"
    )
    collect_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/llm_multi_eval"),
        help="Output directory (default: outputs/llm_multi_eval)",
    )
    collect_parser.add_argument(
        "--samples",
        type=Path,
        help="Path to samples.jsonl (optional, for validation)",
    )

    args = parser.parse_args()

    if args.command == "prepare":
        prepare_prompts(args.samples, args.output_dir)
    elif args.command == "collect":
        collect_results(args.output_dir, args.samples)


if __name__ == "__main__":
    main()
