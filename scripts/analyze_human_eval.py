#!/usr/bin/env python3
"""人手評価結果を集計・分析するスクリプト.

使用方法:
    # HuggingFace Hubから結果をダウンロードして集計
    uv run scripts/analyze_human_eval.py \
        --space-id AtefAndrus/jmoji-human-eval \
        --output outputs/human_eval/results.json

    # ローカルファイルを集計
    uv run scripts/analyze_human_eval.py \
        --local-dir /home/keigo/jmoji-human-eval/responses \
        --output outputs/human_eval/results.json

    # Markdownレポートも出力
    uv run scripts/analyze_human_eval.py \
        --local-dir /home/keigo/jmoji-human-eval/responses \
        --output outputs/human_eval/results.json \
        --report outputs/human_eval/report.md
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
from sklearn.metrics import cohen_kappa_score


def download_responses(space_id: str, local_dir: Path) -> list[Path]:
    """HuggingFace Spaceから評価結果をダウンロード."""
    local_dir.mkdir(parents=True, exist_ok=True)

    # responsesディレクトリ内のファイル一覧を取得
    try:
        files = list_repo_files(space_id, repo_type="space")
    except Exception as e:
        print(f"Error listing files from {space_id}: {e}", file=sys.stderr)
        return []

    response_files = [
        f for f in files if f.startswith("responses/") and f.endswith(".jsonl")
    ]

    if not response_files:
        print("No response files found in the Space.", file=sys.stderr)
        return []

    downloaded = []
    for filepath in response_files:
        try:
            local_path = hf_hub_download(
                repo_id=space_id,
                repo_type="space",
                filename=filepath,
                local_dir=local_dir.parent,
            )
            downloaded.append(Path(local_path))
            print(f"Downloaded: {filepath}", file=sys.stderr)
        except Exception as e:
            print(f"Error downloading {filepath}: {e}", file=sys.stderr)

    return downloaded


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


def compute_average_scores(responses: list[dict]) -> dict[str, dict[str, float]]:
    """モデルごとの平均スコアを計算."""
    scores: dict[str, dict[str, list]] = {
        "gold": {"semantic": [], "naturalness": [], "misleading": []},
        "model_a": {"semantic": [], "naturalness": [], "misleading": []},
        "model_b": {"semantic": [], "naturalness": [], "misleading": []},
    }

    for r in responses:
        for model in ["gold", "model_a", "model_b"]:
            model_data = r.get(model, {})
            if model_data:
                if model_data.get("semantic") is not None:
                    scores[model]["semantic"].append(model_data["semantic"])
                if model_data.get("naturalness") is not None:
                    scores[model]["naturalness"].append(model_data["naturalness"])
                if model_data.get("misleading") is not None:
                    scores[model]["misleading"].append(
                        1 if model_data["misleading"] else 0
                    )

    results: dict[str, dict[str, float]] = {}
    for model, metrics in scores.items():
        results[model] = {}
        for metric, values in metrics.items():
            if values:
                results[model][f"{metric}_mean"] = float(np.mean(values))
                results[model][f"{metric}_std"] = float(np.std(values))
                results[model][f"{metric}_n"] = len(values)

    return results


def compute_preference_distribution(responses: list[dict]) -> dict[str, int]:
    """モデル選好の分布を計算."""
    counts: dict[str, int] = defaultdict(int)

    for r in responses:
        pref = r.get("preference")
        if pref:
            # 正規化
            if "A" in pref:
                counts["model_a"] += 1
            elif "B" in pref:
                counts["model_b"] += 1
            elif "同等" in pref or "equal" in pref.lower():
                counts["equal"] += 1

    return dict(counts)


def compute_cohen_kappa(
    responses: list[dict], min_evaluators: int = 2
) -> dict[str, float | None]:
    """評価者間一致度（Cohen's kappa）を計算.

    同一サンプルを評価した評価者ペアごとに計算し、平均を返す。
    """
    # サンプルIDごとに評価者の回答をグループ化
    by_sample: dict[int, dict[str, dict]] = defaultdict(dict)
    for r in responses:
        sample_id = r.get("sample_id")
        evaluator_id = r.get("evaluator_id")
        if sample_id is not None and evaluator_id:
            by_sample[sample_id][evaluator_id] = r

    # 各指標のkappa値を格納
    kappas: dict[str, list[float]] = {
        "semantic_gold": [],
        "semantic_model_a": [],
        "semantic_model_b": [],
        "naturalness_gold": [],
        "naturalness_model_a": [],
        "naturalness_model_b": [],
        "preference": [],
    }

    # 2人以上が評価したサンプルでkappaを計算
    for sample_id, evaluators in by_sample.items():
        if len(evaluators) < min_evaluators:
            continue

        evaluator_ids = list(evaluators.keys())

        # ペアごとに計算（複数サンプルを蓄積してから計算）
        for i in range(len(evaluator_ids)):
            for j in range(i + 1, len(evaluator_ids)):
                r1 = evaluators[evaluator_ids[i]]
                r2 = evaluators[evaluator_ids[j]]

                # 各指標でkappa計算用のデータを蓄積
                for model in ["gold", "model_a", "model_b"]:
                    for metric in ["semantic", "naturalness"]:
                        v1 = r1.get(model, {}).get(metric)
                        v2 = r2.get(model, {}).get(metric)
                        if v1 is not None and v2 is not None:
                            key = f"{metric}_{model}"
                            # 単一値ではkappa計算不可なのでスキップ
                            # 複数サンプルの蓄積が必要
                            pass

    # 評価者ペアごとに全サンプルのkappa計算
    evaluator_pairs: dict[tuple[str, str], dict[str, dict[str, list]]] = defaultdict(
        lambda: {k: {"r1": [], "r2": []} for k in kappas}
    )

    for sample_id, evaluators in by_sample.items():
        evaluator_ids = sorted(evaluators.keys())
        for i in range(len(evaluator_ids)):
            for j in range(i + 1, len(evaluator_ids)):
                pair = (evaluator_ids[i], evaluator_ids[j])
                r1 = evaluators[evaluator_ids[i]]
                r2 = evaluators[evaluator_ids[j]]

                for model in ["gold", "model_a", "model_b"]:
                    for metric in ["semantic", "naturalness"]:
                        v1 = r1.get(model, {}).get(metric)
                        v2 = r2.get(model, {}).get(metric)
                        if v1 is not None and v2 is not None:
                            key = f"{metric}_{model}"
                            evaluator_pairs[pair][key]["r1"].append(v1)
                            evaluator_pairs[pair][key]["r2"].append(v2)

                # preference
                p1 = r1.get("preference")
                p2 = r2.get("preference")
                if p1 and p2:

                    def normalize_pref(p: str) -> str:
                        if "A" in p:
                            return "A"
                        elif "B" in p:
                            return "B"
                        else:
                            return "equal"

                    evaluator_pairs[pair]["preference"]["r1"].append(normalize_pref(p1))
                    evaluator_pairs[pair]["preference"]["r2"].append(normalize_pref(p2))

    # 各ペアでkappa計算
    for pair, metrics_data in evaluator_pairs.items():
        for key, data in metrics_data.items():
            if len(data["r1"]) >= 2:  # 最低2サンプル必要
                try:
                    kappa = cohen_kappa_score(data["r1"], data["r2"])
                    kappas[key].append(kappa)
                except ValueError:
                    pass  # 計算不可

    # 平均を計算
    results: dict[str, float | None] = {}
    for key, values in kappas.items():
        if values:
            results[key] = float(np.mean(values))
        else:
            results[key] = None

    return results


def generate_report(
    avg_scores: dict,
    preference: dict,
    kappas: dict,
    total_responses: int,
    total_evaluators: int,
) -> str:
    """Markdownレポートを生成."""
    lines = [
        "# 人手評価結果レポート",
        "",
        f"- 総評価数: {total_responses}",
        f"- 評価者数: {total_evaluators}",
        "",
        "## 1. 平均スコア",
        "",
        "| モデル | 意味的一致度 | 自然さ | 誤解率 |",
        "|--------|-------------|--------|--------|",
    ]

    for model, label in [
        ("gold", "教師（Gold）"),
        ("model_a", "モデルA（focal_top50）"),
        ("model_b", "モデルB（top50）"),
    ]:
        sem = avg_scores.get(model, {})
        sem_str = (
            f"{sem.get('semantic_mean', 0):.2f} ± {sem.get('semantic_std', 0):.2f}"
            if sem.get("semantic_mean") is not None
            else "N/A"
        )
        nat_str = (
            f"{sem.get('naturalness_mean', 0):.2f} ± {sem.get('naturalness_std', 0):.2f}"
            if sem.get("naturalness_mean") is not None
            else "N/A"
        )
        mis_str = (
            f"{sem.get('misleading_mean', 0) * 100:.1f}%"
            if sem.get("misleading_mean") is not None
            else "N/A"
        )
        lines.append(f"| {label} | {sem_str} | {nat_str} | {mis_str} |")

    total_pref = sum(preference.values())
    lines.extend(
        [
            "",
            "## 2. モデル選好",
            "",
            f"- モデルA（focal_top50）: {preference.get('model_a', 0)}票 "
            f"({preference.get('model_a', 0) / total_pref * 100:.1f}%)"
            if total_pref > 0
            else "",
            f"- モデルB（top50）: {preference.get('model_b', 0)}票 "
            f"({preference.get('model_b', 0) / total_pref * 100:.1f}%)"
            if total_pref > 0
            else "",
            f"- 同等: {preference.get('equal', 0)}票 "
            f"({preference.get('equal', 0) / total_pref * 100:.1f}%)"
            if total_pref > 0
            else "",
            "",
            "## 3. 評価者間一致度（Cohen's kappa）",
            "",
        ]
    )

    kappa_labels = {
        "semantic_gold": "意味的一致度（Gold）",
        "semantic_model_a": "意味的一致度（モデルA）",
        "semantic_model_b": "意味的一致度（モデルB）",
        "naturalness_gold": "自然さ（Gold）",
        "naturalness_model_a": "自然さ（モデルA）",
        "naturalness_model_b": "自然さ（モデルB）",
        "preference": "モデル選好",
    }

    for key, label in kappa_labels.items():
        value = kappas.get(key)
        if value is not None:
            # kappa値の解釈
            if value < 0.20:
                interpretation = "わずか"
            elif value < 0.40:
                interpretation = "弱い"
            elif value < 0.60:
                interpretation = "中程度"
            elif value < 0.80:
                interpretation = "実質的"
            else:
                interpretation = "ほぼ完全"
            lines.append(f"- {label}: {value:.3f}（{interpretation}な一致）")
        else:
            lines.append(f"- {label}: N/A（データ不足）")

    lines.extend(
        [
            "",
            "---",
            "",
            "*このレポートは `scripts/analyze_human_eval.py` により自動生成されました。*",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """メイン処理."""
    parser = argparse.ArgumentParser(description="人手評価結果を集計する")
    parser.add_argument(
        "--space-id",
        type=str,
        default=None,
        help="HuggingFace SpaceのID（例: AtefAndrus/jmoji-human-eval）",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="ローカルのresponsesディレクトリ",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/human_eval/results.json"),
        help="出力ファイル（JSON）",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Markdownレポート出力先（指定しない場合は標準出力）",
    )
    args = parser.parse_args()

    # ダウンロードまたはローカル読み込み
    if args.space_id:
        local_dir = Path("outputs/human_eval/responses")
        download_responses(args.space_id, local_dir)
        responses = load_all_responses(local_dir)
    elif args.local_dir:
        responses = load_all_responses(args.local_dir)
    else:
        parser.error("--space-id または --local-dir を指定してください")

    if not responses:
        print("No responses found. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(responses)} responses", file=sys.stderr)

    # 集計
    avg_scores = compute_average_scores(responses)
    preference = compute_preference_distribution(responses)
    kappas = compute_cohen_kappa(responses)

    # 評価者数
    evaluators = {r.get("evaluator_id") for r in responses if r.get("evaluator_id")}

    # 結果を保存
    results = {
        "total_responses": len(responses),
        "total_evaluators": len(evaluators),
        "average_scores": avg_scores,
        "preference_distribution": preference,
        "cohen_kappa": kappas,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {args.output}")

    # レポート生成
    report = generate_report(
        avg_scores, preference, kappas, len(responses), len(evaluators)
    )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Saved report to {args.report}")
    else:
        print("\n" + report)


if __name__ == "__main__":
    main()
