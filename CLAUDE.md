# CLAUDE.md

Jmojiプロジェクト用のClaude Code設定ファイル。

## プロジェクト概要

知識蒸留を用いた日本語テキスト→絵文字翻訳モデルの開発プロジェクト。
Qwen3-235B-A22Bを教師モデルとして疑似対訳データセットを構築し、日本語T5（`sonoisa/t5-base-japanese`）へ知識蒸留を行う。

> **Note**: v1〜v3データセットはClaude Haiku 4.5で生成。v4以降はQwen3-235B-A22Bを使用。
> 移行理由は [teacher_model_migration.md](docs/details/teacher_model_migration.md) を参照。

## 技術スタック

- Python 3.12
- パッケージ管理: uv + mise
- 主要ライブラリ: transformers, torch, datasets, emoji, httpx
- 開発ツール: ruff, mypy, pytest, pre-commit, jupytext

## ディレクトリ構成

```text
Jmoji/
├── configs/          # 設定ファイル（YAML）
│   └── default.yaml  # デフォルト設定
├── data/             # データセット
├── docs/             # ドキュメント
│   ├── research_overview.md   # 研究概要
│   ├── implemention_guide.md  # 実装ガイド
│   ├── evaluation.md          # 評価方法
│   └── status.md              # 進捗チェックリスト
├── notebooks/        # Jupyter notebooks
├── outputs/          # 学習済みモデル・ログ・評価結果
│   ├── experiments/  # 実験ログ（Git管理対象）
│   │   └── {version}_{type}_{date}/  # 例: v3_baseline_20251205
│   │       ├── config.yaml           # 実験設定
│   │       ├── train_log.csv         # 学習ログ
│   │       ├── eval_metrics.json     # 評価結果
│   │       ├── predictions_sample.jsonl  # 予測サンプル
│   │       └── summary.md            # 実験サマリー
│   ├── models/       # モデルチェックポイント（.gitignore）
│   └── logs/         # 一時ログ（.gitignore）
├── scripts/          # CLIスクリプト
│   ├── generate_dataset.py              # データセット生成
│   ├── train.py                         # モデル学習
│   ├── generate_predictions.py          # モデル推論（Hub連携）
│   ├── generate_predictions_with_penalty.py  # Repetition penalty適用版
│   ├── test_repetition_penalty.py       # Penalty効果テスト
│   ├── prepare_human_eval.py            # 人手評価サンプル準備
│   └── analyze_human_eval.py            # 人手評価結果分析
├── src/              # ソースコード
│   ├── config.py              # 設定ロード
│   ├── data/                  # データ処理
│   │   ├── emoji_utils.py     # 絵文字ユーティリティ
│   │   ├── text_preprocessor.py  # テキスト前処理
│   │   └── wikipedia_loader.py   # Wikipediaデータローダー
│   ├── evaluation/            # 評価
│   │   └── metrics.py         # 評価指標
│   ├── generation/            # データセット生成
│   │   ├── dataset_generator.py  # 生成パイプライン
│   │   ├── openrouter_client.py  # OpenRouterクライアント
│   │   └── prompts.py            # プロンプトテンプレート
│   └── models/                # モデル
│       ├── bert_baseline.py   # BERTベースライン（未実装）
│       └── t5_trainer.py      # T5学習ユーティリティ
└── tests/            # テスト
```

## 開発コマンド

### 環境構築

```bash
mise install                    # Python 3.12とuvをインストール
UV_CACHE_DIR=.uv-cache uv sync  # 依存関係同期
cp .env.example .env            # 環境変数設定
```

### 実行

```bash
# データセット生成
uv run scripts/generate_dataset.py --config configs/default.yaml

# モデル学習
uv run scripts/train.py --config configs/default.yaml

# モデル推論（HuggingFace Hubから）
uv run scripts/generate_predictions.py \
    --model AtefAndrus/jmoji-t5-v4_focal_top50_20251224 \
    --input texts.txt \
    --output predictions.jsonl

# 人手評価サンプル生成（50件）
uv run scripts/prepare_human_eval.py \
    --model-a-repo AtefAndrus/jmoji-t5-v4_focal_top50_20251224 \
    --model-b-repo AtefAndrus/jmoji-t5-v4_top50_20251224 \
    --input-file data/test.jsonl \
    --max-samples 50

# Repetition penalty適用版の予測生成
uv run scripts/generate_predictions_with_penalty.py \
    --model AtefAndrus/jmoji-t5-v4_top50_20251224 \
    --penalty 1.2 \
    --input texts.txt \
    --output predictions_with_penalty.jsonl

# Repetition penaltyの効果テスト
uv run scripts/test_repetition_penalty.py \
    --model top50 \
    --penalties 1.0 1.1 1.2 1.3

# 人手評価結果の集計・分析
uv run scripts/analyze_human_eval.py \
    --space-id AtefAndrus/jmoji-human-eval \
    --output outputs/human_eval/results.json
```

### テスト・リント

```bash
uv run pytest tests/                      # 全テスト実行
uv run pytest tests/test_metrics.py -v    # 特定テスト
uv run ruff check src/ scripts/ tests/    # リント
uv run mypy src/ scripts/                 # 型チェック
```

### pre-commit

```bash
uv run pre-commit install                 # フックをインストール（初回のみ）
uv run pre-commit run --all-files         # 全ファイルに対して実行
```

commit時に以下が自動実行される:

- jupytext: `notebooks/*.py` → `.ipynb` 変換
- ruff: リント・フォーマット（notebooks/除外）
- mypy: 型チェック（src/, scripts/）
- trailing-whitespace, end-of-file-fixer等

## 設定ファイル

`configs/default.yaml` に以下のセクションがある:

- `data`: Wikipedia取得、サンプリング、出力パス
- `teacher`: Qwen3-235B-A22B（OpenRouter経由）の設定
- `emoji`: 絵文字数制限、肌色正規化
- `training`: T5学習ハイパーパラメータ
- `evaluation`: 評価指標設定

## 主要コンポーネント

### データ生成パイプライン

1. Wikipedia文をロード（`wikipedia_loader.py`）
2. テキスト前処理・文抽出（`text_preprocessor.py`）
3. Qwen3-235B-A22BでSNS風変換→絵文字生成（`openrouter_client.py`, `prompts.py`）
4. 絵文字抽出・検証（`emoji_utils.py`）
5. JSONLとして保存（`dataset_generator.py`）

### 評価指標（`metrics.py`）

- Jaccard類似度（主要指標）
- 集合ベース Precision/Recall/F1
- Micro F1
- 完全一致率（Exact Match）
- 出力長分布分析

### T5学習（`t5_trainer.py`）

- 絵文字トークン追加（`setup_model_with_emoji_tokens`）
- `EmojiDataset` クラス
- `TrainConfig` データクラス（学習設定）
- `build_trainer` 関数（EarlyStoppingCallback対応）
- `build_focal_loss_trainer` 関数（Focal Loss対応版）
- `FocalLossTrainer` クラス（クラス不均衡対策）
- `generate_emoji` 関数（推論）
  - `repetition_penalty`: 繰り返し抑制（デフォルト1.2）
  - `use_sampling`: サンプリング有効化（デフォルトTrue）
  - `temperature`, `top_k`, `top_p`: サンプリングパラメータ
- `evaluate_model` 関数（評価指標計算）
- `ExperimentLoggingCallback` クラス（学習ログCSV出力）

## 環境変数

`.env` ファイルに設定:

```text
OPENROUTER_API_KEY=your_api_key_here
```

## 注意事項

- APIキーは `.env` に保存し、コミットしない
- 大規模データ生成時はレート制限（`request_delay`）に注意
- GPU学習はGoogle Colab Pro（A100）を想定
- 絵文字は Emoji 16.0 準拠、肌色バリアントは基本絵文字に統合
- **絵文字バランス**: ✨（キラキラ）が偏りやすくmode collapseの原因となるため、プロンプトで使用を禁止している

## ノートブック

### Colab学習

`notebooks/train_t5_colab.ipynb` でワンクリック学習が可能。READMEの「Open in Colab」バッジから起動できる。

ノートブックは `notebooks/train_t5_colab.py`（percent format）から jupytext で自動生成される。

### ローカル学習

`notebooks/train_t5.ipynb` でローカルマシン（GPU環境）での学習が可能。Colab版と同等の機能を持つ。

### 推論

`notebooks/inference.ipynb` でHuggingFace Hubから学習済みモデルをロードして推論が可能。

機能:
- インタラクティブ推論（任意テキスト）
- バッチ推論（テストセットから50件）
- 人手評価用CSV/Markdownエクスポート

## 運用ルール

### データセット生成時のシード管理

学習に使用するデータセット（v4以降）は、バージョンごとに `random_seed` を変更する。

| バージョン | seed | 備考 |
|------------|------|------|
| v1, v2, v3 | 42 | 初期開発・品質改善 |
| v4以降 | 43, 44, ... | バージョンごとに変更 |

**理由**: 同じseedでは同じ文が同じ順番で出てくるため、データセット間の独立性が低下する。

詳細は [dataset_generation_v3.md](docs/details/datasets/generation_v3.md) を参照。

### 実験ログ管理

Colab学習の結果は `outputs/experiments/` に保存し、Gitで管理する。

#### 命名規則

```text
{dataset_version}_{experiment_type}_{date}
```

| 要素 | 説明 | 例 |
|------|------|-----|
| dataset_version | データセットバージョン | v3, v4 |
| experiment_type | 実験の種類 | baseline, focal_loss, lr1e-4, top100_emojis |
| date | 実験日（YYYYMMDD） | 20251205 |

例: `v3_baseline_20251205`, `v3_focal_loss_20251206`, `v4_lr1e-4_20251210`

#### 保存ファイル

| ファイル | 内容 | 形式 |
|----------|------|------|
| config.yaml | 実験設定（ハイパラ、データ分割等） | YAML |
| train_log.csv | エポックごとのloss推移 | CSV |
| eval_metrics.json | テストセット評価結果 | JSON |
| predictions_sample.jsonl | 予測サンプル（20件） | JSONL |
| summary.md | 実験サマリー（人間/AI可読） | Markdown |

#### ワークフロー

1. Colabで学習実行（ノートブック末尾で自動保存）
2. Google Driveに自動コピー
3. GitHubに自動プッシュ（`GITHUB_TOKEN` 設定時）
4. モデルをHugging Face Hubにアップロード（`HF_TOKEN` 設定時）

### データセット管理

データセットはHugging Face Hubで管理: `AtefAndrus/jmoji-dataset`

#### データセットのアップロード

```bash
# 環境変数でHFトークンを設定
export HF_TOKEN="hf_..."

# 全バージョンをアップロード
uv run scripts/upload_dataset_to_hf.py

# 特定バージョンのみ
uv run scripts/upload_dataset_to_hf.py --versions v3 v4
```

#### データセットの使用

```python
from datasets import load_dataset

# 最新バージョン（v4）をロード
dataset = load_dataset("AtefAndrus/jmoji-dataset", data_files="data/v4.jsonl", split="train")
```

## ドキュメント

`docs/` 以下に各種ドキュメントを配置している。

### メインドキュメント

| ファイル | 内容 |
|----------|------|
| [research_overview.md](docs/research_overview.md) | 研究概要。タスク定義、手法（データセット構築・モデル構成）、評価方法、スケジュール、先行研究との差異 |
| [implemention_guide.md](docs/implemention_guide.md) | 実装ガイド。環境構築、データパイプライン、Claude API呼び出し、絵文字処理、T5ファインチューニング、トラブルシューティング |
| [evaluation.md](docs/evaluation.md) | 評価方法の詳細。Jaccard類似度、Precision/Recall/F1、人手評価設計、エラー分析カテゴリ |
| [status.md](docs/status.md) | 進捗チェックリスト。実装・スクリプト・テスト・データ運用・モデル評価の完了状況 |

### 詳細ドキュメント（`docs/details/`）

#### 実験記録（`experiments/`）

| ファイル | 内容 |
|----------|------|
| [v1_1000samples.md](docs/details/experiments/v1_1000samples.md) | 実験記録v1。1,000件データセットでの学習結果。mode collapse発生と対策検討 |
| [v3_5000samples.md](docs/details/experiments/v3_5000samples.md) | 実験記録v3。5,000件データセットでの学習結果。soft mode collapse発生 |
| [plan_v3_improvements.md](docs/details/experiments/plan_v3_improvements.md) | 学習改善の実験計画。学習率調整、Top-100制限、Focal Loss |
| [v3_improvements.md](docs/details/experiments/v3_improvements.md) | **学習改善実験の結果**。4実験完了、top100が最良（Jaccard 0.058） |
| [v4_results.md](docs/details/experiments/v4_results.md) | **v4データセット実験結果**。focal_top50が最良（Jaccard 0.182） |

#### データセット（`datasets/`）

| ファイル | 内容 |
|----------|------|
| [generation_v3.md](docs/details/datasets/generation_v3.md) | データセット生成v3の品質改善。事前フィルタ、SNS絵文字除去、件数保証 |

#### 評価結果（`evaluations/`）

| ファイル | 内容 |
|----------|------|
| [llm_eval_results.md](docs/details/evaluations/llm_eval_results.md) | LLM-as-a-Judge評価結果。repetition penalty導入の効果検証 |
| [human_eval_results.md](docs/details/evaluations/human_eval_results.md) | 人手評価パイロット結果（20件）。LLM評価との整合性分析 |

#### その他

| ファイル | 内容 |
|----------|------|
| [teacher_model_migration.md](docs/details/teacher_model_migration.md) | 教師モデル移行。Claude Haiku 4.5→Qwen3-235B-A22Bへの変更理由・コスト比較 |

## 進捗管理

`docs/status.md` でタスクの完了状況を管理している。
