# CLAUDE.md

Jmojiプロジェクト用のClaude Code設定ファイル。

## プロジェクト概要

知識蒸留を用いた日本語テキスト→絵文字翻訳モデルの開発プロジェクト。
Claude Haiku 4.5を教師モデルとして疑似対訳データセットを構築し、日本語T5（`sonoisa/t5-base-japanese`）へ知識蒸留を行う。

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
│   ├── generate_dataset.py    # データセット生成
│   └── train.py               # モデル学習
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
- `teacher`: Claude Haiku 4.5（OpenRouter経由）の設定
- `emoji`: 絵文字数制限、肌色正規化
- `training`: T5学習ハイパーパラメータ
- `evaluation`: 評価指標設定

## 主要コンポーネント

### データ生成パイプライン

1. Wikipedia文をロード（`wikipedia_loader.py`）
2. テキスト前処理・文抽出（`text_preprocessor.py`）
3. Claude Haiku 4.5でSNS風変換→絵文字生成（`openrouter_client.py`, `prompts.py`）
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
- `generate_emoji` 関数（推論）
- `evaluate_model` 関数（評価指標計算）

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

## Colab学習

`notebooks/train_t5_colab.ipynb` でワンクリック学習が可能。READMEの「Open in Colab」バッジから起動できる。

ノートブックは `notebooks/train_t5_colab.py`（percent format）から jupytext で自動生成される。

## 運用ルール

### データセット生成時のシード管理

学習に使用するデータセット（v4以降）は、バージョンごとに `random_seed` を変更する。

| バージョン | seed | 備考 |
|------------|------|------|
| v1, v2, v3 | 42 | 初期開発・品質改善 |
| v4以降 | 43, 44, ... | バージョンごとに変更 |

**理由**: 同じseedでは同じ文が同じ順番で出てくるため、データセット間の独立性が低下する。

詳細は [dataset_generation_v3.md](docs/details/dataset_generation_v3.md) を参照。

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

# 最新バージョン（v3）をロード
dataset = load_dataset("AtefAndrus/jmoji-dataset", data_files="data/v3.jsonl", split="train")
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

| ファイル | 内容 |
|----------|------|
| [experiment_v1_1000samples.md](docs/details/experiment_v1_1000samples.md) | 実験記録v1。1,000件データセットでの学習結果。✨への偏り（18.6%）によるmode collapse発生と対策検討 |
| [experiment_v3_5000samples.md](docs/details/experiment_v3_5000samples.md) | 実験記録v3。5,000件データセットでの学習結果。soft mode collapse（Top5絵文字への偏り）発生と次ステップ |
| [experiment_plan_v3_improvements.md](docs/details/experiment_plan_v3_improvements.md) | 学習改善の実験計画。学習率調整、Top-100絵文字制限、Focal Lossの4実験を計画 |
| [experiment_v3_improvements.md](docs/details/experiment_v3_improvements.md) | **学習改善実験の結果**。4実験完了、top100が最良（Jaccard 0.058）。データ密度向上が次の課題 |
| [dataset_generation_v3.md](docs/details/dataset_generation_v3.md) | データセット生成v3の品質改善。事前フィルタ、SNS絵文字除去、件数保証の実装詳細 |

## 進捗管理

`docs/status.md` でタスクの完了状況を管理している。
