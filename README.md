# Jmoji

知識蒸留を用いた日本語テキスト→絵文字翻訳モデル

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AtefAndrus/Jmoji/blob/main/notebooks/train_t5_colab.ipynb)

## 概要

日本語テキストから、その文の意味・ニュアンス・トーンを表現する絵文字列（1〜5個）を生成するモデルを開発するプロジェクトです。

LLM（Qwen3-235B-A22B）を教師モデルとして疑似対訳データセットを構築し、日本語T5（`sonoisa/t5-base-japanese`）へ知識蒸留を行います。

## セットアップ

### 1. リポジトリのクローン

```bash
git clone https://github.com/AtefAndrus/Jmoji.git
cd Jmoji
```

### 2. mise + uv で環境構築

```bash
# プロジェクトにピンされたツールを取得（Python 3.12 / uv latest）
mise install

# 依存関係同期（.venv と uv.lock を生成）
UV_CACHE_DIR=.uv-cache uv sync

# （pip 互換の要件ファイルが必要な場合）
uv export --format requirements-txt > requirements.txt
```

`uv run <cmd>` で .venv を自動利用できます。手動で有効化したい場合は `source .venv/bin/activate`。

### 3. 環境変数の設定

```bash
cp .env.example .env
# .env を編集してAPIキーを設定
```

## プロジェクト構成

```text
Jmoji/
├── configs/          # 設定ファイル（YAML）
│   └── default.yaml  # デフォルト設定
├── data/             # データセット（v1〜v4）
├── docs/             # ドキュメント
├── notebooks/        # Jupyter notebooks
├── outputs/          # 学習済みモデル・ログ・評価結果
├── scripts/          # CLIスクリプト
├── src/              # ソースコード
│   ├── config.py            # 設定ロード
│   ├── data/                # データ処理
│   ├── evaluation/          # 評価指標
│   ├── generation/          # データセット生成
│   └── models/              # モデル
└── tests/            # テスト
```

詳細は [docs/](docs/) を参照してください。

## 使い方

### データセット生成

```bash
# 基本的な使用方法
uv run scripts/generate_dataset.py --config configs/default.yaml

# 非同期モード（並列リクエストで高速化）
uv run scripts/generate_dataset.py --config configs/default.yaml --async

# 途中から再開しない（最初から生成）
uv run scripts/generate_dataset.py --config configs/default.yaml --no-resume

# NSFWフィルタを無効化
uv run scripts/generate_dataset.py --config configs/default.yaml --no-nsfw-filter
```

**オプション:**

| オプション | 説明 |
|-----------|------|
| `--config` | 設定ファイルのパス（デフォルト: `configs/default.yaml`） |
| `--async` | 非同期モードで並列リクエスト |
| `--no-resume` | 既存ファイルがあっても最初から生成 |
| `--no-nsfw-filter` | NSFWフィルタを無効化 |

### モデル学習

**ローカル:**

```bash
uv run scripts/train.py --config configs/default.yaml
```

**Google Colab（推奨）:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AtefAndrus/Jmoji/blob/main/notebooks/train_t5_colab.ipynb)

上のバッジをクリックしてノートブックを開き、上から順に実行してください。A100 GPUを推奨します。

**Colab Secretsの設定（オプション）:**

自動コミット・モデルアップロードを有効にする場合、Colabの左サイドバー「鍵」アイコンから以下を設定:

| Secret名 | 用途 | 取得方法 |
|----------|------|----------|
| `GITHUB_TOKEN` | 実験ログの自動コミット | GitHub → Settings → Developer settings → Fine-grained tokens (Contents: Read and write) |
| `HF_TOKEN` | モデルのHF Hubアップロード | huggingface.co/settings/tokens (Write権限) |

未設定の場合はスキップされます（エラーにはなりません）。

### モデル推論

HuggingFace Hubから学習済みモデルをロードして推論:

```bash
# 基本的な使用方法
uv run scripts/generate_predictions.py \
    --model AtefAndrus/jmoji-t5-v4_top50_20251224 \
    --input texts.txt \
    --output predictions.jsonl

# Repetition penalty適用（推奨）
uv run scripts/generate_predictions_with_penalty.py \
    --model AtefAndrus/jmoji-t5-v4_top50_20251224 \
    --penalty 1.2 \
    --input texts.txt \
    --output predictions.jsonl
```

### 人手評価

```bash
# 評価サンプルの準備（50件）
uv run scripts/prepare_human_eval.py \
    --model-a-repo AtefAndrus/jmoji-t5-v4_focal_top50_20251224 \
    --model-b-repo AtefAndrus/jmoji-t5-v4_top50_20251224 \
    --input-file data/test.jsonl \
    --max-samples 50

# 評価結果の集計・分析
uv run scripts/analyze_human_eval.py \
    --space-id AtefAndrus/jmoji-human-eval \
    --output outputs/human_eval/results.json
```

### 開発コマンド

```bash
# テスト実行
uv run pytest tests/ -v

# リント
uv run ruff check src/ scripts/ tests/

# 型チェック
uv run mypy src/ scripts/

# pre-commit（初回のみインストール）
uv run pre-commit install
uv run pre-commit run --all-files
```

## データセット

データセットはHuggingFace Hubで管理しています: [AtefAndrus/jmoji-dataset](https://huggingface.co/datasets/AtefAndrus/jmoji-dataset)

| バージョン | 件数 | 教師モデル | 備考 |
|-----------|------|-----------|------|
| v4 | 20,000 | Qwen3-235B-A22B | 最新・推奨 |
| v3 | 5,000 | Claude Haiku 4.5 | 品質改善版 |
| v1-v2 | 1,000-5,000 | Claude Haiku 4.5 | 初期版 |

```python
from datasets import load_dataset

# 最新バージョン（v4）をロード
dataset = load_dataset("AtefAndrus/jmoji-dataset", data_files="data/v4.jsonl", split="train")
```

### データセットのアップロード

新しいバージョンをアップロードする場合:

```bash
export HF_TOKEN="hf_..."
uv run scripts/upload_dataset_to_hf.py --versions v4
```

## 公開モデル

学習済みモデルはHuggingFace Hubで公開しています:

| モデル | Jaccard | 多様性 | 用途 |
|--------|---------|--------|------|
| [jmoji-t5-v4_top50](https://huggingface.co/AtefAndrus/jmoji-t5-v4_top50_20251224) | 0.165 | 21% | **推奨（バランス型）** |
| [jmoji-t5-v4_focal_top50](https://huggingface.co/AtefAndrus/jmoji-t5-v4_focal_top50_20251224) | 0.182 | 14% | 精度重視 |

**推奨設定**: `v4_top50` + `repetition_penalty=1.2`

- repetition penaltyにより過剰生成（😊😊😊）を抑制
- 自然さと精度のバランスが良好

## 実験結果

v4データセット（20,000件）での学習実験結果:

| 実験 | データ件数 | Jaccard | 多様性 |
|------|-----------|---------|--------|
| v4_focal_top50 | 1,337 | **0.182** | 14% |
| v4_top50 | 1,337 | 0.165 | 21% |
| v4_focal_top100 | 4,237 | 0.115 | **25%** |
| v4_top100 | 4,237 | 0.120 | 21% |

詳細は [v4実験結果](docs/details/experiments/v4_results.md) を参照。

## 評価結果

### LLM-as-a-Judge評価

Claude Opus 4.5による自動評価（20サンプル）:

- v4_top50がv4_focal_top50より優位（9勝6敗）
- Focal Lossによる過剰生成が自然さを低下

詳細は [LLM評価結果](docs/details/evaluations/llm_eval_results.md) を参照。

### 人手評価（パイロット）

パイロット評価（20サンプル、1名）:

| モデル | 意味的一致度 | 自然さ |
|--------|-------------|--------|
| 教師（Gold） | 2.30/4.0 | 2.15/4.0 |
| focal_top50 | 1.00/4.0 | 1.30/4.0 |
| top50 | 0.90/4.0 | 1.25/4.0 |

詳細は [人手評価結果](docs/details/evaluations/human_eval_results.md) を参照。

## ドキュメント

### メインドキュメント

- [研究概要](docs/research_overview.md)
- [実装ガイド](docs/implemention_guide.md)
- [評価方法](docs/evaluation.md)
- [進捗チェックリスト](docs/status.md)

### 詳細ドキュメント

- 実験記録: [v4結果](docs/details/experiments/v4_results.md) / [v3改善](docs/details/experiments/v3_improvements.md)
- 評価結果: [LLM評価](docs/details/evaluations/llm_eval_results.md) / [人手評価](docs/details/evaluations/human_eval_results.md)
- その他: [教師モデル移行](docs/details/teacher_model_migration.md)

## 開発環境

- Python 3.12
- パッケージ管理: uv + mise
- Google Colab Pro（A100 80GB）での学習を想定
- 教師モデル: Qwen3-235B-A22B（OpenRouter経由）
  - v1〜v3データセットはClaude Haiku 4.5で生成
  - 移行理由: [docs/details/teacher_model_migration.md](docs/details/teacher_model_migration.md)

## ライセンス

MIT License
