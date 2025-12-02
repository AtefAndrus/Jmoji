# Jmoji

知識蒸留を用いた日本語テキスト→絵文字翻訳モデル

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AtefAndrus/Jmoji/blob/main/notebooks/train_t5.ipynb)

## 概要

日本語テキストから、その文の意味・ニュアンス・トーンを表現する絵文字列（1〜5個）を生成するモデルを開発するプロジェクトです。

Claude Haiku 4.5を教師モデルとして疑似対訳データセットを構築し、日本語T5（`sonoisa/t5-base-japanese`）へ知識蒸留を行います。

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
├── data/             # データセット
│   └── outputs/      # 生成されたデータセット
├── docs/             # ドキュメント
├── notebooks/        # Jupyter notebooks
├── outputs/          # 学習済みモデル・ログ・評価結果
├── scripts/          # CLIスクリプト
│   ├── generate_dataset.py  # データセット生成
│   └── train.py             # モデル学習
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

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AtefAndrus/Jmoji/blob/main/notebooks/train_t5.ipynb)

上のバッジをクリックしてノートブックを開き、上から順に実行してください。A100 GPUを推奨します。

### 開発コマンド

```bash
# テスト実行
uv run pytest tests/ -v

# リント
uv run ruff check src/ scripts/ tests/

# 型チェック
uv run mypy src/ scripts/
```

## ドキュメント

- [研究概要](docs/research_overview.md)
- [実装ガイド](docs/implemention_guide.md)
- [評価方法](docs/evaluation.md)
- [進捗チェックリスト](docs/status.md)

## 開発環境

- Python 3.12
- パッケージ管理: uv + mise
- Google Colab Pro（A100 80GB）での学習を想定
- 教師モデル: Claude Haiku 4.5（OpenRouter経由）

## ライセンス

MIT License
