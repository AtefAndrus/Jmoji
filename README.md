# Jmoji

知識蒸留を用いた日本語テキスト→絵文字翻訳モデル

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
j-moji/
├── configs/          # 設定ファイル
├── data/             # データセット
├── docs/             # ドキュメント
├── notebooks/        # Jupyter notebooks
├── outputs/          # 学習済みモデル・ログ
├── scripts/          # CLIスクリプト
└── src/              # ソースコード
```

詳細は [docs/](docs/) を参照してください。

## 使い方

### データセット生成

```bash
uv run scripts/generate_dataset.py --config configs/default.yaml
```

### モデル学習

```bash
uv run scripts/train.py --config configs/default.yaml
```

### 推論

```python
from src.models.t5_trainer import EmojiTranslator

model = EmojiTranslator.load("outputs/models/best_model")
emojis = model.translate("今日はいい天気ですね")
print(emojis)  # 😊 ☀️
```

## ドキュメント

- [研究概要](docs/research_overview.md)
- [実装ガイド](docs/implemention_guide.md)
- [評価方法](docs/evaluation.md)
- [進捗チェックリスト](docs/status.md)

## 開発環境

- Python 3.12
- Google Colab Pro（A100 80GB）
- 教師モデル: Claude Haiku 4.5（OpenRouter経由）

## ライセンス

MIT License
