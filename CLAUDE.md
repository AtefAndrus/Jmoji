# CLAUDE.md

Jmojiプロジェクト用のClaude Code設定ファイル。

## プロジェクト概要

知識蒸留を用いた日本語テキスト→絵文字翻訳モデルの開発プロジェクト。
Claude Haiku 4.5を教師モデルとして疑似対訳データセットを構築し、日本語T5（`sonoisa/t5-base-japanese`）へ知識蒸留を行う。

## 技術スタック

- Python 3.12
- パッケージ管理: uv + mise
- 主要ライブラリ: transformers, torch, datasets, emoji, httpx
- 開発ツール: ruff, mypy, pytest

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

- 絵文字トークン追加
- EmojiDataset クラス
- Trainer構築ユーティリティ

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

`notebooks/train_t5.ipynb` でワンクリック学習が可能。READMEの「Open in Colab」バッジから起動できる。

## 進捗管理

`docs/status.md` でタスクの完了状況を管理している。
