# 進捗チェックリスト（更新: 2025-12-03）

## 実装

- [x] OpenRouterクライアント（同期・非同期・ストリーム対応）
- [x] プロンプトテンプレート（SNS変換/絵文字生成）
- [x] テキスト前処理（正規化・文抽出・NSFWフィルタ）
- [x] 絵文字ユーティリティ（取得/肌色正規化/抽出）
- [x] データセット生成ユーティリティ（検証・保存/読込・進捗表示・途中保存・並列対応・resume対応・API拒否ログ）
- [x] 評価指標（Jaccard, Precision/Recall/F1, Micro-F1, Exact Match, 長さ分布）
- [x] T5用DatasetとTrainer生成ユーティリティ（絵文字トークン追加含む）
- [x] Wikipediaデータローダー（`src/data/wikipedia_loader.py`）
- [x] NSFWコンテンツフィルタ（キーワードブラックリストによる事前フィルタ + API拒否ログ）
- [ ] BERTベースライン実装
- [ ] 設定ファイルのスキーマ/バリデーション（現状は単純なYAMLロードのみ）

## スクリプト/CLI

- [x] `scripts/generate_dataset.py`: 実データ取得→前処理→OpenRouter呼び出し→JSONL保存
  - [x] `src/data/wikipedia_loader.py` を実装し、`wikimedia/wikipedia` (20231101.ja) から `max_samples` 抽出
  - [x] `configs/default.yaml` の data/teacher/emoji 設定を反映（長さフィルタ、min/max絵文字数、request_delay 等）
  - [x] OpenRouter呼び出しを実データに適用し、JSONLを保存
  - [x] 途中保存機能（追記モードで即時保存、resume対応）
  - [x] 進捗表示（tqdm + サンプルプレビュー + 統計情報）
  - [x] 並列リクエスト（`--async`オプション、`AsyncOpenRouterClient`）
  - [x] NSFWフィルタ（`--no-nsfw-filter`オプション、設定ファイルでキーワード指定可能）
  - [x] 非同期モードのresume対応
- [x] `scripts/train.py`: JSONL分割→学習→チェックポイント保存→評価
  - [x] JSONLロード→ train/val/test 分割（`training.train_ratio/val_ratio/test_ratio`）
  - [x] 絵文字トークン追加済みモデルのロードとTrainer構築（early stopping, logging）
  - [x] 評価結果を `outputs/evaluation` に保存

## テスト

- [x] 前処理・絵文字・生成・評価・T5ユーティリティの単体テスト
- [x] Wikipediaローダーの単体テスト
- [x] OpenRouterクライアントのモックテスト（同期・非同期・レート制限対応）
- [ ] スクリプト統合テスト（小サンプル）

## データ/運用

- [x] `datasets` でのWikipediaダウンロードを組み込む（ストリーミング・キャッシュ利用）
- [x] 小規模（〜1k）疑似対訳の生成と共有
- [ ] 中規模（5k）疑似対訳の生成（✨禁止プロンプト適用）
- [ ] 大規模データ生成ジョブの運用手順化（レート制限・エラーリトライ）

## モデル/評価マイルストーン

- [ ] 小規模T5学習のスモーク（数エポック）
- [ ] ベースラインvs学生モデルの自動評価レポート
- [ ] 人手評価フレームの整備と評価者リクルート
- [ ] エラー分析テンプレートでの事例収集
