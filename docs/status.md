# 進捗チェックリスト（更新: 2025-11-30）

## 実装

- [x] OpenRouterクライアント（同期・ストリーム対応）
- [x] プロンプトテンプレート（SNS変換/絵文字生成）
- [x] テキスト前処理（正規化・文抽出）
- [x] 絵文字ユーティリティ（取得/肌色正規化/抽出）
- [x] データセット生成ユーティリティ（検証・保存/読込）
- [x] 評価指標（Jaccard, Precision/Recall/F1, Micro-F1, Exact Match, 長さ分布）
- [x] T5用DatasetとTrainer生成ユーティリティ（絵文字トークン追加含む）
- [ ] BERTベースライン実装
- [ ] 実データ読み込み（Wikipedia/mC4）を含むデータ生成パイプライン
- [ ] 設定ファイルのスキーマ/バリデーション

## スクリプト/CLI

- [ ] `scripts/generate_dataset.py`: 実データ取得→前処理→OpenRouter呼び出し→JSONL保存
  - [ ] `src/data/wikipedia_loader.py` を実装し、`wikimedia/wikipedia` (20231101.ja) から `max_samples` 抽出
  - [ ] `configs/default.yaml` の data/teacher/emoji 設定を反映（長さフィルタ、min/max絵文字数、request_delay 等）
  - [ ] OpenRouter呼び出しを実データに適用し、JSONLを保存（途中保存含む）
- [ ] `scripts/train.py`: JSONL分割→学習→チェックポイント保存→評価
  - [ ] JSONLロード→ train/val/test 分割（`training.train_ratio/val_ratio/test_ratio`）
  - [ ] 絵文字トークン追加済みモデルのロードとTrainer構築（early stopping, logging）
  - [ ] 評価結果を `outputs/evaluation` に保存

## テスト

- [x] 前処理・絵文字・生成・評価・T5ユーティリティの単体テスト
- [ ] OpenRouterクライアントのモックテスト（ストリーム含む）
- [ ] スクリプト統合テスト（小サンプル）

## データ/運用

- [ ] `datasets` でのWikipedia/mC4ダウンロードを組み込む（キャッシュ利用）
- [ ] 小規模（〜1k）疑似対訳の生成と共有
- [ ] 大規模データ生成ジョブの運用手順化（レート制限・エラーリトライ）

## モデル/評価マイルストーン

- [ ] 小規模T5学習のスモーク（数エポック）
- [ ] ベースラインvs学生モデルの自動評価レポート
- [ ] 人手評価フレームの整備と評価者リクルート
- [ ] エラー分析テンプレートでの事例収集
