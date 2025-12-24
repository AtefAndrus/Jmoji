# 進捗チェックリスト（更新: 2025-12-25 v4_focal_top100完了、多様性25%達成）

## 実装

- [x] OpenRouterクライアント（同期・非同期・ストリーム対応）
- [x] プロンプトテンプレート（SNS変換/絵文字生成）
- [x] テキスト前処理（正規化・文抽出・NSFWフィルタ・文完全性フィルタ・絵文字除去）
- [x] 絵文字ユーティリティ（取得/肌色正規化/抽出）
- [x] データセット生成ユーティリティ（検証・保存/読込・進捗表示・途中保存・並列対応・resume対応・API拒否ログ・件数保証）
- [x] 評価指標（Jaccard, Precision/Recall/F1, Micro-F1, Exact Match, 長さ分布）
- [x] T5用DatasetとTrainer生成ユーティリティ（絵文字トークン追加含む）
- [x] T5推論・評価関数（`generate_emoji`, `evaluate_model`, `EvaluationResult`）
- [x] Wikipediaデータローダー（事前フィルタ統合・フィルタログ出力）
- [x] NSFWコンテンツフィルタ（キーワードブラックリストによる事前フィルタ + API拒否ログ）
- [x] pre-commit設定（jupytext, ruff, mypy, trailing-whitespace等）
- [x] FocalLossTrainer実装（`src/models/t5_trainer.py`）
- [x] 多様性指標（`diversity_ratio`, `emoji_distribution`）実装（`src/evaluation/metrics.py`）
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
  - [x] 文完全性フィルタ（`--no-complete-filter`オプション）
  - [x] フィルタログ出力（`filtered_sentences.jsonl`）
  - [x] 件数保証（`target_count`、`buffer_ratio`設定）
  - [x] SNS変換結果からの絵文字除去
- [x] `scripts/train.py`: JSONL分割→学習→チェックポイント保存→評価
  - [x] JSONLロード→ train/val/test 分割（`training.train_ratio/val_ratio/test_ratio`）
  - [x] 絵文字トークン追加済みモデルのロードとTrainer構築（early stopping, logging）
  - [x] 評価結果を `outputs/evaluation` に保存
- [x] `scripts/upload_dataset_to_hf.py`: データセットをHuggingFace Hubにアップロード

## テスト

- [x] 前処理・絵文字・生成・評価・T5ユーティリティの単体テスト
- [x] Wikipediaローダーの単体テスト
- [x] OpenRouterクライアントのモックテスト（同期・非同期・レート制限対応）
- [ ] スクリプト統合テスト（小サンプル）

## データ/運用

- [x] `datasets` でのWikipediaダウンロードを組み込む（ストリーミング・キャッシュ利用）
- [x] 小規模（〜1k）疑似対訳の生成と共有
- [x] 中規模（5k）疑似対訳の生成（✨禁止プロンプト適用）→ dataset_v2.jsonl
- [x] 中規模（5k）品質改善版の生成（事前フィルタ・件数保証適用）→ dataset_v3.jsonl、詳細は [dataset_generation_v3.md](details/dataset_generation_v3.md)
- [x] データセットのHuggingFace Hub移行（`AtefAndrus/jmoji-dataset`）
- [x] **v4データセット生成（20k件）** → dataset_v4.jsonl、Qwen3-235B-A22B使用、フィルタ除外1,309件
- [x] **v4データセットをHuggingFace Hubにアップロード**（2025-12-23）
- [ ] 大規模データ生成ジョブの運用手順化（レート制限・エラーリトライ）

## モデル/評価マイルストーン

- [x] 小規模T5学習のスモーク（数エポック）→ mode collapse発生、詳細は [実験記録v1](details/experiment_v1_1000samples.md)
- [x] 中規模（5k）データでのT5学習 → soft mode collapse発生、詳細は [実験記録v3](details/experiment_v3_5000samples.md)
- [x] **学習改善実験** → 完了、詳細は [実験結果](details/experiment_v3_improvements.md)
  - [x] Exp1: 学習率調整 (v3_lr1e-4) → Jaccard 0.048（+7%）
  - [x] Exp2: Top-100絵文字制限 (v3_top100) → **Jaccard 0.058（+29%、最良）**
  - [x] Exp3: 組み合わせ (v3_lr1e-4_top100) → 失敗（学習率低すぎ）
  - [x] Exp4: Focal Loss + Top-100 (v3_focal_top100) → Jaccard 0.058、初Exact Match
- [x] **v4データセット学習実験** → 完了、詳細は [実験結果](details/experiment_v4_results.md)
  - [x] v4_lr1e-4: 20k件全件使用 → Jaccard 0.066（+14% vs v3_top100）
  - [x] v4_top100: top100フィルタ適用 → **Jaccard 0.120（+106% vs v3_top100、目標達成）**
  - [x] v4_focal_top100: Focal Loss適用 → Jaccard 0.115、**多様性25%（+19%、最良）**
- [ ] ベースラインvs学生モデルの自動評価レポート
- [ ] 人手評価フレームの整備と評価者リクルート
- [ ] エラー分析テンプレートでの事例収集

## ノートブック/Colab

- [x] `notebooks/train_t5_colab.py`: Colab学習用ノートブック（src/からモジュールインポート）
- [x] jupytextによる`.py`→`.ipynb`自動変換（pre-commit連携）
- [x] `requirements-colab.txt`廃止、`pip install .`方式に移行
- [x] 実験ログの自動保存（`outputs/experiments/{name}/` に config.yaml, train_log.csv, eval_metrics.json, summary.md）
- [x] GitHubへの自動コミット（`GITHUB_TOKEN` 設定時）
- [x] HuggingFace Hubへのモデルアップロード（`HF_TOKEN` 設定時）
- [x] HuggingFace Hubからのデータセットロード（`datasets.load_dataset()`）
- [x] 実験切り替え機能（`EXPERIMENT_TYPE`で設定自動調整）
- [x] Top-100絵文字フィルタリング（`use_top100_filter`オプション）
- [x] Focal Loss対応（`use_focal_loss`オプション）
- [x] 多様性指標の評価・保存

## 実験記録・技術ドキュメント

- [experiment_v1_1000samples.md](details/experiment_v1_1000samples.md): 1,000件データセットでの学習結果（mode collapse発生）
- [experiment_v3_5000samples.md](details/experiment_v3_5000samples.md): 5,000件データセットでの学習結果（soft mode collapse発生）
- [experiment_plan_v3_improvements.md](details/experiment_plan_v3_improvements.md): 学習改善の実験計画（lr調整、Top-100制限、Focal Loss）
- [experiment_v3_improvements.md](details/experiment_v3_improvements.md): 学習改善実験の結果（4実験完了、top100が最良）
- [experiment_v4_results.md](details/experiment_v4_results.md): **v4データセット学習実験の結果**（Jaccard 0.12達成）
- [dataset_generation_v3.md](details/dataset_generation_v3.md): データセット生成v3の品質改善と件数保証
- [teacher_model_migration.md](details/teacher_model_migration.md): 教師モデル移行（Claude Haiku 4.5→Qwen3-235B-A22B）

## 教師モデル移行

詳細は [teacher_model_migration.md](details/teacher_model_migration.md) を参照。

- [x] `configs/default.yaml`のモデルIDを`qwen/qwen3-235b-a22b-2507`に変更
- [x] OpenRouterにクレジットをチャージ（$10〜$20推奨）
- [x] 小規模テスト（100サンプル程度）でQwen3出力品質を確認
  - 3設定を比較検証（デフォルト / min_p=0.1 / Shisa推奨）
  - Shisa推奨（temp=0.2, min_p=0.1）で最良: 成功率100%、Cross-lingual leakage 0件
- [x] Shisa.AI推奨設定を`configs/default.yaml`に反映
- [x] v4データセット生成を実行 → **20,000件生成完了**（2025-12-19）

## 次のステップ

- [x] **v4データセットでの学習実験** → **完了**（2025-12-24）、Jaccard 0.12達成
- [x] **v4_focal_top100の検証** → **完了**（2025-12-25）、多様性25%達成（Jaccard微減）
- [ ] **v4_top50の検証** — 件/絵文字 84件でJaccard 0.15+、多様性30%+を期待
- [ ] v4_focal_top50の検証 — top50 + Focal Lossで多様性のさらなる改善
- [ ] 人手評価の実施（Jaccard以外の品質指標確認）
- [x] v4データセットをHuggingFace Hubにアップロード → **完了**（2025-12-23）

## ユーザータスク

- [ ] **Colab H100スペック確認** — 接続できたらVRAM、料金（CU/h）、利用制限を確認
