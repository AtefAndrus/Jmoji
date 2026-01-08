# 進捗チェックリスト（更新: 2026-01-08 人手評価パイロット完了）

## 実装

- [x] OpenRouterクライアント（同期・非同期・ストリーム対応）
- [x] プロンプトテンプレート（SNS変換/絵文字生成）
- [x] テキスト前処理（正規化・文抽出・NSFWフィルタ・文完全性フィルタ・絵文字除去）
- [x] 絵文字ユーティリティ（取得/肌色正規化/抽出）
- [x] データセット生成ユーティリティ（検証・保存/読込・進捗表示・途中保存・並列対応・resume対応・API拒否ログ・件数保証）
- [x] 評価指標（Jaccard, Precision/Recall/F1, Micro-F1, Exact Match, 長さ分布）
- [x] T5用DatasetとTrainer生成ユーティリティ（絵文字トークン追加含む）
- [x] T5推論・評価関数（`generate_emoji`, `evaluate_model`, `EvaluationResult`）
- [x] HuggingFace Hubモデルロード（`load_model_from_hub`）
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
- [x] `scripts/generate_predictions.py`: HuggingFace Hubからモデルをロードして推論
  - [x] テキストファイル/JSONL/標準入力からの推論
  - [x] JSONL/テキスト形式での出力
  - [x] サンプリング/Beam Search切り替え
- [x] `scripts/prepare_human_eval.py`: 人手評価サンプル準備（Hub連携機能追加）
  - [x] 既存の予測ファイルをマージ（従来機能）
  - [x] HuggingFace Hubからモデルをロードして推論（新機能）
  - [x] テストセットから50件をランダム抽出
  - [x] JSONL/CSV/Markdown形式で出力

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
  - [x] v4_top100: top100フィルタ適用 → Jaccard 0.120（+106% vs v3_top100、目標達成）
  - [x] v4_focal_top100: Focal Loss適用 → Jaccard 0.115、**多様性25%（多様性最良）**
  - [x] v4_top50: top50フィルタ適用 → Jaccard 0.165（+38%）、多様性21%
  - [x] v4_focal_top50: Focal Loss + top50 → **Jaccard 0.182（精度最良）**、多様性14%
- [ ] ベースラインvs学生モデルの自動評価レポート
- [x] 人手評価フレームの整備 → 実施計画作成済み、詳細は [evaluation.md](evaluation.md) セクション3.5
- [x] **LLM-as-a-Judge評価** → 完了、詳細は [LLM評価結果](details/llm_eval_results.md)
  - Claude Opus 4.5 subagentによる自動評価（20サンプル×2モデル）
  - v4_focal_top50 vs v4_top50 比較: **Jaccardと主観評価が逆転**
  - Jaccard最良のfocal_top50（0.182）よりtop50（0.165）がLLM評価で優位（9勝6敗）
  - 原因: Focal Lossによる過剰生成（😊😊😊😊😊）が自然さを低下
- [ ] 人手評価の実施（評価サンプル抽出→フォーム作成→評価→集計）
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
- [x] `notebooks/inference.py`: Colab推論用ノートブック
  - [x] HuggingFace Hubからモデルロード
  - [x] インタラクティブ推論（任意テキスト）
  - [x] バッチ推論（テストセットから50件）
  - [x] 人手評価用CSV/Markdownエクスポート

## 実験記録・技術ドキュメント

- [experiment_v1_1000samples.md](details/experiment_v1_1000samples.md): 1,000件データセットでの学習結果（mode collapse発生）
- [experiment_v3_5000samples.md](details/experiment_v3_5000samples.md): 5,000件データセットでの学習結果（soft mode collapse発生）
- [experiment_plan_v3_improvements.md](details/experiment_plan_v3_improvements.md): 学習改善の実験計画（lr調整、Top-100制限、Focal Loss）
- [experiment_v3_improvements.md](details/experiment_v3_improvements.md): 学習改善実験の結果（4実験完了、top100が最良）
- [experiment_v4_results.md](details/experiment_v4_results.md): **v4データセット学習実験の結果**（Jaccard 0.12達成）
- [llm_eval_results.md](details/llm_eval_results.md): **LLM-as-a-Judge評価結果**（Jaccardと主観評価の逆転を発見）
- [human_eval_results.md](details/human_eval_results.md): **人手評価結果**（パイロット20件×1名）
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

- [x] **v4データセットでの学習実験** → **完了**（2025-12-24〜25）
  - 精度最良: v4_focal_top50（Jaccard 0.182）
  - 多様性最良: v4_focal_top100（多様性25%）
  - バランス型: v4_top50（Jaccard 0.165、多様性21%）
- [x] **LLM-as-a-Judge評価** → **完了**（2026-01-03）
  - v4_focal_top50 vs v4_top50 の比較評価
  - **発見**: Jaccardと主観評価が逆転（top50がLLM評価で優位）
  - 詳細は [llm_eval_results.md](details/llm_eval_results.md) 参照
- [x] **人手評価の実施** → **パイロット完了**（2026-01-08）、詳細は [human_eval_results.md](details/human_eval_results.md)
  - [x] Step 1: 評価サンプル抽出（scripts/prepare_human_eval.py作成済み、20件生成）
  - [x] Step 1.5: モデル推論機能追加（任意テキストから予測生成、50件への拡張）
    - `src/models/t5_trainer.py` に `load_model_from_hub()` 追加
    - `scripts/generate_predictions.py` 新規作成（CLI推論）
    - `scripts/prepare_human_eval.py` 拡張（Hub連携機能）
    - `notebooks/inference.py` 新規作成（Colab用）
  - [x] Step 2: 評価アプリ作成（Gradio + HuggingFace Spaces）
    - `/home/keigo/jmoji-human-eval/` に評価アプリを作成
    - `scripts/analyze_human_eval.py` 新規作成（結果集計）
  - [x] Step 3: パイロット評価実施（20件×1名）
    - 結果: Gold 2.30/4.0、モデルA 1.00/4.0、モデルB 0.90/4.0（意味的一致度）
    - 選好: focal_top50 6票、top50 3票、同等 11票
  - [ ] Step 4: 追加評価者による評価（オプション）
  - [ ] Step 5: HuggingFace Spacesデプロイ（オプション）
- [x] **モデル改善**: repetition penalty導入（過剰生成対策） → **完了**（2026-01-03）
  - `generate_emoji`関数に`repetition_penalty`パラメータ追加（デフォルト: 1.2）
  - 過剰生成が60%→96%改善、gold絵文字（🏛️🔥🤔）の出現率向上
  - **推奨設定**: v4_top50 + repetition_penalty=1.2
- [x] モデル選択方針決定 — **単一モデル運用**: v4_top50 + repetition_penalty=1.2
- [x] v4データセットをHuggingFace Hubにアップロード → **完了**（2025-12-23）

## ユーザータスク

- [ ] **Colab H100スペック確認** — 接続できたらVRAM、料金（CU/h）、利用制限を確認
