# 進捗チェックリスト（更新: 2025-12-03）

## 実装

- [x] OpenRouterクライアント（同期・非同期・ストリーム対応）
- [x] プロンプトテンプレート（SNS変換/絵文字生成）
- [x] テキスト前処理（正規化・文抽出）
- [x] 絵文字ユーティリティ（取得/肌色正規化/抽出）
- [x] データセット生成ユーティリティ（検証・保存/読込・進捗表示・途中保存・並列対応）
- [x] 評価指標（Jaccard, Precision/Recall/F1, Micro-F1, Exact Match, 長さ分布）
- [x] T5用DatasetとTrainer生成ユーティリティ（絵文字トークン追加含む）
- [x] Wikipediaデータローダー（`src/data/wikipedia_loader.py`）
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
- [ ] 小規模（〜1k）疑似対訳の生成と共有
- [ ] 大規模データ生成ジョブの運用手順化（レート制限・エラーリトライ）

## モデル/評価マイルストーン

- [ ] 小規模T5学習のスモーク（数エポック）
- [ ] ベースラインvs学生モデルの自動評価レポート
- [ ] 人手評価フレームの整備と評価者リクルート
- [ ] エラー分析テンプレートでの事例収集

---

## 未実装: NSFWコンテンツ対策

### 背景

WikipediaにはNSFW（性的・暴力的）な記事が存在し、Claude APIがこれらの処理を拒否する可能性がある。
現状はエラー時にスキップするが、拒否率が高いとコスト効率が低下する。

### 実装方針: A（キーワードフィルタ）+ C（ログ監視）

#### A. キーワードブラックリストによる事前フィルタ

**変更ファイル:** `src/data/text_preprocessor.py`

```python
# NSFWキーワードブラックリスト
NSFW_KEYWORDS: set[str] = {
    # 性的コンテンツ
    "性行為", "性交", "ポルノ", "アダルト", "風俗",
    # 暴力的コンテンツ
    "殺人", "虐殺", "拷問", "処刑",
    # 必要に応じて追加
}

def is_safe_sentence(text: str) -> bool:
    """NSFWキーワードを含まないかチェック"""
    return not any(kw in text for kw in NSFW_KEYWORDS)
```

**変更ファイル:** `src/data/wikipedia_loader.py`

```python
from src.data.text_preprocessor import is_safe_sentence

# extract_sentences の結果をフィルタ
sents = [s for s in extract_sentences(text, ...) if is_safe_sentence(s)]
```

#### C. API拒否のログ記録

**変更ファイル:** `src/generation/dataset_generator.py`

```python
# エラー種別を記録
except httpx.HTTPStatusError as e:
    if "content_policy" in str(e) or e.response.status_code == 400:
        logger.warning(f"Content policy rejection at {idx}: {sentence[:50]}...")
        stats.content_rejections += 1
    else:
        stats.errors += 1
```

**GenerationStats に追加:**

```python
@dataclass
class GenerationStats:
    ...
    content_rejections: int = 0  # API拒否数
```

### 設定

`configs/default.yaml` に追加:

```yaml
data:
  nsfw_filter: true  # NSFWフィルタを有効にするか
```

### テスト

- `is_safe_sentence` の単体テスト
- 既知のNSFWキーワードを含む文がフィルタされることを確認

### 優先度

中（大規模データ生成前に実装推奨）

### 要調査

**API拒否の判定ロジック:**

現在の案:

```python
if "content_policy" in str(e) or e.response.status_code == 400:
```

確認が必要な点:

1. OpenRouter経由でClaudeのコンテンツポリシー拒否が発生した場合のHTTPステータスコード（400? 403? 422?）
2. エラーレスポンスのJSON構造（`error.type`, `error.code` 等のフィールド）
3. `content_policy` というキーワードが実際に含まれるか

調査方法:

- OpenRouterドキュメントのエラーハンドリングセクションを確認
- Anthropic APIドキュメントのエラーコード一覧を確認
- 実際にNSFWコンテンツを送信してエラーレスポンスを観察（テスト環境で）

参考リンク:

- <https://openrouter.ai/docs/api/reference/errors>
- <https://platform.claude.com/docs/en/api/errors>
