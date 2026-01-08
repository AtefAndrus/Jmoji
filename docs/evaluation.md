# 評価方法

## 1. 概要

本研究では、教師LLM（Qwen3-235B-A22B）が生成した絵文字列を「疑似正解」として、学生モデルの出力を評価します。

> **Note**: v1〜v3データセットはClaude Haiku 4.5で生成。v4以降はQwen3-235B-A22Bを使用。
> 移行理由は [teacher_model_migration.md](details/teacher_model_migration.md) を参照。

絵文字の「正解」は本質的に一意ではないため、複数の定量指標と人手評価を組み合わせて多面的に評価を行います。

## 2. 定量的指標

### 2.1 主要指標: Jaccard類似度

教師出力と学生出力の絵文字**集合**の重なり度合いを測定します。

#### 定義

```text
J(A, B) = |A ∩ B| / |A ∪ B|
```

- A: 教師LLMの出力絵文字集合
- B: 学生モデルの出力絵文字集合

#### 特徴

| 利点 | 説明 |
|------|------|
| 順序非依存 | 絵文字の出現順序に影響されない |
| 直感的 | 0〜1の範囲で解釈しやすい |
| 部分一致対応 | 完全一致でなくても評価可能 |

#### 実装例

```python
def jaccard_similarity(pred: set, gold: set) -> float:
    """Jaccard類似度を計算"""
    if not pred and not gold:
        return 1.0  # 両方空の場合は完全一致
    if not pred or not gold:
        return 0.0

    intersection = len(pred & gold)
    union = len(pred | gold)
    return intersection / union

# 使用例
pred = {"😊", "🎉", "✨"}
gold = {"😊", "🎉", "💕"}
print(jaccard_similarity(pred, gold))  # 0.5
```

#### 報告形式

- 平均値 ± 標準偏差
- 中央値
- 分布ヒストグラム

### 2.2 補助指標: 集合ベース Precision / Recall / F1

#### 定義（Precision/Recall/F1）

```text
Precision = |A ∩ B| / |B|  （学生出力のうち正解した割合）
Recall    = |A ∩ B| / |A|  （教師出力のうち再現した割合）
F1        = 2 * P * R / (P + R)
```

#### 実装

```python
def set_based_metrics(pred: set, gold: set) -> dict:
    """集合ベースのPrecision/Recall/F1を計算"""
    if not pred and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    intersection = len(pred & gold)
    precision = intersection / len(pred)
    recall = intersection / len(gold)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}
```

#### Micro vs Macro

| 方式 | 計算方法 | 用途 |
|------|---------|------|
| Micro | 全サンプルのTP/FP/FNを集計してから計算 | 全体性能 |
| Macro | 各サンプルで計算後に平均 | サンプル間の公平性 |

```python
def micro_f1(predictions: list[set], golds: list[set]) -> float:
    """Micro F1を計算"""
    total_intersection = 0
    total_pred = 0
    total_gold = 0

    for pred, gold in zip(predictions, golds):
        total_intersection += len(pred & gold)
        total_pred += len(pred)
        total_gold += len(gold)

    precision = total_intersection / total_pred if total_pred > 0 else 0
    recall = total_intersection / total_gold if total_gold > 0 else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
```

### 2.3 完全一致率（Exact Match Rate）

教師出力と学生出力が**完全に一致**するサンプルの割合。

```python
def exact_match_rate(predictions: list[set], golds: list[set]) -> float:
    """完全一致率を計算"""
    matches = sum(1 for p, g in zip(predictions, golds) if p == g)
    return matches / len(predictions)
```

**注意**: 絵文字翻訳は正解が一意でないため、この指標は参考値として扱います。

### 2.4 出力長分布の比較

教師出力と学生出力の絵文字数分布を比較します。

```python
import numpy as np
from scipy import stats

def length_distribution_analysis(
    pred_lengths: list[int],
    gold_lengths: list[int]
) -> dict:
    """出力長分布の分析"""
    return {
        "pred_mean": np.mean(pred_lengths),
        "pred_std": np.std(pred_lengths),
        "gold_mean": np.mean(gold_lengths),
        "gold_std": np.std(gold_lengths),
        "correlation": np.corrcoef(pred_lengths, gold_lengths)[0, 1],
        "ks_statistic": stats.ks_2samp(pred_lengths, gold_lengths).statistic
    }
```

### 2.5 絵文字カテゴリ一致率（オプション）

Unicodeの絵文字カテゴリ（Smileys, Animals, Food等）レベルでの一致を評価。

```python
import emoji

def get_emoji_category(e: str) -> str:
    """絵文字のカテゴリを取得"""
    data = emoji.EMOJI_DATA.get(e, {})
    # カテゴリ情報を取得（ライブラリのバージョンにより異なる）
    return data.get("group", "unknown")

def category_accuracy(pred: set, gold: set) -> float:
    """カテゴリレベルの一致率"""
    pred_categories = {get_emoji_category(e) for e in pred}
    gold_categories = {get_emoji_category(e) for e in gold}

    if not gold_categories:
        return 1.0 if not pred_categories else 0.0

    return len(pred_categories & gold_categories) / len(gold_categories)
```

## 3. 人手評価

### 3.1 評価設計

| 項目 | 設定 |
|------|------|
| 評価対象 | テストセットからランダム抽出した50〜100文 |
| 評価者 | 3〜5名の日本語話者 |
| 評価方式 | 教師出力と学生出力を並列表示し、それぞれ独立に評価 |

### 3.2 評価項目

#### (1) 意味的一致度（0〜4段階）

絵文字が入力文の意味・ニュアンスをどの程度表現しているか。

| スコア | 説明 |
|--------|------|
| 0 | ほとんど関係がない |
| 1 | 部分的に関連しているが不自然 |
| 2 | 一応意味は通るが不十分 |
| 3 | 概ね妥当 |
| 4 | 非常に妥当で自然 |

#### (2) 自然さ（0〜4段階）

実際のSNSで見かけそうかどうかの主観評価。

| スコア | 説明 |
|--------|------|
| 0 | 全く不自然、違和感がある |
| 1 | やや不自然 |
| 2 | 普通 |
| 3 | 自然 |
| 4 | 非常に自然、よく見かける使い方 |

#### (3) 誤解を招く可能性（Yes/No）

絵文字の選択が、元の文の意図と逆の印象を与えそうかどうか。

例:

- 入力: 「残念だった」 → 出力: 😊🎉 → **Yes**（誤解を招く）
- 入力: 「残念だった」 → 出力: 😢 → **No**

### 3.3 評価フォーマット

```text
=== サンプル #1 ===
入力文: 今日のライブ最高だった

【教師出力】 🎉 🎵 ✨
意味的一致度: [0-4] ___
自然さ: [0-4] ___
誤解の可能性: [Yes/No] ___
コメント: _______________

【学生出力】 😊 🎶
意味的一致度: [0-4] ___
自然さ: [0-4] ___
誤解の可能性: [Yes/No] ___
コメント: _______________
```

### 3.4 評価者間一致度

Krippendorff's alpha または Cohen's kappa で評価者間の一致度を報告。

```python
from sklearn.metrics import cohen_kappa_score

def inter_rater_agreement(ratings_a: list, ratings_b: list) -> float:
    """評価者間一致度（Cohen's kappa）"""
    return cohen_kappa_score(ratings_a, ratings_b)
```

### 3.5 実施計画（v4データセット）

v4データセットでの学習実験完了後、以下の計画で人手評価を実施する。

#### 3.5.1 評価対象モデル

| モデル | Jaccard | 多様性 | 選定理由 |
|--------|---------|--------|----------|
| v4_focal_top50 | 0.182 | 14% | 精度最良 |
| v4_top50 | 0.165 | 21% | バランス型 |
| 教師モデル（Qwen3） | - | - | 比較基準（Gold） |

#### 3.5.2 評価サンプル

| 項目 | 設定 |
|------|------|
| 件数 | 20〜50件 |
| 抽出元 | v4_top50テストセット または 任意のテキスト |
| 抽出条件 | 両モデルで予測可能なサンプル |
| 保存先 | `outputs/human_eval/samples.jsonl` |

**現状**: モデル推論機能実装済み。50件の評価サンプル生成が可能。

**実装済みの機能**:

1. **HuggingFace Hubからのモデルロード** (`src/models/t5_trainer.py`)

   ```python
   from src.models.t5_trainer import load_model_from_hub, generate_emoji
   tokenizer, model = load_model_from_hub("AtefAndrus/jmoji-t5-v4_focal_top50_20251224")
   result = generate_emoji(model, tokenizer, "今日は楽しかった")
   ```

2. **推論スクリプト** (`scripts/generate_predictions.py`)

   ```bash
   # 任意のテキストファイルから予測を生成
   uv run scripts/generate_predictions.py \
       --model AtefAndrus/jmoji-t5-v4_focal_top50_20251224 \
       --input texts.txt \
       --output predictions.jsonl
   ```

3. **人手評価サンプル生成** (`scripts/prepare_human_eval.py`)

   ```bash
   # HuggingFace Hubから推論して50件抽出
   uv run scripts/prepare_human_eval.py \
       --model-a-repo AtefAndrus/jmoji-t5-v4_focal_top50_20251224 \
       --model-b-repo AtefAndrus/jmoji-t5-v4_top50_20251224 \
       --input-file data/test.jsonl \
       --max-samples 50
   ```

4. **Colab推論ノートブック** (`notebooks/inference.py`)
   - インタラクティブ推論
   - バッチ推論（50件）
   - CSV/Markdownエクスポート

#### 3.5.3 評価フォーマット

Googleフォームを使用（集計自動化のため）。

**フォーム構成**:

1. サンプルID（自動記録）
2. 入力文（表示のみ）
3. 教師出力の評価（意味的一致度、自然さ、誤解の可能性）
4. 学生出力Aの評価（v4_focal_top50）
5. 学生出力Bの評価（v4_top50）
6. どちらが良いか（A/B/同等）
7. 自由コメント（任意）

#### 3.5.4 評価者

| 項目 | 設定 |
|------|------|
| 人数 | 1〜3名 |
| 条件 | 日本語ネイティブ、SNS利用経験あり |
| 所要時間 | 約30分（50件 × 30秒/件） |

#### 3.5.5 実施手順

```text
Step 1: 評価サンプル抽出
        └─ scripts/prepare_human_eval.py
        └─ 出力: outputs/human_eval/samples.jsonl

Step 2: 評価アプリ作成（完了）
        └─ /home/keigo/jmoji-human-eval/ に Gradio アプリを作成
        └─ HuggingFace Spaces にデプロイ

Step 3: 評価実施
        └─ 評価者に Space URL を共有
        └─ URL: https://huggingface.co/spaces/AtefAndrus/jmoji-human-eval
        └─ 回答は自動的に responses/ に保存

Step 4: 結果集計
        └─ scripts/analyze_human_eval.py
        └─ 出力: outputs/human_eval/results.json

Step 5: レポート作成
        └─ 定量評価との比較分析
        └─ docs/details/human_eval_results.md
```

**評価アプリの使い方**:

```bash
# ローカルで動作確認
cd /home/keigo/jmoji-human-eval
pip install -r requirements.txt
python app.py

# HuggingFace Spaces にデプロイ
huggingface-cli login
git remote add origin https://huggingface.co/spaces/AtefAndrus/jmoji-human-eval
git add . && git commit -m "Initial deployment" && git push -u origin main

# 結果集計（Jmojiリポジトリから実行）
uv run scripts/analyze_human_eval.py \
    --space-id AtefAndrus/jmoji-human-eval \
    --output outputs/human_eval/results.json \
    --report outputs/human_eval/report.md
```

#### 3.5.6 成果物

| ファイル | 内容 |
|----------|------|
| `outputs/human_eval/samples.jsonl` | 評価サンプル（20件） |
| `/home/keigo/jmoji-human-eval/responses/*.jsonl` | 評価者の回答（自動保存） |
| `outputs/human_eval/results.json` | 集計結果 |
| `docs/details/human_eval_results.md` | 分析レポート |

## 4. 評価パイプライン

### 4.1 自動評価の実行

```python
from pathlib import Path
import json

def evaluate_model(
    predictions_path: Path,
    gold_path: Path
) -> dict:
    """モデル評価を実行"""
    # データ読み込み
    predictions = load_jsonl(predictions_path)
    golds = load_jsonl(gold_path)

    # 絵文字集合に変換
    pred_sets = [set(p["emojis"]) for p in predictions]
    gold_sets = [set(g["emojis"]) for g in golds]

    # 各指標を計算
    results = {
        "jaccard": {
            "mean": np.mean([jaccard_similarity(p, g) for p, g in zip(pred_sets, gold_sets)]),
            "std": np.std([jaccard_similarity(p, g) for p, g in zip(pred_sets, gold_sets)])
        },
        "micro_f1": micro_f1(pred_sets, gold_sets),
        "exact_match": exact_match_rate(pred_sets, gold_sets),
        "length_analysis": length_distribution_analysis(
            [len(p) for p in pred_sets],
            [len(g) for g in gold_sets]
        )
    }

    return results
```

### 4.2 結果の可視化

```python
import matplotlib.pyplot as plt

def plot_jaccard_distribution(jaccards: list[float], save_path: Path):
    """Jaccard類似度の分布をプロット"""
    plt.figure(figsize=(10, 6))
    plt.hist(jaccards, bins=20, edgecolor="black", alpha=0.7)
    plt.xlabel("Jaccard Similarity")
    plt.ylabel("Frequency")
    plt.title("Distribution of Jaccard Similarity")
    plt.axvline(np.mean(jaccards), color="red", linestyle="--", label=f"Mean: {np.mean(jaccards):.3f}")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
```

## 5. エラー分析

### 5.1 分析カテゴリ

| カテゴリ | 説明 | 例 |
|---------|------|-----|
| 感情の誤判定 | ポジティブ/ネガティブの取り違え | 「残念」→ 😊 |
| 過剰生成 | 不必要な絵文字が多い | 5個制限超過 |
| 過少生成 | 絵文字が少なすぎる | 常に1個のみ |
| 文化依存 | 日本特有の表現の誤解 | 皮肉、謙遜 |
| 頻出偏り | 特定絵文字への過度な集中 | 常に😊 |

### 5.2 エラー収集

```python
def collect_error_samples(
    predictions: list[dict],
    golds: list[dict],
    threshold: float = 0.3
) -> list[dict]:
    """低スコアサンプルを収集"""
    errors = []

    for pred, gold in zip(predictions, golds):
        pred_set = set(pred["emojis"])
        gold_set = set(gold["emojis"])

        jaccard = jaccard_similarity(pred_set, gold_set)

        if jaccard < threshold:
            errors.append({
                "input": pred["sns_text"],
                "predicted": pred["emojis"],
                "gold": gold["emojis"],
                "jaccard": jaccard
            })

    return errors
```

## 6. 報告テンプレート

### 6.1 定量評価結果

```markdown
## 定量評価結果

| 指標 | スコア |
|------|--------|
| Jaccard類似度 | 0.XX ± 0.XX |
| Micro F1 | 0.XX |
| Exact Match Rate | XX.X% |
| 出力長相関 | 0.XX |
```

### 6.2 人手評価結果

```markdown
## 人手評価結果（N=XX文、評価者X名）

| モデル | 意味的一致度 | 自然さ | 誤解率 |
|--------|-------------|--------|--------|
| 教師LLM | X.XX ± X.XX | X.XX ± X.XX | XX% |
| 学生モデル | X.XX ± X.XX | X.XX ± X.XX | XX% |

評価者間一致度（κ）: 0.XX
```
