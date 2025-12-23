# 教師モデル移行: Claude Haiku 4.5 → Qwen3-235B-A22B

## 概要

本プロジェクトの教師モデルをClaude Haiku 4.5からQwen3-235B-A22Bへ移行する。

| 項目 | 変更前 | 変更後 |
|------|--------|--------|
| 教師モデル | Claude Haiku 4.5 | **Qwen3-235B-A22B** |
| プロバイダ | Anthropic (via OpenRouter) | Alibaba (via OpenRouter) |
| ライセンス | 競合モデル学習禁止 | **Apache 2.0（蒸留可）** |
| モデルID | `anthropic/claude-haiku-4.5` | `qwen/qwen3-235b-a22b-2507` |

---

## 移行理由

### 1. 規約上の問題（Claude Haiku 4.5）

Anthropicの利用規約において、以下の制限が存在する:

> "build a competing product or service, including to train competing AI models except as expressly approved by Anthropic"

本研究は知識蒸留によりT5モデルを学習するため、規約違反のリスクがあった。

### 2. Qwen3選定理由

1. **ライセンス**: Apache 2.0で蒸留を完全に許可
2. **日本語性能**: 119言語対応、日本語ベンチマーク（Shaberi等）で最高クラス
3. **コスト効率**: MoEアーキテクチャ（235B中22Bのみ活性化）で高性能・低コスト
4. **品質**: GPT-4o、DeepSeek-V3を上回る日本語推論精度（91.4%）

---

## コスト比較

### トークン単価

| モデル | 入力 $/1M tokens | 出力 $/1M tokens |
|--------|-----------------|-----------------|
| Claude Haiku 4.5 | $1.00 | $5.00 |
| **Qwen3-235B-A22B** | $0.18 | $0.54 |

### データセット生成コスト

基準: 1,000サンプル生成時のClaude Haiku 4.5実績 = **$0.68**

| モデル | 1,000サンプル | 5,000サンプル | 10,000サンプル |
|--------|-------------|--------------|---------------|
| Claude Haiku 4.5 | $0.68（実績） | $3.40 | $6.80 |
| **Qwen3-235B-A22B** | $0.23 | $1.17 | $2.34 |

Qwen3-235B-A22BはClaude Haiku 4.5の約1/3のコストとなる見込み。

---

## 検討した候補モデル

| モデル | 日本語性能 | ライセンス | 5,000サンプル | 選定 |
|--------|-----------|-----------|--------------|------|
| **Qwen3-235B-A22B** | 最高 | Apache 2.0 | $1.17 | **採用** |
| Qwen3-32B | 高い | Apache 2.0 | $0.52 | 次点 |
| DeepSeek V3.2 | 良好 | MIT | $1.00 | 候補 |
| Llama 4 Maverick | 良好 | Llama License* | $1.22 | 候補 |
| Claude Haiku 4.5 | 高い | 蒸留禁止 | $3.40 | 却下 |

*Llama 3.1以降は蒸留可能だが、生成モデル名に"Llama"を含める必要あり

---

## OpenRouter設定

### レート制限

| 制限種別 | 値 |
|---------|-----|
| 分次制限 | 20 RPM（requests per minute） |
| 日次制限 | なし（有料モデル使用時） |

### 推奨設定（`configs/default.yaml`）

```yaml
teacher:
  model: qwen/qwen3-235b-a22b-2507  # instruct版（thinkingモードなし）
  temperature: 0.2   # Shisa.AI推奨
  max_tokens: 100
  min_p: 0.1         # cross-lingual leakage抑制
  max_concurrent: 10  # 20 RPMの半分で安全マージン
  request_delay: 0.3  # 秒
```

> **注意**: `qwen/qwen3-235b-a22b`（サフィックスなし）はthinkingモードがデフォルトで有効であり、非ストリーミングでは空の応答を返す。`-2507`サフィックス付きのinstruct版を使用すること。

### 必要クレジット

- 最低チャージ: $5（OpenRouter最小単位）
- 推奨チャージ: $10〜$20

---

## プロンプト調整

Qwen3は日本語に強いが、中国語モデル由来のため以下に注意:

- `min_p=0.1`, `temp=0.2`でcross-lingual token leakageを抑制（Shisa.AI推奨）
- 出力言語を明示的に指定するプロンプトが有効な場合あり

現行プロンプト（`src/generation/prompts.py`）は基本的にそのまま使用可能。品質検証の結果、Shisa.AI推奨設定で十分な品質が確認された。

---

## 品質テスト結果（2025-12-19）

100サンプルで3つのパラメータ設定を比較検証した。

### テスト条件

- データソース: Wikipedia日本語版（20231101.ja）
- サンプル数: 100件
- 文字数制限: 10〜100文字
- 設定ファイル: `configs/experiment/test_qwen3.yaml`

### 比較結果

| 設定 | 成功率 | Cross-lingual Leakage | カジュアル表現率 | ユニーク絵文字数 |
|------|--------|----------------------|-----------------|-----------------|
| デフォルト (temp=0.7) | 99.2% | 1件 | 86% | 152種類 |
| min_p=0.1のみ追加 | 100% | 1件 | 88% | 115種類 |
| **Shisa推奨 (temp=0.2, min_p=0.1)** | **100%** | **0件** | **89%** | 127種類 |

### 評価基準

- **成功率**: APIが有効な絵文字を返した割合
- **Cross-lingual Leakage**: 日本語文中に英語等の混入がある件数
- **カジュアル表現率**: SNS風の口語表現に変換できた割合

### 結論

Shisa.AI推奨設定（temp=0.2, min_p=0.1）を採用:

- Cross-lingual leakageが完全に抑制された
- カジュアル表現率が最も高い（89%）
- 成功率100%で安定

---

## 移行タスク

- [x] `configs/default.yaml`のモデルIDを`qwen/qwen3-235b-a22b-2507`に変更
- [x] OpenRouterにクレジットをチャージ（$10〜$20推奨）
- [x] 小規模テスト（100サンプル程度）でQwen3出力品質を確認 → **Shisa推奨設定で良好**
- [x] v4データセット生成（20k件）を実行 → **2025-12-19完了**、フィルタ除外1,309件

---

## 関連ドキュメント

- [dataset_generation_v3.md](dataset_generation_v3.md): v3データセット生成の詳細
- [experiment_v3_improvements.md](experiment_v3_improvements.md): 学習改善実験の結果
