# 実装ガイド

## 1. 環境構築

### 1.1 ローカル環境

```bash
# リポジトリクローン
git clone https://github.com/AtefAndrus/Jmoji.git
cd Jmoji

# mise でツールを取得（Python 3.12 / uv latest）
mise install

# uv で依存関係同期（.venv と uv.lock を生成）
UV_CACHE_DIR=.uv-cache uv sync

# pip 互換の要件ファイルを書き出す場合
uv export --format requirements-txt > requirements.txt

# 必要なら .venv を有効化
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 環境変数設定
cp .env.example .env
# .env を編集
```

### 1.2 Google Colab

```python
# リポジトリクローン
!git clone https://github.com/AtefAndrus/Jmoji.git
%cd Jmoji

# uv をインストールして同期（Colab はシステム Python を利用）
!pip install -q uv
!uv sync --frozen

# 環境変数設定（Colabシークレット推奨）
import os
from google.colab import userdata
os.environ["OPENROUTER_API_KEY"] = userdata.get("OPENROUTER_API_KEY")

# src/ をインポート可能にする
import sys
sys.path.append("/content/Jmoji")
```

## 2. データパイプライン

> **詳細ドキュメント**: データセット生成の品質改善については [dataset_generation_v3.md](details/datasets/generation_v3.md) を参照。

### 2.1 Wikipedia データ取得

```python
from datasets import load_dataset

# 日本語Wikipedia（約1.4M記事）
ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train")

# サンプル確認
print(ds[0])
# {'id': '...', 'url': '...', 'title': '...', 'text': '...'}
```

### 2.2 文の抽出とフィルタリング

```python
import re

def extract_sentences(text: str, min_len: int = 10, max_len: int = 100) -> list[str]:
    """テキストから文を抽出"""
    # 文分割（簡易版）
    sentences = re.split(r'(?<=[。！？])', text)

    # フィルタリング
    filtered = []
    for s in sentences:
        s = s.strip()
        if min_len <= len(s) <= max_len:
            # 記号のみ、URLのみ等を除外
            if re.search(r'[ぁ-んァ-ン一-龥]', s):
                filtered.append(s)

    return filtered
```

### 2.3 テキスト正規化

```python
import unicodedata

def normalize_text(text: str) -> str:
    """テキストの正規化"""
    # NFKC正規化（全角英数→半角、半角カナ→全角等）
    text = unicodedata.normalize("NFKC", text)

    # 連続空白を単一に
    text = re.sub(r'\s+', ' ', text)

    # 前後の空白除去
    text = text.strip()

    return text
```

### 2.4 NSFWコンテンツフィルタ

WikipediaにはNSFW（性的・暴力的）な記事が存在し、Claude APIがこれらの処理を拒否する可能性がある。
事前フィルタリングでAPIコストを削減し、拒否率を監視する。

```python
from typing import Optional, Set

# デフォルトのNSFWキーワード
DEFAULT_NSFW_KEYWORDS: Set[str] = {
    "性行為", "性交", "ポルノ", "アダルト", "風俗",
    "売春", "淫行", "殺人", "虐殺", "拷問", "処刑", "惨殺",
}

def is_safe_sentence(text: str, keywords: Optional[Set[str]] = None) -> bool:
    """NSFWキーワードを含まないかチェック"""
    if keywords is None:
        keywords = DEFAULT_NSFW_KEYWORDS
    return not any(kw in text for kw in keywords)

def filter_safe_sentences(sentences: list[str], keywords: Optional[Set[str]] = None) -> list[str]:
    """NSFWキーワードを含む文をフィルタリング"""
    if keywords is None:
        keywords = DEFAULT_NSFW_KEYWORDS
    return [s for s in sentences if is_safe_sentence(s, keywords)]
```

設定ファイル（`configs/default.yaml`）でキーワードをカスタマイズ可能:

```yaml
data:
  nsfw_filter:
    enabled: true
    keywords:
      - "性行為"
      - "殺人"
      # ... 必要に応じて追加
```

### 2.5 文完全性フィルタ

Wikipediaの文分割では、半端な文（メタ情報、途中で切れた文、閉じ括弧で始まる文など）が混入する。
これらをClaudeに渡す前にフィルタリングすることで、API回答混入やSNS変換失敗を防ぐ。

```python
import re

def is_complete_sentence(text: str) -> tuple[bool, str]:
    """文として完全かどうかを判定"""
    # メタセクション（Wikipediaの構造情報）
    if re.match(r'^(関連項目|参考文献|外部リンク|脚注|出典|注釈)', text):
        return False, "meta_section"
    # 途中切れ（開き括弧で終わる）
    if re.search(r'[（(「『][^）)」』]{0,30}$', text):
        return False, "truncated"
    # 閉じ括弧で始まる（前の文脈がない）
    if re.match(r'^[」』）)]', text):
        return False, "orphan_close"
    # 句読点なし
    if not re.search(r'[。！？!?」』)]$', text):
        return False, "no_ending"
    return True, ""
```

設定ファイルで有効/無効を切り替え可能:

```yaml
data:
  complete_sentence_filter: true  # 半端な文を除外するか
  buffer_ratio: 1.3               # 件数保証のためのバッファ率
```

### 2.6 フィルタログ

フィルタで除外された文は `data/outputs/filtered_sentences.jsonl` に記録される:

```json
{"reason": "nsfw", "detail": "殺人", "text": "..."}
{"reason": "incomplete", "detail": "meta_section", "text": "関連項目 ..."}
{"reason": "incomplete", "detail": "truncated", "text": "『天才・たけしの..."}
```

## 3. 教師LLM呼び出し

> **教師モデルの履歴**
>
> - **v1〜v3**: Claude Haiku 4.5（OpenRouter経由）
> - **v4以降**: Qwen3-235B-A22B（OpenRouter経由）
>
> 移行理由: コスト削減（約1/3）と品質の維持。詳細は [teacher_model_migration.md](details/teacher_model_migration.md) を参照。

### 3.1 OpenRouter クライアント

```python
import os
import httpx
from typing import Optional

class OpenRouterClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen/qwen3-235b-a22b",  # v4以降
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.base_url = base_url
        self.client = httpx.Client(timeout=60.0)

    def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 100
    ) -> str:
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
```

### 3.2 プロンプトテンプレート

```python
# SNS風文体変換
SNS_CONVERSION_PROMPT = """以下の文章を、日本のSNS（X、LINE等）で投稿されるようなカジュアルな文体に変換してください。
意味は変えずに、話し言葉や口語表現を使ってください。
変換後の文章のみを出力し、それ以外は何も出力しないでください。

入力: {text}
出力:"""

# 絵文字生成
EMOJI_GENERATION_PROMPT = """以下の日本語文に対して、文末に付与するのに適切な絵文字を1〜5個選んでください。

【ルール】
- 絵文字のみを空白区切りで出力し、それ以外は何も出力しないでください
- ✨（キラキラ）は使用しないでください。他の絵文字で表現してください
- 文の具体的な内容（スポーツ、音楽、食べ物、動物など）に関連する絵文字を優先してください
- 感情を表す場合は顔の絵文字（😊😢😂😅🥺😭🤣など）を使ってください
- 日本のSNS（X、LINEなど）で自然に見える使い方を意識してください

入力: {text}
出力:"""
```

**注意**: 初期実験で✨（キラキラ）が全絵文字の18.6%を占め、mode collapseの原因となったため、プロンプトで✨の使用を禁止している。

### 3.3 レート制限対応

```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60)
)
def generate_with_retry(client: OpenRouterClient, prompt: str) -> str:
    """リトライ付きAPI呼び出し"""
    return client.complete(prompt)

def batch_generate(
    client: OpenRouterClient,
    texts: list[str],
    prompt_template: str,
    delay: float = 0.5
) -> list[str]:
    """バッチ生成（レート制限考慮）"""
    results = []
    for text in texts:
        prompt = prompt_template.format(text=text)
        result = generate_with_retry(client, prompt)
        results.append(result)
        time.sleep(delay)  # レート制限回避
    return results
```

### 3.4 API レート制限の詳細

#### Qwen3-235B-A22B のレート制限

OpenRouter経由でQwen3を使用する場合:

| 制限種別 | 値 |
|---------|-----|
| 分次制限 | 20 RPM（requests per minute） |
| 日次制限 | なし（有料モデル使用時） |

推奨設定: `max_concurrent: 10`, `request_delay: 0.3`

#### Anthropic API のレート制限（参考: v1〜v3で使用）

OpenRouter経由でClaude（Haiku等）を使用する場合、Anthropicのレート制限が適用される。

| 制限種別 | 説明 |
|---------|------|
| RPM | Requests per minute（1分あたりのリクエスト数） |
| ITPM | Input tokens per minute（1分あたりの入力トークン数） |
| OTPM | Output tokens per minute（1分あたりの出力トークン数） |

- **Tierベース**: 利用額に応じてTier 1〜4に自動昇格、制限が緩和される
- **トークンバケット**: 連続補充型アルゴリズム。固定リセットではなく徐々に回復
- **モデル別独立**: モデルごとに別々のレート制限が適用される

#### OpenRouter経由の場合

- OpenRouter自体は有料モデルにレート制限を設けない
- プロバイダ（Anthropic）のレート制限がそのまま適用される
- BYOK（Bring Your Own Key）の場合は自身のAnthropicアカウントの制限が適用

#### 429エラー時の対応

レスポンスヘッダーで制限状況を確認可能:

| ヘッダー | 説明 |
|---------|------|
| `retry-after` | 待機すべき秒数 |
| `anthropic-ratelimit-requests-remaining` | 残りリクエスト数 |
| `anthropic-ratelimit-tokens-remaining` | 残りトークン数 |

参考:

- [Anthropic Rate Limits](https://docs.anthropic.com/en/api/rate-limits)
- [OpenRouter Rate Limits](https://openrouter.ai/docs/api/reference/limits)

### 3.5 並列リクエスト（高速化）

順次処理では1リクエスト2-3秒かかる場合、1000サンプル×2回で約80分以上かかる。
並列化により大幅な高速化が可能。

#### 実装アプローチ

```python
import asyncio
import httpx

class AsyncOpenRouterClient:
    def __init__(self, api_key: str, model: str, max_concurrent: int = 5):
        self.api_key = api_key
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.client = httpx.AsyncClient(timeout=60.0)

    async def complete(self, prompt: str) -> str:
        async with self.semaphore:
            response = await self.client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            if response.status_code == 429:
                retry_after = int(response.headers.get("retry-after", 10))
                await asyncio.sleep(retry_after)
                return await self.complete(prompt)  # リトライ
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

async def batch_generate_async(
    client: AsyncOpenRouterClient,
    texts: list[str],
    prompt_template: str
) -> list[str]:
    tasks = [
        client.complete(prompt_template.format(text=t))
        for t in texts
    ]
    return await asyncio.gather(*tasks)
```

#### 推奨設定

| 並列度 | 用途 |
|-------|------|
| 5 | 控えめ（Tier 1向け） |
| 10 | 中程度（Tier 2-3向け） |
| 20+ | 高負荷（Tier 4+、要確認） |

429エラーが頻発する場合は並列度を下げる。

### 3.6 APIコスト実績

#### Qwen3-235B-A22B（v4以降の見積もり）

| サンプル数 | 推定コスト |
|-----------|-----------|
| 1,000 | $0.23 |
| 5,000 | $1.17 |
| 10,000 | $2.34 |

#### Claude Haiku 4.5（v1〜v3の実績）

| 項目 | 値 |
|------|-----|
| モデル | Claude Haiku 4.5 (via OpenRouter) |
| サンプル数 | 1,000 |
| リクエスト数 | 2,000（SNS変換 + 絵文字生成） |
| 総コスト | $0.682 |
| 1サンプルあたり | 約 $0.00068 |
| 1リクエストあたり | 約 $0.00034 |

**スケール見積もり（Claude Haiku 4.5）:**

- 10,000サンプル: 約 $6.8
- 100,000サンプル: 約 $68

Qwen3-235B-A22BはClaude Haiku 4.5の約1/3のコスト。

### 3.7 コンテンツポリシー拒否の検出

OpenRouter経由でLLMを使用する場合、NSFWコンテンツはコンテンツモデレーションにより拒否される場合がある。
エラーを検出してログに記録し、拒否率を監視する。

> **Note**: Claude（v1〜v3）は厳格なコンテンツモデレーションを持つ。Qwen3（v4以降）は比較的緩いが、プロバイダによっては制限がある場合がある。

```python
import httpx

def is_content_policy_error(error: Exception) -> bool:
    """APIコンテンツポリシー拒否かどうかを判定"""
    if isinstance(error, httpx.HTTPStatusError):
        # 403: コンテンツモデレーション違反
        if error.response.status_code == 403:
            return True
        # レスポンスにmoderation/content/flagged/policyキーワードがあるか確認
        try:
            body = error.response.text.lower()
            if any(kw in body for kw in ["moderation", "content", "flagged", "policy"]):
                return True
        except Exception:
            pass
    return False
```

**OpenRouterの403エラーレスポンス:**

```json
{
  "error": {
    "code": 403,
    "message": "Content moderation violation",
    "metadata": {
      "reasons": ["violence"],
      "flagged_input": "...",
      "provider_name": "anthropic"
    }
  }
}
```

## 4. 絵文字処理

### 4.1 絵文字リスト取得

```python
import emoji

def get_all_emojis() -> set[str]:
    """全絵文字のセットを取得"""
    return set(emoji.EMOJI_DATA.keys())

# 約3,700個の絵文字
all_emojis = get_all_emojis()
```

### 4.2 肌色バリアント正規化

```python
import re

# 肌色修飾子のパターン
SKIN_TONE_PATTERN = re.compile(r'[\U0001F3FB-\U0001F3FF]')

def normalize_skin_tone(text: str) -> str:
    """肌色バリアントを基本絵文字に統合"""
    return SKIN_TONE_PATTERN.sub('', text)

# 例
normalize_skin_tone("👋🏻")  # → "👋"
normalize_skin_tone("👨🏽‍💻")  # → "👨‍💻"
```

### 4.3 絵文字抽出

```python
def extract_emojis(text: str, max_count: int = 5) -> list[str]:
    """テキストから絵文字を抽出"""
    # 絵文字リストを取得
    emoji_list = emoji.emoji_list(text)

    # 絵文字のみ抽出
    emojis = [item['emoji'] for item in emoji_list]

    # 肌色正規化
    emojis = [normalize_skin_tone(e) for e in emojis]

    # 最大数で切り捨て
    return emojis[:max_count]

# 例
extract_emojis("楽しい😊🎉✨最高！")  # → ["😊", "🎉", "✨"]
```

## 5. データセット生成

### 5.1 生成パイプライン

```python
import json
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class DataSample:
    original_text: str      # 元のWikipedia文
    sns_text: str           # SNS風変換後
    emojis: list[str]       # 生成された絵文字
    emoji_string: str       # 空白区切り絵文字列

def generate_dataset(
    client: OpenRouterClient,
    sentences: list[str],
    output_path: Path,
    batch_size: int = 100
) -> list[DataSample]:
    """データセット生成"""
    samples = []

    for i, sentence in enumerate(tqdm(sentences)):
        try:
            # SNS風変換
            sns_text = client.complete(
                SNS_CONVERSION_PROMPT.format(text=sentence)
            ).strip()

            # 絵文字生成
            emoji_output = client.complete(
                EMOJI_GENERATION_PROMPT.format(text=sns_text)
            ).strip()

            # 絵文字抽出・検証
            emojis = extract_emojis(emoji_output)
            if not emojis:
                continue  # 絵文字がない場合はスキップ

            sample = DataSample(
                original_text=sentence,
                sns_text=sns_text,
                emojis=emojis,
                emoji_string=" ".join(emojis)
            )
            samples.append(sample)

            # 定期保存
            if (i + 1) % batch_size == 0:
                save_dataset(samples, output_path)

        except Exception as e:
            print(f"Error at {i}: {e}")
            continue

        time.sleep(0.5)  # レート制限

    save_dataset(samples, output_path)
    return samples

def save_dataset(samples: list[DataSample], path: Path):
    """JSONL形式で保存"""
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.__dict__, ensure_ascii=False) + "\n")
```

### 5.2 品質チェック

```python
def validate_sample(sample: DataSample) -> bool:
    """サンプルの品質チェック"""
    # 絵文字数チェック
    if not (1 <= len(sample.emojis) <= 5):
        return False

    # SNSテキストが空でないか
    if not sample.sns_text.strip():
        return False

    # 絵文字以外の文字が混入していないか
    for e in sample.emojis:
        if not emoji.is_emoji(e):
            return False

    return True
```

## 6. T5ファインチューニング

### 6.1 データセット準備

```python
from torch.utils.data import Dataset
from transformers import T5Tokenizer

class EmojiDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        tokenizer: T5Tokenizer,
        max_input_length: int = 128,
        max_output_length: int = 32
    ):
        self.samples = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def _load_data(self, path: Path) -> list[dict]:
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 入力: SNSテキスト
        input_text = sample["sns_text"]

        # 出力: 絵文字列
        output_text = sample["emoji_string"]

        # トークナイズ
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        output_encoding = self.tokenizer(
            output_text,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": output_encoding["input_ids"].squeeze()
        }
```

### 6.2 絵文字トークン追加

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

def setup_model_with_emoji_tokens(model_name: str = "sonoisa/t5-base-japanese"):
    """絵文字トークンを追加したモデルを準備"""
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # 絵文字を特殊トークンとして追加
    emoji_tokens = list(get_all_emojis())
    num_added = tokenizer.add_tokens(emoji_tokens)
    print(f"Added {num_added} emoji tokens")

    # 埋め込み層をリサイズ
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model
```

### 6.3 学習ループ

```python
from transformers import Trainer, TrainingArguments

def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str = "outputs/models"
):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=1e-3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=100,
        warmup_steps=500,
        fp16=True,  # A100では有効
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    return trainer
```

## 7. 推論

```python
def translate_to_emoji(
    model,
    tokenizer,
    text: str,
    max_length: int = 32,
    num_beams: int = 4
) -> str:
    """テキストから絵文字を生成"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        truncation=True
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# 使用例
emojis = translate_to_emoji(model, tokenizer, "今日は楽しかった")
print(emojis)  # 😊 🎉
```

## 8. トラブルシューティング

### OOMエラー（GPU メモリ不足）

```python
# バッチサイズを小さくする
training_args.per_device_train_batch_size = 8

# 勾配累積を使う
training_args.gradient_accumulation_steps = 2

# FP16を有効にする（A100では標準で有効）
training_args.fp16 = True
```

### 絵文字がOOVになる

```python
# トークナイザに絵文字を追加したか確認
print(tokenizer.encode("😊"))  # [絵文字のID, </s>]

# 追加されていない場合は再度追加
tokenizer.add_tokens(["😊", "🎉", ...])
model.resize_token_embeddings(len(tokenizer))
```

### API呼び出しエラー

```python
# タイムアウトを延長
client = httpx.Client(timeout=120.0)

# リトライ設定を調整
@retry(stop=stop_after_attempt(5), wait=wait_exponential(max=120))
def call_api(...):
    ...
```
