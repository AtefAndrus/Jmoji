# %% [markdown]
# # Jmoji T5 Training on Google Colab
#
# 日本語テキスト→絵文字翻訳モデルの学習ノートブック

# %% [markdown]
# ## 1. 環境セットアップ

# %%
# Google Driveをマウント
from google.colab import drive
drive.mount('/content/drive')

# %%
# リポジトリクローン
!git clone https://github.com/AtefAndrus/Jmoji.git
%cd /content/Jmoji

# %%
# 依存関係インストール
!pip install -r requirements-colab.txt

# %%
# GPU確認
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## 2. 設定

# %%
# パス設定
DATA_PATH = "/content/drive/MyDrive/school/ai_application/dataset_v1.jsonl"
OUTPUT_DIR = "/content/Jmoji/outputs/models"
EVAL_DIR = "/content/Jmoji/outputs/evaluation"

# 学習設定
CONFIG = {
    "model_name": "sonoisa/t5-base-japanese",
    "num_epochs": 50,
    "batch_size": 16,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "warmup_steps": 150,
    "max_input_length": 128,
    "max_output_length": 32,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "fp16": False,  # NaN防止のためオフ
    "logging_steps": 50,
    "label_smoothing": 0.1,  # mode collapse対策
}

print("Config:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 3. データ準備

# %%
import sys
sys.path.append("/content/Jmoji")

import json
import random
from pathlib import Path

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def split_dataset(samples, train_ratio, val_ratio, seed=42):
    data = list(samples)
    random.seed(seed)
    random.shuffle(data)
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return data[:train_end], data[train_end:val_end], data[val_end:]

# データ読み込み
samples = load_jsonl(DATA_PATH)
print(f"Total samples: {len(samples)}")

# 分割
train_samples, val_samples, test_samples = split_dataset(
    samples,
    CONFIG["train_ratio"],
    CONFIG["val_ratio"]
)
print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

# サンプル確認
print("\nSample data:")
for i, s in enumerate(train_samples[:3]):
    print(f"  [{i}] {s['sns_text'][:50]}... -> {s['emoji_string']}")

# %% [markdown]
# ## 3.5 絵文字分布の確認

# %%
from collections import Counter

# 全サンプルの絵文字を集計
all_emojis = []
for sample in samples:
    emojis = sample["emoji_string"].split()
    all_emojis.extend(emojis)

# 頻度カウント
emoji_counts = Counter(all_emojis)
print(f"Total emoji occurrences: {len(all_emojis)}")
print(f"Unique emojis: {len(emoji_counts)}")
print("\nTop 20 emojis:")
for emoji, count in emoji_counts.most_common(20):
    pct = count / len(all_emojis) * 100
    print(f"  {emoji}: {count} ({pct:.1f}%)")

# 最頻出絵文字の割合を警告
top_emoji, top_count = emoji_counts.most_common(1)[0]
top_pct = top_count / len(all_emojis) * 100
if top_pct > 15:
    print(f"\n⚠️ Warning: '{top_emoji}' is {top_pct:.1f}% of all emojis. This may cause mode collapse.")

# %% [markdown]
# ## 4. モデル・トークナイザ準備

# %%
from transformers import T5Tokenizer, T5ForConditionalGeneration
from src.data.emoji_utils import get_all_emojis

def setup_model_with_emoji_tokens(model_name):
    """絵文字トークンを追加したモデルを準備"""
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # 絵文字を特殊トークンとして追加
    emoji_tokens = list(get_all_emojis())
    num_added = tokenizer.add_tokens(emoji_tokens)
    print(f"Added {num_added} emoji tokens")

    # 埋め込み層をリサイズ
    model.resize_token_embeddings(len(tokenizer))
    print(f"Vocab size: {len(tokenizer)}")

    return tokenizer, model

tokenizer, model = setup_model_with_emoji_tokens(CONFIG["model_name"])

# 絵文字トークン化確認
test_emoji = "😊 🎉"
ids = tokenizer.encode(test_emoji, add_special_tokens=False)
decoded = tokenizer.decode(ids)
print(f"Emoji tokenization test: '{test_emoji}' -> {ids} -> '{decoded}'")

# %% [markdown]
# ## 5. Dataset準備

# %%
from torch.utils.data import Dataset

class EmojiDataset(Dataset):
    def __init__(self, samples, tokenizer, max_input_length=128, max_output_length=32):
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_text = sample["sns_text"]
        output_text = sample["emoji_string"]

        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        output_encoding = self.tokenizer(
            output_text,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # パディングトークンを-100に置き換え（損失計算から除外）
        labels = output_encoding["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }

# Dataset作成
train_dataset = EmojiDataset(
    train_samples, tokenizer,
    CONFIG["max_input_length"],
    CONFIG["max_output_length"]
)
val_dataset = EmojiDataset(
    val_samples, tokenizer,
    CONFIG["max_input_length"],
    CONFIG["max_output_length"]
)
test_dataset = EmojiDataset(
    test_samples, tokenizer,
    CONFIG["max_input_length"],
    CONFIG["max_output_length"]
)

print(f"Train dataset: {len(train_dataset)}")
print(f"Val dataset: {len(val_dataset)}")
print(f"Test dataset: {len(test_dataset)}")

# データ確認
item = train_dataset[0]
print(f"\nSample item shapes:")
print(f"  input_ids: {item['input_ids'].shape}")
print(f"  attention_mask: {item['attention_mask'].shape}")
print(f"  labels: {item['labels'].shape}")
print(f"  Non -100 labels: {(item['labels'] != -100).sum().item()}")

# %% [markdown]
# ## 6. 学習

# %%
from transformers import Trainer, TrainingArguments
import os

# 出力ディレクトリ作成
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# TrainingArguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=CONFIG["num_epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    learning_rate=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=CONFIG["logging_steps"],
    warmup_steps=CONFIG["warmup_steps"],
    fp16=CONFIG["fp16"],
    label_smoothing_factor=CONFIG["label_smoothing"],  # mode collapse対策
    report_to="none",  # wandbを無効化
    save_total_limit=3,  # チェックポイント数を制限
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# GPU移動
if torch.cuda.is_available():
    model.to("cuda")

print("Starting training...")

# %%
# 学習実行
trainer.train()

# %%
# モデル保存
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")

# 評価結果保存
eval_result = trainer.evaluate()
print(f"\nEval results: {eval_result}")

with open(f"{EVAL_DIR}/train_eval_results.txt", "w") as f:
    f.write(str(eval_result))

# %% [markdown]
# ## 7. 推論テスト

# %%
def generate_emoji(model, tokenizer, text, max_length=32, use_sampling=True):
    """テキストから絵文字を生成"""
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        truncation=True
    )
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        if use_sampling:
            # Temperature sampling（多様性重視）
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                top_p=0.95,
            )
        else:
            # Beam search（精度重視）
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 学習データでテスト（暗記確認）
print("=== 学習データでのテスト（Sampling） ===")
for sample in train_samples[:5]:
    text = sample["sns_text"]
    expected = sample["emoji_string"]
    result = generate_emoji(model, tokenizer, text, use_sampling=True)
    match = "OK" if result.strip() == expected.strip() else "NG"
    print(f"[{match}] 入力: {text[:40]}...")
    print(f"     期待: {expected}")
    print(f"     出力: {result}")
    print()

# %%
# Beam search との比較
print("=== Beam Search vs Sampling 比較 ===")
for sample in train_samples[:3]:
    text = sample["sns_text"]
    expected = sample["emoji_string"]
    result_beam = generate_emoji(model, tokenizer, text, use_sampling=False)
    result_sample = generate_emoji(model, tokenizer, text, use_sampling=True)
    print(f"入力: {text[:40]}...")
    print(f"  期待: {expected}")
    print(f"  Beam: {result_beam}")
    print(f"  Sample: {result_sample}")
    print()

# %%
# 新規テキストでテスト
print("=== 新規テキストでのテスト ===")
test_texts = [
    "今日は楽しかった",
    "明日は雨らしい",
    "おなかすいた",
    "試験に合格した",
    "推しが尊い",
    "めっちゃ眠い",
]

for text in test_texts:
    result = generate_emoji(model, tokenizer, text, use_sampling=True)
    print(f"入力: {text}")
    print(f"出力: {result}")
    print()

# %% [markdown]
# ## 8. 評価指標

# %%
def jaccard_similarity(pred_set, gold_set):
    """Jaccard類似度を計算"""
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    intersection = len(pred_set & gold_set)
    union = len(pred_set | gold_set)
    return intersection / union

def evaluate_model(model, tokenizer, samples, max_samples=100):
    """モデルを評価"""
    results = []

    for sample in samples[:max_samples]:
        text = sample["sns_text"]
        gold = sample["emoji_string"]
        pred = generate_emoji(model, tokenizer, text)

        # 絵文字をセットに変換
        gold_set = set(gold.split())
        pred_set = set(pred.split())

        jaccard = jaccard_similarity(pred_set, gold_set)
        exact_match = 1 if gold_set == pred_set else 0

        results.append({
            "jaccard": jaccard,
            "exact_match": exact_match,
            "pred": pred,
            "gold": gold,
        })

    # 集計
    avg_jaccard = sum(r["jaccard"] for r in results) / len(results)
    exact_match_rate = sum(r["exact_match"] for r in results) / len(results)

    return {
        "avg_jaccard": avg_jaccard,
        "exact_match_rate": exact_match_rate,
        "num_samples": len(results),
        "details": results,
    }

# テストセットで評価
print("=== テストセット評価 ===")
eval_results = evaluate_model(model, tokenizer, test_samples)
print(f"Average Jaccard: {eval_results['avg_jaccard']:.4f}")
print(f"Exact Match Rate: {eval_results['exact_match_rate']:.4f}")
print(f"Samples evaluated: {eval_results['num_samples']}")

# 結果保存
import json
with open(f"{EVAL_DIR}/test_metrics.json", "w", encoding="utf-8") as f:
    json.dump({
        "avg_jaccard": eval_results["avg_jaccard"],
        "exact_match_rate": eval_results["exact_match_rate"],
        "num_samples": eval_results["num_samples"],
    }, f, ensure_ascii=False, indent=2)

# %% [markdown]
# ## 9. モデルをDriveに保存（オプション）

# %%
# Google Driveにモデルを保存
DRIVE_MODEL_PATH = "/content/drive/MyDrive/school/ai_application/jmoji_model"

import shutil
shutil.copytree(OUTPUT_DIR, DRIVE_MODEL_PATH, dirs_exist_ok=True)
print(f"Model copied to {DRIVE_MODEL_PATH}")
