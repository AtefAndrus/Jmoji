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
# 依存関係インストール（pyproject.tomlから自動解決）
!pip install -q .

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
DATA_PATH = "/content/drive/MyDrive/school/ai_application/dataset_v3.jsonl"
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
    "early_stopping_patience": 5,
}

print("Config:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 3. データ準備

# %%
import sys
sys.path.append("/content/Jmoji")

from src.models.t5_trainer import (
    EmojiDataset,
    TrainConfig,
    setup_model_with_emoji_tokens,
    build_trainer,
    split_dataset,
    load_jsonl,
    generate_emoji,
    evaluate_model,
)

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
    print(f"\nWarning: '{top_emoji}' is {top_pct:.1f}% of all emojis. This may cause mode collapse.")

# %% [markdown]
# ## 4. モデル・トークナイザ準備

# %%
tokenizer, model = setup_model_with_emoji_tokens(CONFIG["model_name"])
print(f"Vocab size: {len(tokenizer)}")

# 絵文字トークン化確認
test_emoji = "😊 🎉"
ids = tokenizer.encode(test_emoji, add_special_tokens=False)
decoded = tokenizer.decode(ids)
print(f"Emoji tokenization test: '{test_emoji}' -> {ids} -> '{decoded}'")

# %% [markdown]
# ## 5. Dataset準備

# %%
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
import os

# 出力ディレクトリ作成
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# TrainConfigを構築
train_config = TrainConfig(
    model_name=CONFIG["model_name"],
    output_dir=OUTPUT_DIR,
    num_train_epochs=CONFIG["num_epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    learning_rate=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
    warmup_steps=CONFIG["warmup_steps"],
    logging_steps=CONFIG["logging_steps"],
    fp16=CONFIG["fp16"],
    label_smoothing_factor=CONFIG["label_smoothing"],
    early_stopping_patience=CONFIG["early_stopping_patience"],
    save_total_limit=3,
)

# Trainer構築
trainer = build_trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    cfg=train_config,
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
# テストセットで評価（全件）
print("=== テストセット評価 ===")
eval_results = evaluate_model(model, tokenizer, test_samples)
print(f"Average Jaccard: {eval_results.avg_jaccard:.4f}")
print(f"Exact Match Rate: {eval_results.exact_match_rate:.4f}")
print(f"Micro F1: {eval_results.micro_f1:.4f}")
print(f"Avg Precision: {eval_results.avg_precision:.4f}")
print(f"Avg Recall: {eval_results.avg_recall:.4f}")
print(f"Avg F1: {eval_results.avg_f1:.4f}")
print(f"Samples evaluated: {eval_results.num_samples}")

# %%
# 結果保存
import json
with open(f"{EVAL_DIR}/test_metrics.json", "w", encoding="utf-8") as f:
    json.dump({
        "avg_jaccard": eval_results.avg_jaccard,
        "exact_match_rate": eval_results.exact_match_rate,
        "micro_f1": eval_results.micro_f1,
        "avg_precision": eval_results.avg_precision,
        "avg_recall": eval_results.avg_recall,
        "avg_f1": eval_results.avg_f1,
        "num_samples": eval_results.num_samples,
    }, f, ensure_ascii=False, indent=2)

print(f"Metrics saved to {EVAL_DIR}/test_metrics.json")
