# %% [markdown]
# # Jmoji T5 Inference Notebook
#
# HuggingFace Hubから学習済みモデルをロードして絵文字予測を行うノートブック

# %% [markdown]
# ## 1. 環境セットアップ

# %%
# Google Driveをマウント（結果保存用）
from google.colab import drive
drive.mount('/content/drive')

# %%
# リポジトリクローン
!git clone https://github.com/AtefAndrus/Jmoji.git
%cd /content/Jmoji

# %%
# 依存関係インストール
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
#
# 使用するモデルを選択する。
#
# | モデル | Jaccard | 多様性 | 特徴 |
# |--------|---------|--------|------|
# | v4_focal_top50 | 0.182 | 14% | 精度最良 |
# | v4_top50 | 0.165 | 21% | バランス型 |
# | v4_focal_top100 | 0.115 | 25% | 多様性最良 |
# | v4_top100 | 0.120 | 21% | 標準 |

# %%
import os

# =============================================================================
# 設定（ここを変更してモデルを切り替える）
# =============================================================================

# HuggingFace Hub認証トークン（プライベートリポジトリ用）
# Colab Secrets または直接入力
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get('HF_TOKEN')
except:
    HF_TOKEN = None  # 環境変数から取得される

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    print("HF_TOKEN loaded from Colab Secrets")
else:
    print("Warning: HF_TOKEN not found. Set it in Colab Secrets or environment variable.")

# モデル選択
MODEL_A_REPO = "AtefAndrus/jmoji-t5-v4_focal_top50_20251224"  # 精度重視
MODEL_B_REPO = "AtefAndrus/jmoji-t5-v4_top50_20251224"        # バランス型

# 出力設定
OUTPUT_DIR = "/content/Jmoji/outputs/human_eval"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/school/ai_application/human_eval"

# %% [markdown]
# ## 3. モデルロード

# %%
from src.models.t5_trainer import load_model_from_hub, generate_emoji

print("Loading Model A (focal_top50)...")
tokenizer_a, model_a = load_model_from_hub(MODEL_A_REPO)
device = str(next(model_a.parameters()).device)
print(f"Model A loaded on {device}")

print("\nLoading Model B (top50)...")
tokenizer_b, model_b = load_model_from_hub(MODEL_B_REPO)
print(f"Model B loaded on {device}")

# %% [markdown]
# ## 4. インタラクティブ推論
#
# 任意のテキストを入力して絵文字を予測する。

# %%
def predict_emoji(text: str, use_model_a: bool = True, use_sampling: bool = True) -> str:
    """テキストから絵文字を予測する。"""
    if use_model_a:
        return generate_emoji(model_a, tokenizer_a, text, use_sampling=use_sampling, device=device)
    else:
        return generate_emoji(model_b, tokenizer_b, text, use_sampling=use_sampling, device=device)

def compare_models(text: str) -> None:
    """2つのモデルの予測を比較する。"""
    pred_a = predict_emoji(text, use_model_a=True)
    pred_b = predict_emoji(text, use_model_a=False)
    print(f"入力: {text}")
    print(f"Model A (focal_top50): {pred_a}")
    print(f"Model B (top50):       {pred_b}")
    print()

# %%
# テスト
test_texts = [
    "今日は楽しかった",
    "明日は雨らしい",
    "新しいプロジェクトを始めた",
    "美味しいラーメンを食べた",
    "友達と映画を見た",
    "仕事が忙しい",
]

print("=== モデル比較 ===\n")
for text in test_texts:
    compare_models(text)

# %% [markdown]
# ## 5. バッチ推論（テストセットから50件）
#
# 人手評価用のサンプルを生成する。

# %%
from datasets import load_dataset
import random
import json
from pathlib import Path

from src.evaluation.metrics import jaccard_similarity

# データセットロード
print("Loading dataset from HuggingFace Hub...")
dataset = load_dataset(
    "AtefAndrus/jmoji-dataset",
    data_files="data/v4.jsonl",
    split="train",
)
print(f"Total samples: {len(dataset)}")

# Top-50絵文字でフィルタリング（v4_top50モデルと同じ条件）
from collections import Counter

# 絵文字頻度を計算
emoji_counter = Counter()
for sample in dataset:
    emojis = sample["emoji_string"].split()
    emoji_counter.update(emojis)

top_50_emojis = set([emoji for emoji, _ in emoji_counter.most_common(50)])
print(f"Top 50 emojis: {len(top_50_emojis)}")

# Top-50のみを含むサンプルをフィルタ
def is_top50_only(sample):
    emojis = set(sample["emoji_string"].split())
    return emojis.issubset(top_50_emojis)

filtered_samples = [s for s in dataset if is_top50_only(s)]
print(f"Filtered samples (top50 only): {len(filtered_samples)}")

# %%
# サンプリング
MAX_SAMPLES = 50
SEED = 42

random.seed(SEED)
test_samples = random.sample(filtered_samples, min(MAX_SAMPLES, len(filtered_samples)))
print(f"Selected {len(test_samples)} samples for evaluation")

# %%
# 両モデルで推論
print("\n=== Model A 推論 ===")
results_a = []
for i, sample in enumerate(test_samples):
    text = sample["sns_text"]
    gold = sample["emoji_string"]
    pred = generate_emoji(model_a, tokenizer_a, text, use_sampling=True, device=device)

    gold_set = set(gold.split())
    pred_set = set(pred.split())
    jacc = jaccard_similarity(pred_set, gold_set)

    results_a.append({
        "text": text,
        "gold": gold,
        "pred": pred,
        "jaccard": jacc,
    })

    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(test_samples)} samples...")

print(f"Model A completed. Avg Jaccard: {sum(r['jaccard'] for r in results_a) / len(results_a):.3f}")

# %%
print("\n=== Model B 推論 ===")
results_b = []
for i, sample in enumerate(test_samples):
    text = sample["sns_text"]
    gold = sample["emoji_string"]
    pred = generate_emoji(model_b, tokenizer_b, text, use_sampling=True, device=device)

    gold_set = set(gold.split())
    pred_set = set(pred.split())
    jacc = jaccard_similarity(pred_set, gold_set)

    results_b.append({
        "text": text,
        "gold": gold,
        "pred": pred,
        "jaccard": jacc,
    })

    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(test_samples)} samples...")

print(f"Model B completed. Avg Jaccard: {sum(r['jaccard'] for r in results_b) / len(results_b):.3f}")

# %% [markdown]
# ## 6. 人手評価用CSVエクスポート

# %%
import csv

# マージ
merged_samples = []
for i, (a, b) in enumerate(zip(results_a, results_b)):
    merged_samples.append({
        "id": i + 1,
        "text": a["text"],
        "gold": a["gold"],
        "pred_focal_top50": a["pred"],
        "pred_top50": b["pred"],
        "jaccard_focal_top50": a["jaccard"],
        "jaccard_top50": b["jaccard"],
    })

# ディレクトリ作成
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# JSONL保存
jsonl_path = Path(OUTPUT_DIR) / "samples.jsonl"
with open(jsonl_path, "w", encoding="utf-8") as f:
    for sample in merged_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
print(f"Saved JSONL to {jsonl_path}")

# CSV保存（Googleフォーム用）
csv_path = Path(OUTPUT_DIR) / "samples.csv"
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["ID", "入力文", "教師出力（Gold）", "モデルA出力", "モデルB出力"])
    writer.writeheader()
    for s in merged_samples:
        writer.writerow({
            "ID": s["id"],
            "入力文": s["text"],
            "教師出力（Gold）": s["gold"],
            "モデルA出力": s["pred_focal_top50"],
            "モデルB出力": s["pred_top50"],
        })
print(f"Saved CSV to {csv_path}")

# Markdown保存
md_path = Path(OUTPUT_DIR) / "samples.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# 人手評価サンプル\n\n")
    f.write(f"サンプル数: {len(merged_samples)}件\n\n")
    f.write("---\n\n")
    for s in merged_samples:
        f.write(f"## サンプル #{s['id']}\n\n")
        f.write(f"**入力文**: {s['text']}\n\n")
        f.write(f"**教師出力（Gold）**: {s['gold']}\n\n")
        f.write(f"**モデルA（focal_top50）**: {s['pred_focal_top50']} (Jaccard: {s['jaccard_focal_top50']:.3f})\n\n")
        f.write(f"**モデルB（top50）**: {s['pred_top50']} (Jaccard: {s['jaccard_top50']:.3f})\n\n")
        f.write("---\n\n")
print(f"Saved Markdown to {md_path}")

# %%
# 統計情報
print("\n=== 統計情報 ===")
jaccard_a = [s["jaccard_focal_top50"] for s in merged_samples]
jaccard_b = [s["jaccard_top50"] for s in merged_samples]
print(f"Model A (focal_top50) 平均Jaccard: {sum(jaccard_a)/len(jaccard_a):.3f}")
print(f"Model B (top50) 平均Jaccard: {sum(jaccard_b)/len(jaccard_b):.3f}")

a_wins = sum(1 for a, b in zip(jaccard_a, jaccard_b) if a > b)
b_wins = sum(1 for a, b in zip(jaccard_a, jaccard_b) if b > a)
ties = sum(1 for a, b in zip(jaccard_a, jaccard_b) if a == b)
print(f"\nJaccard比較:")
print(f"  Model A wins: {a_wins}")
print(f"  Model B wins: {b_wins}")
print(f"  Ties: {ties}")

# %% [markdown]
# ## 7. Google Driveに保存

# %%
import shutil

# Driveにコピー
Path(DRIVE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
for filename in ["samples.jsonl", "samples.csv", "samples.md"]:
    src = Path(OUTPUT_DIR) / filename
    dst = Path(DRIVE_OUTPUT_DIR) / filename
    shutil.copy(src, dst)
    print(f"Copied to {dst}")

print("\nDone! Files saved to Google Drive.")

# %% [markdown]
# ## 8. カスタムテキスト推論
#
# 以下のセルでカスタムテキストを入力して推論できる。

# %%
# カスタムテキストを入力
custom_text = "今日は天気が良くて散歩した"  # ここを変更

print(f"入力: {custom_text}")
print(f"Model A (focal_top50): {predict_emoji(custom_text, use_model_a=True)}")
print(f"Model B (top50):       {predict_emoji(custom_text, use_model_a=False)}")
