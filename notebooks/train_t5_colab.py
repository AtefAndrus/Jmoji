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
#
# 実験タイプを変更することで、異なる設定で学習を実行できる。
#
# | 実験タイプ | 説明 |
# |-----------|------|
# | baseline | ベースライン（lr=3e-4） |
# | lr1e-4 | 学習率調整（lr=1e-4） |
# | top100 | Top-100絵文字制限 |
# | lr1e-4_top100 | lr=1e-4 + Top-100制限 |
# | focal | Focal Loss（γ=2.0） |

# %%
from datetime import datetime

# =============================================================================
# 実験設定（ここを変更して実験を切り替える）
# =============================================================================
DATASET_VERSION = "v3"
EXPERIMENT_TYPE = "lr1e-4"  # baseline, lr1e-4, top100, lr1e-4_top100, focal, focal_top100

# =============================================================================
# 実験タイプに応じた設定の自動調整
# =============================================================================
EXPERIMENT_DATE = datetime.now().strftime("%Y%m%d")
EXPERIMENT_NAME = f"{DATASET_VERSION}_{EXPERIMENT_TYPE}_{EXPERIMENT_DATE}"

# パス設定
HF_DATASET_REPO = "AtefAndrus/jmoji-dataset"
OUTPUT_DIR = "/content/Jmoji/outputs/models"
EXP_DIR = f"/content/Jmoji/outputs/experiments/{EXPERIMENT_NAME}"
DRIVE_EXP_DIR = f"/content/drive/MyDrive/school/ai_application/experiments/{EXPERIMENT_NAME}"

# ベース設定
CONFIG = {
    "experiment_name": EXPERIMENT_NAME,
    "dataset_version": DATASET_VERSION,
    "experiment_type": EXPERIMENT_TYPE,
    "model_name": "sonoisa/t5-base-japanese",
    "num_epochs": 50,
    "batch_size": 16,
    "learning_rate": 3e-4,  # デフォルト
    "weight_decay": 0.01,
    "warmup_steps": 150,
    "max_input_length": 128,
    "max_output_length": 32,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "fp16": False,
    "logging_steps": 50,
    "label_smoothing": 0.1,
    "early_stopping_patience": 5,
    # 実験固有の設定
    "use_focal_loss": False,
    "focal_gamma": 2.0,
    "use_top100_filter": False,
}

# 実験タイプに応じた設定の上書き
if "lr1e-4" in EXPERIMENT_TYPE:
    CONFIG["learning_rate"] = 1e-4
    print("Setting: learning_rate = 1e-4")

if "top100" in EXPERIMENT_TYPE:
    CONFIG["use_top100_filter"] = True
    print("Setting: Top-100 emoji filter enabled")

if "focal" in EXPERIMENT_TYPE:
    CONFIG["use_focal_loss"] = True
    CONFIG["label_smoothing"] = 0.0  # Focal Lossと併用しない
    print("Setting: Focal Loss enabled (gamma=2.0)")

print(f"\nExperiment: {EXPERIMENT_NAME}")
print(f"Dataset: {HF_DATASET_REPO} ({DATASET_VERSION})")
print(f"Experiment dir: {EXP_DIR}")
print("\nConfig:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 3. データ準備

# %%
import sys
sys.path.append("/content/Jmoji")

from datasets import load_dataset

from src.models.t5_trainer import (
    EmojiDataset,
    TrainConfig,
    setup_model_with_emoji_tokens,
    build_trainer,
    build_focal_loss_trainer,
    ExperimentLoggingCallback,
    split_dataset,
    generate_emoji,
    evaluate_model,
)
from src.evaluation.metrics import (
    diversity_ratio,
    emoji_distribution,
    compute_emoji_stats,
)
from src.data.emoji_utils import filter_samples_by_top_emojis

# HuggingFace Hubからデータセットをロード
hf_dataset = load_dataset(
    HF_DATASET_REPO,
    data_files=f"data/{DATASET_VERSION}.jsonl",
    split="train",
)
samples = list(hf_dataset)
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
# 絵文字統計を計算（src/evaluation/metrics.pyの関数を使用）
emoji_counts, total_emoji_count, unique_emoji_count = compute_emoji_stats(samples)

print(f"Total emoji occurrences: {total_emoji_count}")
print(f"Unique emojis: {unique_emoji_count}")
print("\nTop 20 emojis:")
for emoji, count in emoji_counts.most_common(20):
    pct = count / total_emoji_count * 100
    print(f"  {emoji}: {count} ({pct:.1f}%)")

# 最頻出絵文字の割合を警告
top_emoji, top_count = emoji_counts.most_common(1)[0]
top_pct = top_count / total_emoji_count * 100
if top_pct > 15:
    print(f"\nWarning: '{top_emoji}' is {top_pct:.1f}% of all emojis. This may cause mode collapse.")

# Top-5絵文字を保存（評価時の多様性指標で使用）
TOP_5_EMOJIS = set(e for e, _ in emoji_counts.most_common(5))
print(f"\nTop 5 emojis (for diversity metric): {TOP_5_EMOJIS}")

# %% [markdown]
# ## 3.6 Top-100絵文字フィルタリング（オプション）
#
# `use_top100_filter=True` の場合、Top-100絵文字のみを含むサンプルに制限する。

# %%
if CONFIG["use_top100_filter"]:
    # フィルタリング（src/data/emoji_utils.pyの関数を使用）
    original_count = len(samples)
    samples, emoji_counts, top_100_emojis = filter_samples_by_top_emojis(
        samples, top_n=100
    )
    print(f"Top-100 emojis: {len(top_100_emojis)}")
    print(f"Filtered samples: {len(samples)} / {original_count} ({len(samples)/original_count*100:.1f}%)")
    print(f"Unique emojis after filter: {len(emoji_counts)}")

    # 分割を再実行
    train_samples, val_samples, test_samples = split_dataset(
        samples,
        CONFIG["train_ratio"],
        CONFIG["val_ratio"]
    )
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
else:
    print("Top-100 filter: DISABLED")

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
import yaml

# 出力ディレクトリ作成
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(DRIVE_EXP_DIR, exist_ok=True)

# 設定をYAMLで保存
config_with_metadata = {
    **CONFIG,
    "timestamp": datetime.now().isoformat(),
    "data_source": f"{HF_DATASET_REPO}/data/{DATASET_VERSION}.jsonl",
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "total_samples": len(samples),
    "train_samples": len(train_samples),
    "val_samples": len(val_samples),
    "test_samples": len(test_samples),
    "unique_emojis": len(emoji_counts),
}
with open(f"{EXP_DIR}/config.yaml", "w", encoding="utf-8") as f:
    yaml.dump(config_with_metadata, f, allow_unicode=True, default_flow_style=False)
print(f"Config saved to {EXP_DIR}/config.yaml")

# 学習ログを記録するCallback（src/models/t5_trainer.pyから使用）
logging_callback = ExperimentLoggingCallback(f"{EXP_DIR}/train_log.csv")

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

# Trainer構築（Focal Lossの有無で切り替え）
if CONFIG["use_focal_loss"]:
    print(f"Using FocalLossTrainer (gamma={CONFIG['focal_gamma']})")
    trainer = build_focal_loss_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        cfg=train_config,
        gamma=CONFIG["focal_gamma"],
    )
else:
    print("Using standard Trainer")
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        cfg=train_config,
    )
# カスタムCallbackを追加
trainer.add_callback(logging_callback)

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

# 学習結果を取得
train_result = trainer.state
best_epoch = train_result.best_metric if hasattr(train_result, 'best_metric') else None
print(f"Best model checkpoint: {train_result.best_model_checkpoint}")

# 評価結果保存
eval_result = trainer.evaluate()
print(f"\nEval results: {eval_result}")

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

# 多様性指標（Top-5絵文字以外の出力割合）
predictions = [d["pred"] for d in eval_results.details]
diversity = diversity_ratio(predictions, TOP_5_EMOJIS)
print(f"\n=== 多様性指標 ===")
print(f"Non-Top5 Ratio: {diversity['non_top_n_ratio']:.4f}")
print(f"Unique Emojis in Output: {diversity['unique_emojis']}")
print(f"Top5 Count: {diversity['top_n_count']}, Non-Top5 Count: {diversity['non_top_n_count']}")

# 出力絵文字の分布（Top 10）
pred_dist = emoji_distribution(predictions)
print(f"\n=== 出力絵文字の分布（Top 10） ===")
for i, (emoji, count) in enumerate(list(pred_dist.items())[:10]):
    print(f"  {emoji}: {count}")

# %%
# 結果保存
import json
import shutil

eval_metrics = {
    "avg_jaccard": eval_results.avg_jaccard,
    "exact_match_rate": eval_results.exact_match_rate,
    "micro_f1": eval_results.micro_f1,
    "avg_precision": eval_results.avg_precision,
    "avg_recall": eval_results.avg_recall,
    "avg_f1": eval_results.avg_f1,
    "num_samples": eval_results.num_samples,
    # 多様性指標
    "diversity_non_top5_ratio": diversity["non_top_n_ratio"],
    "diversity_unique_emojis": diversity["unique_emojis"],
    "diversity_top5_count": diversity["top_n_count"],
    "diversity_non_top5_count": diversity["non_top_n_count"],
}

with open(f"{EXP_DIR}/eval_metrics.json", "w", encoding="utf-8") as f:
    json.dump(eval_metrics, f, ensure_ascii=False, indent=2)
print(f"Metrics saved to {EXP_DIR}/eval_metrics.json")

# 予測サンプルを保存（最初の20件）
with open(f"{EXP_DIR}/predictions_sample.jsonl", "w", encoding="utf-8") as f:
    for detail in eval_results.details[:20]:
        f.write(json.dumps(detail, ensure_ascii=False) + "\n")
print(f"Prediction samples saved to {EXP_DIR}/predictions_sample.jsonl")

# %% [markdown]
# ## 9. 実験サマリー生成

# %%
# サマリーMarkdown生成
summary_md = f"""# Experiment: {EXPERIMENT_NAME}

## Overview
- **Dataset**: {DATASET_VERSION} ({len(samples)} samples)
- **Experiment Type**: {EXPERIMENT_TYPE}
- **Date**: {EXPERIMENT_DATE}
- **GPU**: {config_with_metadata.get('gpu_name', 'Unknown')}

## Configuration
| Parameter | Value |
|-----------|-------|
| Model | {CONFIG['model_name']} |
| Epochs | {CONFIG['num_epochs']} |
| Batch Size | {CONFIG['batch_size']} |
| Learning Rate | {CONFIG['learning_rate']} |
| Label Smoothing | {CONFIG['label_smoothing']} |
| Early Stopping | {CONFIG['early_stopping_patience']} epochs |
| FP16 | {CONFIG['fp16']} |

## Data Split
- Train: {len(train_samples)}
- Validation: {len(val_samples)}
- Test: {len(test_samples)}
- Unique Emojis: {len(emoji_counts)}

## Results
| Metric | Value |
|--------|-------|
| Average Jaccard | {eval_results.avg_jaccard:.4f} |
| Exact Match Rate | {eval_results.exact_match_rate:.4f} |
| Micro F1 | {eval_results.micro_f1:.4f} |
| Avg Precision | {eval_results.avg_precision:.4f} |
| Avg Recall | {eval_results.avg_recall:.4f} |
| Avg F1 | {eval_results.avg_f1:.4f} |

## Diversity Metrics
| Metric | Value |
|--------|-------|
| Non-Top5 Ratio | {diversity['non_top_n_ratio']:.4f} |
| Unique Emojis in Output | {diversity['unique_emojis']} |
| Top5 Count | {diversity['top_n_count']} |
| Non-Top5 Count | {diversity['non_top_n_count']} |

## Training Info
- Best Checkpoint: {train_result.best_model_checkpoint}
- Final Eval Loss: {eval_result.get('eval_loss', 'N/A')}

## Notes
<!-- 実験に関するメモをここに記載 -->

"""

with open(f"{EXP_DIR}/summary.md", "w", encoding="utf-8") as f:
    f.write(summary_md)
print(f"Summary saved to {EXP_DIR}/summary.md")

# %%
# Google Driveにコピー
shutil.copytree(EXP_DIR, DRIVE_EXP_DIR, dirs_exist_ok=True)
print(f"\nExperiment files copied to Google Drive: {DRIVE_EXP_DIR}")
print("\nFiles saved:")
for fname in os.listdir(EXP_DIR):
    print(f"  - {fname}")

# %% [markdown]
# ## 10. GitHubに実験ログをコミット
#
# Colab Secretsに `GITHUB_TOKEN` を設定しておく必要がある。
# GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens
# - Repository access: AtefAndrus/Jmoji のみ
# - Permissions: Contents (Read and write)

# %%
from google.colab import userdata

try:
    GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
    REPO_URL = f"https://{GITHUB_TOKEN}@github.com/AtefAndrus/Jmoji.git"

    # git設定
    !git config user.name "AtefAndrus"
    !git config user.email "77284388+AtefAndrus@users.noreply.github.com"

    # 実験ログをコミット・プッシュ
    !git add outputs/experiments/{EXPERIMENT_NAME}/
    !git commit -m "[experiment] {EXPERIMENT_NAME}"
    !git push {REPO_URL} main

    print(f"\nExperiment {EXPERIMENT_NAME} pushed to GitHub")
except userdata.SecretNotFoundError:
    print("GITHUB_TOKEN not found in Colab Secrets. Skipping auto-commit.")
    print("To enable auto-commit, add GITHUB_TOKEN to Colab Secrets.")

# %% [markdown]
# ## 11. Hugging Face Hubにモデルをアップロード
#
# Colab Secretsに `HF_TOKEN` を設定しておく必要がある。
# https://huggingface.co/settings/tokens → Create new token (Write権限)

# %%
from huggingface_hub import login, HfApi

try:
    HF_TOKEN = userdata.get('HF_TOKEN')
    login(token=HF_TOKEN)

    # リポジトリ名: jmoji-t5-{experiment_name}
    repo_name = f"jmoji-t5-{EXPERIMENT_NAME}"
    repo_id = f"AtefAndrus/{repo_name}"

    print(f"Uploading model to Hugging Face Hub: {repo_id}")

    # モデルとトークナイザをアップロード（privateリポジトリ）
    model.push_to_hub(repo_id, private=True)
    tokenizer.push_to_hub(repo_id, private=True)

    # 実験設定もアップロード
    api = HfApi()
    api.upload_file(
        path_or_fileobj=f"{EXP_DIR}/config.yaml",
        path_in_repo="experiment_config.yaml",
        repo_id=repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=f"{EXP_DIR}/eval_metrics.json",
        path_in_repo="eval_metrics.json",
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"\nModel uploaded to: https://huggingface.co/{repo_id}")

except userdata.SecretNotFoundError:
    print("HF_TOKEN not found in Colab Secrets. Skipping HF Hub upload.")
    print("To enable upload, add HF_TOKEN (Write permission) to Colab Secrets.")
except Exception as e:
    print(f"Failed to upload to HF Hub: {e}")
