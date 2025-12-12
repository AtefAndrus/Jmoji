#!/usr/bin/env python3
"""データセットをHugging Face Hubにアップロードするスクリプト。

使用方法:
    # 環境変数でHFトークンを設定
    export HF_TOKEN="hf_..."

    # 全バージョンをアップロード
    uv run scripts/upload_dataset_to_hf.py

    # 特定バージョンのみ
    uv run scripts/upload_dataset_to_hf.py --versions v3

    # 公開リポジトリとして
    uv run scripts/upload_dataset_to_hf.py --public
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi, login


def load_jsonl_as_dataset(path: Path) -> Dataset:
    """JSONLファイルをHuggingFace Datasetとして読み込む。"""
    import json

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload datasets to HuggingFace Hub")
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["v1", "v2", "v3"],
        help="Dataset versions to upload (default: v1 v2 v3)",
    )
    parser.add_argument(
        "--repo-id",
        default="AtefAndrus/jmoji-dataset",
        help="HuggingFace repository ID",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make repository public (default: private)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/outputs",
        help="Directory containing dataset files",
    )
    args = parser.parse_args()

    # HFにログイン
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")
    login(token=token)

    api = HfApi()
    data_dir = Path(args.data_dir)

    # リポジトリ作成（存在しない場合）
    try:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=not args.public,
            exist_ok=True,
        )
        print(f"Repository: https://huggingface.co/datasets/{args.repo_id}")
    except Exception as e:
        print(f"Repository creation/check: {e}")

    # 各バージョンをアップロード
    for version in args.versions:
        jsonl_path = data_dir / f"dataset_{version}.jsonl"
        if not jsonl_path.exists():
            print(f"Skipping {version}: {jsonl_path} not found")
            continue

        print(f"\nUploading {version}...")

        # JSONLファイルを直接アップロード
        api.upload_file(
            path_or_fileobj=str(jsonl_path),
            path_in_repo=f"data/{version}.jsonl",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"  Uploaded: data/{version}.jsonl")

        # データセット統計を表示
        dataset = load_jsonl_as_dataset(jsonl_path)
        print(f"  Samples: {len(dataset)}")

    # README.mdを作成・アップロード
    readme_content = f"""---
license: mit
language:
- ja
tags:
- emoji
- text-to-emoji
- japanese
---

# Jmoji Dataset

日本語テキスト→絵文字翻訳のための疑似対訳データセット。

## データセット構成

| バージョン | サンプル数 | 説明 |
|------------|------------|------|
| v1 | 1,000 | 初期版 |
| v2 | ~5,000 | ✨禁止プロンプト適用 |
| v3 | 5,000 | 品質改善版（事前フィルタ・件数保証） |

## 使用方法

```python
from datasets import load_dataset

# 最新バージョン（v3）をロード
dataset = load_dataset("{args.repo_id}", data_files="data/v3.jsonl", split="train")

# 特定バージョンをロード
dataset_v1 = load_dataset("{args.repo_id}", data_files="data/v1.jsonl", split="train")
```

## データ形式

各サンプルは以下のフィールドを持つ:

- `original_text`: 元のWikipedia文
- `sns_text`: SNS風に変換されたテキスト
- `emoji_string`: 対応する絵文字（スペース区切り）

## 生成方法

Claude Haiku 4.5を教師モデルとして、Wikipedia日本語版から抽出した文をSNS風テキストに変換し、対応する絵文字を生成。

詳細: https://github.com/AtefAndrus/Jmoji
"""

    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print("\nREADME.md uploaded")

    print(
        f"\nDone! Dataset available at: https://huggingface.co/datasets/{args.repo_id}"
    )


if __name__ == "__main__":
    main()
