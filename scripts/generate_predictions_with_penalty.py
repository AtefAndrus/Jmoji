#!/usr/bin/env python3
"""Repetition penalty適用後の予測を生成するスクリプト。"""

import json

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def generate_with_penalty(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    text: str,
    repetition_penalty: float = 1.2,
    device: str = "cpu",
) -> str:
    """repetition_penaltyを適用して生成。"""
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=32,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            repetition_penalty=repetition_penalty,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    # 元の予測サンプルを読み込む
    input_file = "outputs/experiments/v4_top50_20251224/predictions_sample.jsonl"

    with open(input_file) as f:
        samples = [json.loads(line) for line in f]

    print(f"サンプル数: {len(samples)}")

    # モデルロード
    model_name = "AtefAndrus/jmoji-t5-v4_top50_20251224"
    print(f"モデルをロード中: {model_name}")

    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"デバイス: {device}")

    # 予測生成
    results = []
    for i, sample in enumerate(samples):
        pred = generate_with_penalty(
            model,
            tokenizer,
            sample["text"],
            repetition_penalty=1.2,
            device=device,
        )
        results.append(
            {
                "sample_id": i + 1,
                "text": sample["text"],
                "gold": sample["gold"],
                "pred_original": sample["pred"],
                "pred_with_penalty": pred,
            }
        )
        print(f"[{i+1}/{len(samples)}] {pred}")

    # 結果を保存
    output_file = "outputs/predictions_with_penalty.jsonl"
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n結果を保存: {output_file}")

    # 結果をJSON形式で出力（LLM評価用）
    print("\n=== LLM評価用データ ===")
    for r in results:
        print(
            f"Sample {r['sample_id']}: text={r['text'][:40]}... | gold={r['gold']} | pred={r['pred_with_penalty']}"
        )


if __name__ == "__main__":
    main()
