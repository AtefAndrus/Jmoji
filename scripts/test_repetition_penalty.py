#!/usr/bin/env python3
"""Repetition penaltyの効果をテストするスクリプト。"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# テスト用サンプル（過剰生成が見られたもの）
TEST_SAMPLES = [
    {
        "text": "自性分別ってのは、尋と伺のことで、分別の一種ってことね。",
        "gold": "🤔 🧠 📖 ✨ 🔍",
        "original_pred": "😊 😊 😊 😊 😊",  # v4_focal_top50
    },
    {
        "text": "港区の大会で上位入賞したりして、結構アピールしたわ〜",
        "gold": "🏆 🎉 👏 🇯🇵 💪",
        "original_pred": "📺 😊 😊 😊 😊",  # v4_focal_top50
    },
    {
        "text": "三木改造内閣の時に、自民党の党三役（幹事長、政調会長、総務会長）って、主流派じゃない「三木おろし」の中心だった挙党協には属してない人を閣僚に抜擢したんだって。",
        "gold": "🤔 🇯🇵 🏛️ 💬 🤝",
        "original_pred": "👏 👏 📖 👏 👏",  # v4_focal_top50
    },
    {
        "text": "大紫ってのは壬申の功臣の中でもけっこう上のクラスなんだけど、『書紀』の壬申の乱のとこ見ても星川麻呂の名前出てこないから、結局どんな活躍したのかはよくわかんないんだよな〜。",
        "gold": "🤔 📚 📖 🇯🇵 🔍",
        "original_pred": "📖 😊 🇯🇵 📚 👏",  # v4_focal_top50
    },
    {
        "text": "初のオールスターゲーム出場で、7月11日の第1戦（キャンドルスティック）の8回に代打で登場。マイク・フォーニレスからいきなり初打席本塁打！",
        "gold": "⚾ 🎉 👏 🔥 💥",
        "original_pred": "📚 📚 😊 📖 📚",  # v4_focal_top50
    },
]


def generate_with_penalty(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    text: str,
    repetition_penalty: float = 1.0,
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
    import sys

    print("=" * 60)
    print("Repetition Penalty テスト")
    print("=" * 60)

    # コマンドライン引数でモデルを切り替え
    if len(sys.argv) > 1 and sys.argv[1] == "top50":
        model_name = "AtefAndrus/jmoji-t5-v4_top50_20251224"
    else:
        model_name = "AtefAndrus/jmoji-t5-v4_focal_top50_20251224"

    print(f"\nモデルをロード中: {model_name}")

    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"デバイス: {device}")

    # テストするrepetition_penalty値
    penalties = [1.0, 1.2, 1.5, 2.0]

    print("\n" + "=" * 60)

    for i, sample in enumerate(TEST_SAMPLES, 1):
        print(f"\n### Sample {i}")
        print(f"Text: {sample['text'][:50]}...")
        print(f"Gold: {sample['gold']}")
        print(f"Original (penalty=1.0): {sample['original_pred']}")
        print()

        for penalty in penalties:
            pred = generate_with_penalty(
                model,
                tokenizer,
                sample["text"],
                repetition_penalty=penalty,
                device=device,
            )
            # 重複をカウント
            emojis = pred.split()
            unique_count = len(set(emojis))
            total_count = len(emojis)
            print(f"  penalty={penalty}: {pred} (unique: {unique_count}/{total_count})")

        print("-" * 40)

    print("\n" + "=" * 60)
    print("テスト完了")


if __name__ == "__main__":
    main()
