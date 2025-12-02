from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import load_config
from src.models.t5_trainer import (
    EmojiDataset,
    TrainConfig,
    build_trainer,
    load_jsonl,
    setup_model_with_emoji_tokens,
    split_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default="data/outputs/dataset_v1.jsonl")
    args = parser.parse_args()

    cfg = load_config(args.config)
    training_cfg = cfg.get("training", {})

    samples = load_jsonl(Path(args.data))

    train_ratio = float(training_cfg.get("train_ratio", 0.8))
    val_ratio = float(training_cfg.get("val_ratio", 0.1))
    test_ratio = float(training_cfg.get("test_ratio", 0.1))
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    train_samples, val_samples, test_samples = split_dataset(samples, train_ratio, val_ratio)

    tokenizer, model = setup_model_with_emoji_tokens(training_cfg.get("model_name", "sonoisa/t5-base-japanese"))

    train_ds = EmojiDataset(train_samples, tokenizer, training_cfg.get("max_input_length", 128), training_cfg.get("max_output_length", 32))
    val_ds = EmojiDataset(val_samples, tokenizer, training_cfg.get("max_input_length", 128), training_cfg.get("max_output_length", 32))

    tcfg = TrainConfig(
        model_name=training_cfg.get("model_name", "sonoisa/t5-base-japanese"),
        output_dir=training_cfg.get("output_dir", "outputs/models"),
        num_train_epochs=int(training_cfg.get("num_epochs", 10)),
        per_device_train_batch_size=int(training_cfg.get("batch_size", 16)),
        per_device_eval_batch_size=int(training_cfg.get("batch_size", 16)),
        learning_rate=float(training_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        evaluation_strategy=training_cfg.get("evaluation_strategy", "epoch"),
        save_strategy=training_cfg.get("save_strategy", "epoch"),
        load_best_model_at_end=training_cfg.get("early_stopping_patience", 0) > 0,
        metric_for_best_model=training_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=training_cfg.get("greater_is_better", False),
        logging_steps=int(training_cfg.get("logging_steps", 100)),
        warmup_steps=int(training_cfg.get("warmup_steps", 500)),
        fp16=bool(training_cfg.get("fp16", True)),
    )

    trainer = build_trainer(model, tokenizer, train_ds, val_ds, tcfg)

    if torch.cuda.is_available():
        model.to("cuda")  # type: ignore[arg-type]

    trainer.train()
    trainer.save_model(tcfg.output_dir)
    tokenizer.save_pretrained(tcfg.output_dir)

    # 簡易評価（検証データのloss）
    eval_result = trainer.evaluate()
    output_eval_dir = Path(cfg.get("evaluation", {}).get("results_dir", "outputs/evaluation"))
    output_eval_dir.mkdir(parents=True, exist_ok=True)
    (output_eval_dir / "train_eval_results.txt").write_text(str(eval_result), encoding="utf-8")


if __name__ == "__main__":
    main()
