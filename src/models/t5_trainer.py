from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from src.data.emoji_utils import get_all_emojis
from src.evaluation.metrics import (
    exact_match_rate,
    jaccard_similarity,
    micro_f1,
    set_based_metrics,
)


class EmojiDataset(Dataset):
    """JSONLから読み込んだサンプルをT5用テンソルに変換するDataset。

    samples: List[dict] を受け取り、"sns_text" -> 入力、"emoji_string" -> 出力
    """

    def __init__(
        self,
        samples: Sequence[Dict[str, Any]],
        tokenizer: Any,
        max_input_length: int = 128,
        max_output_length: int = 32,
    ) -> None:
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
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


@dataclass
class TrainConfig:
    """T5学習の設定。"""

    model_name: str
    output_dir: str
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    logging_steps: int = 100
    warmup_steps: int = 500
    fp16: bool = True
    # 追加設定
    label_smoothing_factor: float = 0.0
    early_stopping_patience: Optional[int] = None
    save_total_limit: Optional[int] = None
    report_to: str = "none"


def setup_model_with_emoji_tokens(
    model_name: str = "sonoisa/t5-base-japanese",
) -> Tuple[T5Tokenizer, T5ForConditionalGeneration]:
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    emoji_tokens = list(get_all_emojis())
    num_added = tokenizer.add_tokens(emoji_tokens)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


def load_model_from_hub(
    repo_id: str,
    token: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[T5Tokenizer, T5ForConditionalGeneration]:
    """HuggingFace Hubからモデルとトークナイザをロードする。

    Args:
        repo_id: HuggingFace Hubのリポジトリ ID
                 例: "AtefAndrus/jmoji-t5-v4_focal_top50_20251224"
        token: HuggingFaceトークン（プライベートリポジトリ用）。
               Noneの場合は環境変数 HF_TOKEN を使用。
        device: デバイス（None時は自動検出: cuda優先）

    Returns:
        Tuple[T5Tokenizer, T5ForConditionalGeneration]: トークナイザとモデル

    Raises:
        EnvironmentError: プライベートリポジトリでトークンが見つからない場合
        OSError: リポジトリが存在しない場合
    """
    # トークンの優先順位: 引数 > 環境変数
    if token is None:
        token = os.environ.get("HF_TOKEN")

    # デバイス自動検出
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # トークナイザとモデルをロード
    # Hubにはトークナイザも一緒に保存されている（絵文字トークン追加済み）
    tokenizer = T5Tokenizer.from_pretrained(repo_id, token=token, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(repo_id, token=token)

    # デバイスに移動
    model = model.to(device)

    return tokenizer, model


def build_trainer(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    cfg: TrainConfig,
) -> Trainer:
    """TrainConfigからTrainerを構築する。"""
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        logging_steps=cfg.logging_steps,
        warmup_steps=cfg.warmup_steps,
        fp16=cfg.fp16,
        label_smoothing_factor=cfg.label_smoothing_factor,
        save_total_limit=cfg.save_total_limit,
        report_to=cfg.report_to,
    )

    # DataCollator（パディングを-100に）
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    # Callbacks
    callbacks = []
    if cfg.early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)
        )

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks if callbacks else None,
    )


class FocalLossTrainer(Trainer):
    """Focal Lossを使用するカスタムTrainer。

    クラス不均衡問題に対処するため、簡単な例（高確率で正解）の損失を下げ、
    難しい例に集中して学習する。

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 1.0, **kwargs: Any) -> None:
        """FocalLossTrainerを初期化する。

        Args:
            gamma: Focusing parameter。大きいほど簡単な例の損失が小さくなる。
                   0.0でクロスエントロピーと同等。推奨値: 2.0
            alpha: Class weight。1.0で均等。
            **kwargs: Trainerの引数
        """
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, Any]:
        """Focal Lossを計算する。

        Args:
            model: T5モデル
            inputs: 入力辞書（input_ids, attention_mask, labels）
            return_outputs: Trueの場合、(loss, outputs)を返す
            num_items_in_batch: バッチ内のアイテム数（未使用）

        Returns:
            損失テンソル、またはreturn_outputs=Trueの場合は(loss, outputs)
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Flatten
        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)

        # クロスエントロピー（reduction='none'で各要素の損失を取得）
        ce_loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            reduction="none",
            ignore_index=-100,
        )

        # Focal Loss: FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
        # p_t = exp(-ce_loss) は正解クラスの確率
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        # ignore_index=-100の要素を除いて平均
        mask = labels_flat != -100
        loss = focal_loss[mask].mean() if mask.any() else focal_loss.mean()

        return (loss, outputs) if return_outputs else loss


def build_focal_loss_trainer(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    cfg: TrainConfig,
    gamma: float = 2.0,
    alpha: float = 1.0,
) -> FocalLossTrainer:
    """TrainConfigからFocalLossTrainerを構築する。"""
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        logging_steps=cfg.logging_steps,
        warmup_steps=cfg.warmup_steps,
        fp16=cfg.fp16,
        label_smoothing_factor=cfg.label_smoothing_factor,
        save_total_limit=cfg.save_total_limit,
        report_to=cfg.report_to,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    callbacks = []
    if cfg.early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)
        )

    return FocalLossTrainer(
        gamma=gamma,
        alpha=alpha,
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks if callbacks else None,
    )


class ExperimentLoggingCallback(TrainerCallback):
    """学習ログをCSVファイルに記録するCallback。

    学習中のloss, eval_loss, learning_rateをステップごとに記録し、
    学習終了時にCSVファイルとして保存する。
    """

    def __init__(self, log_path: str) -> None:
        """ExperimentLoggingCallbackを初期化する。

        Args:
            log_path: ログを保存するCSVファイルのパス
        """
        self.log_path = log_path
        self.logs: List[Dict[str, Any]] = []

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        """ログイベント時に呼ばれる。"""
        if logs:
            log_entry: Dict[str, Any] = {
                "step": state.global_step,
                "epoch": round(state.epoch, 2) if state.epoch else 0,
            }
            for key in ["loss", "eval_loss", "learning_rate"]:
                if key in logs:
                    log_entry[key] = logs[key]
            # step, epoch以外のデータがある場合のみ記録
            if len(log_entry) > 2:
                self.logs.append(log_entry)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """学習終了時に呼ばれる。ログをCSVに保存する。"""
        if self.logs:
            import csv

            # 全ログエントリのキーを集約（eval_lossなど後から追加されるキーに対応）
            all_keys: set[str] = set()
            for log in self.logs:
                all_keys.update(log.keys())
            fieldnames = sorted(all_keys)

            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.logs)


def split_dataset(
    samples: Sequence[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    *,
    shuffle: bool = True,
    seed: Optional[int] = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    data = list(samples)
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(data)
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]
    return train, val, test


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def generate_emoji(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    text: str,
    max_input_length: int = 128,
    max_output_length: int = 32,
    use_sampling: bool = True,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    num_beams: int = 4,
    device: Optional[str] = None,
) -> str:
    """テキストから絵文字を生成する。

    Args:
        model: T5モデル
        tokenizer: T5トークナイザ
        text: 入力テキスト
        max_input_length: 入力の最大長
        max_output_length: 出力の最大長
        use_sampling: Trueならtemperature sampling、Falseならbeam search
        temperature: sampling時の温度
        top_k: sampling時のtop-k
        top_p: sampling時のtop-p (nucleus sampling)
        num_beams: beam search時のビーム数
        device: デバイス（None時は自動検出）

    Returns:
        生成された絵文字文字列（スペース区切り）
    """
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        if use_sampling:
            outputs = model.generate(
                **inputs,
                max_length=max_output_length,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_length=max_output_length,
                num_beams=num_beams,
                early_stopping=True,
            )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


@dataclass
class EvaluationResult:
    """評価結果を格納するデータクラス。"""

    avg_jaccard: float
    exact_match_rate: float
    micro_f1: float
    avg_precision: float
    avg_recall: float
    avg_f1: float
    num_samples: int
    details: List[Dict[str, Any]]


def evaluate_model(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    samples: Sequence[Dict[str, Any]],
    max_samples: Optional[int] = None,
    use_sampling: bool = True,
    device: Optional[str] = None,
) -> EvaluationResult:
    """モデルを評価する。

    Args:
        model: T5モデル
        tokenizer: T5トークナイザ
        samples: 評価サンプル（sns_text, emoji_stringを含むdict）
        max_samples: 評価するサンプル数の上限（Noneなら全件）
        use_sampling: 生成時にsamplingを使用するか
        device: デバイス（None時は自動検出）

    Returns:
        EvaluationResult: 評価結果
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_samples = samples[:max_samples] if max_samples else samples
    details: List[Dict[str, Any]] = []
    all_pred_sets: List[set] = []
    all_gold_sets: List[set] = []

    for sample in eval_samples:
        text = sample["sns_text"]
        gold = sample["emoji_string"]
        pred = generate_emoji(
            model, tokenizer, text, use_sampling=use_sampling, device=device
        )

        gold_set = set(gold.split())
        pred_set = set(pred.split())

        all_pred_sets.append(pred_set)
        all_gold_sets.append(gold_set)

        jacc = jaccard_similarity(pred_set, gold_set)
        metrics = set_based_metrics(pred_set, gold_set)

        details.append(
            {
                "text": text,
                "gold": gold,
                "pred": pred,
                "jaccard": jacc,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "exact_match": gold_set == pred_set,
            }
        )

    # 集計
    n = len(details)
    avg_jaccard = sum(d["jaccard"] for d in details) / n if n > 0 else 0.0
    em_rate = exact_match_rate(all_pred_sets, all_gold_sets)
    mf1 = micro_f1(all_pred_sets, all_gold_sets)
    avg_precision = sum(d["precision"] for d in details) / n if n > 0 else 0.0
    avg_recall = sum(d["recall"] for d in details) / n if n > 0 else 0.0
    avg_f1 = sum(d["f1"] for d in details) / n if n > 0 else 0.0

    return EvaluationResult(
        avg_jaccard=avg_jaccard,
        exact_match_rate=em_rate,
        micro_f1=mf1,
        avg_precision=avg_precision,
        avg_recall=avg_recall,
        avg_f1=avg_f1,
        num_samples=n,
        details=details,
    )


__all__ = [
    "EmojiDataset",
    "TrainConfig",
    "setup_model_with_emoji_tokens",
    "load_model_from_hub",
    "build_trainer",
    "FocalLossTrainer",
    "build_focal_loss_trainer",
    "ExperimentLoggingCallback",
    "split_dataset",
    "load_jsonl",
    "generate_emoji",
    "evaluate_model",
    "EvaluationResult",
]
