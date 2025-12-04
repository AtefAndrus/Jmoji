from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.models.t5_trainer import (
    EmojiDataset,
    EvaluationResult,
    TrainConfig,
    build_trainer,
    evaluate_model,
    generate_emoji,
    load_jsonl,
    split_dataset,
)


class FakeTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        text,
        max_length=10,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ):
        length = min(len(text), max_length)
        ids = list(range(length)) + [0] * (max_length - length)
        attn = [1] * length + [0] * (max_length - length)
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.tensor([attn]),
        }


def test_emoji_dataset_shapes():
    samples = [
        {"sns_text": "今日は楽しい", "emoji_string": "😊 🎉"},
        {"sns_text": "明日は晴れ", "emoji_string": "☀️"},
    ]
    tok = FakeTokenizer()
    ds = EmojiDataset(samples, tok, max_input_length=8, max_output_length=4)
    item = ds[0]
    assert item["input_ids"].shape == (8,)
    assert item["attention_mask"].shape == (8,)
    assert item["labels"].shape == (4,)


def test_split_dataset():
    samples = [{"id": i} for i in range(10)]
    train, val, test = split_dataset(samples, 0.6, 0.2)
    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2


def test_split_dataset_shuffle_with_seed():
    """同じseedで同じ結果、異なるseedで異なる結果になることを確認"""
    samples = [{"id": i} for i in range(100)]

    train1, _, _ = split_dataset(samples, 0.8, 0.1, seed=42)
    train2, _, _ = split_dataset(samples, 0.8, 0.1, seed=42)
    train3, _, _ = split_dataset(samples, 0.8, 0.1, seed=123)

    assert train1 == train2  # 同じseedなら同じ結果
    assert train1 != train3  # 異なるseedなら異なる結果


def test_split_dataset_no_shuffle():
    """shuffle=Falseで順序が保持されることを確認"""
    samples = [{"id": i} for i in range(10)]
    train, val, test = split_dataset(samples, 0.6, 0.2, shuffle=False)

    assert [s["id"] for s in train] == [0, 1, 2, 3, 4, 5]
    assert [s["id"] for s in val] == [6, 7]
    assert [s["id"] for s in test] == [8, 9]


def test_load_jsonl(tmp_path: Path):
    """JSONLファイルを正しく読み込めることを確認"""
    jsonl_file = tmp_path / "test.jsonl"
    jsonl_file.write_text(
        '{"sns_text": "テスト1", "emoji_string": "😊"}\n'
        '{"sns_text": "テスト2", "emoji_string": "🎉"}\n',
        encoding="utf-8",
    )

    data = load_jsonl(jsonl_file)
    assert len(data) == 2
    assert data[0]["sns_text"] == "テスト1"
    assert data[1]["emoji_string"] == "🎉"


def test_load_jsonl_empty(tmp_path: Path):
    """空のJSONLファイルは空リストを返す"""
    jsonl_file = tmp_path / "empty.jsonl"
    jsonl_file.write_text("", encoding="utf-8")

    data = load_jsonl(jsonl_file)
    assert data == []


class TestTrainConfig:
    """TrainConfigのテスト"""

    def test_default_values(self):
        """デフォルト値が正しく設定されることを確認"""
        cfg = TrainConfig(model_name="test-model", output_dir="/tmp/test")
        assert cfg.model_name == "test-model"
        assert cfg.output_dir == "/tmp/test"
        assert cfg.num_train_epochs == 10
        assert cfg.fp16 is True
        assert cfg.label_smoothing_factor == 0.0
        assert cfg.early_stopping_patience is None
        assert cfg.save_total_limit is None
        assert cfg.report_to == "none"

    def test_custom_values(self):
        """カスタム値が正しく設定されることを確認"""
        cfg = TrainConfig(
            model_name="test-model",
            output_dir="/tmp/test",
            num_train_epochs=50,
            fp16=False,
            label_smoothing_factor=0.1,
            early_stopping_patience=5,
            save_total_limit=3,
        )
        assert cfg.num_train_epochs == 50
        assert cfg.fp16 is False
        assert cfg.label_smoothing_factor == 0.1
        assert cfg.early_stopping_patience == 5
        assert cfg.save_total_limit == 3


class TestEvaluationResult:
    """EvaluationResultのテスト"""

    def test_dataclass_fields(self):
        """全フィールドが正しく設定されることを確認"""
        result = EvaluationResult(
            avg_jaccard=0.5,
            exact_match_rate=0.1,
            micro_f1=0.4,
            avg_precision=0.6,
            avg_recall=0.7,
            avg_f1=0.65,
            num_samples=100,
            details=[{"text": "test"}],
        )
        assert result.avg_jaccard == 0.5
        assert result.exact_match_rate == 0.1
        assert result.micro_f1 == 0.4
        assert result.avg_precision == 0.6
        assert result.avg_recall == 0.7
        assert result.avg_f1 == 0.65
        assert result.num_samples == 100
        assert len(result.details) == 1


class TestGenerateEmoji:
    """generate_emoji関数のテスト"""

    def test_generate_emoji_sampling(self):
        """sampling modeでの生成が動作することを確認"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # モックの設定
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_model.generate.return_value = torch.tensor([[4, 5, 6]])
        mock_tokenizer.decode.return_value = "😊 🎉"

        result = generate_emoji(
            mock_model, mock_tokenizer, "テスト", use_sampling=True, device="cpu"
        )

        assert result == "😊 🎉"
        mock_model.eval.assert_called_once()
        mock_model.generate.assert_called_once()
        # sampling=Trueなのでdo_sample=Trueが渡される
        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["do_sample"] is True

    def test_generate_emoji_beam_search(self):
        """beam search modeでの生成が動作することを確認"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_model.generate.return_value = torch.tensor([[4, 5, 6]])
        mock_tokenizer.decode.return_value = "😊"

        result = generate_emoji(
            mock_model, mock_tokenizer, "テスト", use_sampling=False, device="cpu"
        )

        assert result == "😊"
        call_kwargs = mock_model.generate.call_args[1]
        assert "num_beams" in call_kwargs
        assert call_kwargs["num_beams"] == 4


class TestEvaluateModel:
    """evaluate_model関数のテスト"""

    def test_evaluate_model_basic(self):
        """基本的な評価が動作することを確認"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # generate_emojiをモック
        with patch("src.models.t5_trainer.generate_emoji") as mock_gen:
            mock_gen.side_effect = ["😊 🎉", "📚"]

            samples = [
                {"sns_text": "楽しい", "emoji_string": "😊 🎉"},
                {"sns_text": "勉強", "emoji_string": "📚 ✏️"},
            ]

            result = evaluate_model(mock_model, mock_tokenizer, samples, device="cpu")

            assert result.num_samples == 2
            assert len(result.details) == 2
            # 1つ目は完全一致
            assert result.details[0]["exact_match"] is True
            # 2つ目は部分一致
            assert result.details[1]["exact_match"] is False

    def test_evaluate_model_max_samples(self):
        """max_samplesで評価数を制限できることを確認"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("src.models.t5_trainer.generate_emoji") as mock_gen:
            mock_gen.return_value = "😊"

            samples = [
                {"sns_text": f"test{i}", "emoji_string": "😊"} for i in range(10)
            ]

            result = evaluate_model(
                mock_model, mock_tokenizer, samples, max_samples=3, device="cpu"
            )

            assert result.num_samples == 3
            assert mock_gen.call_count == 3

    def test_evaluate_model_empty_samples(self):
        """空のサンプルリストで評価した場合"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        result = evaluate_model(mock_model, mock_tokenizer, [], device="cpu")

        assert result.num_samples == 0
        assert result.avg_jaccard == 0.0
        assert result.details == []


class TestBuildTrainer:
    """build_trainer関数のテスト"""

    @pytest.fixture
    def mock_components(self):
        """テスト用のモックコンポーネント"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_train_dataset = MagicMock()
        mock_eval_dataset = MagicMock()
        return mock_model, mock_tokenizer, mock_train_dataset, mock_eval_dataset

    def test_build_trainer_without_early_stopping(self, mock_components, tmp_path):
        """EarlyStoppingなしでTrainerを構築"""
        model, tokenizer, train_ds, eval_ds = mock_components

        cfg = TrainConfig(
            model_name="test",
            output_dir=str(tmp_path),
            early_stopping_patience=None,
        )

        with patch("src.models.t5_trainer.Trainer") as mock_trainer_cls:
            with patch("src.models.t5_trainer.TrainingArguments"):
                with patch("src.models.t5_trainer.DataCollatorForSeq2Seq"):
                    build_trainer(model, tokenizer, train_ds, eval_ds, cfg)

                    # callbacksがNoneまたは空で呼ばれる
                    call_kwargs = mock_trainer_cls.call_args[1]
                    assert (
                        call_kwargs["callbacks"] is None
                        or call_kwargs["callbacks"] == []
                    )

    def test_build_trainer_with_early_stopping(self, mock_components, tmp_path):
        """EarlyStoppingありでTrainerを構築"""
        model, tokenizer, train_ds, eval_ds = mock_components

        cfg = TrainConfig(
            model_name="test",
            output_dir=str(tmp_path),
            early_stopping_patience=5,
        )

        with patch("src.models.t5_trainer.Trainer") as mock_trainer_cls:
            with patch("src.models.t5_trainer.TrainingArguments"):
                with patch("src.models.t5_trainer.DataCollatorForSeq2Seq"):
                    with patch(
                        "src.models.t5_trainer.EarlyStoppingCallback"
                    ) as mock_es:
                        build_trainer(model, tokenizer, train_ds, eval_ds, cfg)

                        # EarlyStoppingCallbackが作成される
                        mock_es.assert_called_once_with(early_stopping_patience=5)

                        # callbacksにEarlyStoppingCallbackが含まれる
                        call_kwargs = mock_trainer_cls.call_args[1]
                        assert call_kwargs["callbacks"] is not None
                        assert len(call_kwargs["callbacks"]) == 1
