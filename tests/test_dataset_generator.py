import asyncio
from pathlib import Path

import pytest

from src.generation.dataset_generator import (
    DataSample,
    GenerationStats,
    append_sample,
    count_existing_samples,
    generate_dataset,
    generate_dataset_async,
    load_dataset,
    save_dataset,
    validate_sample,
)


class FakeClient:
    """同期クライアントのモック"""

    def __init__(self, fail_indices: set[int] | None = None):
        self.sample_idx = 0  # 現在処理中のサンプルインデックス
        self.is_first_call = True  # そのサンプルの最初の呼び出しか
        self.fail_indices = fail_indices or set()

    def complete(self, prompt: str) -> str:
        if self.is_first_call:
            # SNS変換の呼び出し
            if self.sample_idx in self.fail_indices:
                self.sample_idx += 1
                # is_first_call は True のまま（次のサンプルの最初の呼び出し）
                raise RuntimeError("Fake error")
            result = f"SNS文{self.sample_idx}"
            self.is_first_call = False
        else:
            # 絵文字生成の呼び出し
            result = "😊 🎉"
            self.sample_idx += 1
            self.is_first_call = True
        return result


class FakeAsyncClient:
    """非同期クライアントのモック"""

    def __init__(self, fail_indices: set[int] | None = None):
        self.sample_count = 0
        self.fail_indices = fail_indices or set()
        self._lock = asyncio.Lock()
        self._call_state: dict[int, int] = {}  # sample_idx -> call count (0 or 1)

    async def complete(self, prompt: str) -> str:
        async with self._lock:
            # プロンプトからサンプルを特定（簡易的に）
            sample_idx = self.sample_count

            if sample_idx not in self._call_state:
                self._call_state[sample_idx] = 0
                # 最初の呼び出し（SNS変換）
                if sample_idx in self.fail_indices:
                    self.sample_count += 1
                    raise RuntimeError("Fake error")
                result = f"SNS文{sample_idx}"
            else:
                # 2回目の呼び出し（絵文字生成）
                self.sample_count += 1
                result = "😊 🎉"

            self._call_state[sample_idx] += 1
            return result


# =============================================================================
# GenerationStats テスト
# =============================================================================


def test_generation_stats_success_rate_zero():
    """total=0の場合、success_rateは0.0を返す"""
    stats = GenerationStats()
    assert stats.success_rate() == 0.0


def test_generation_stats_success_rate():
    """success_rateは正しい割合を返す"""
    stats = GenerationStats(total=10, success=8, skipped=1, errors=1)
    assert stats.success_rate() == 80.0


# =============================================================================
# validate_sample テスト
# =============================================================================


def test_validate_sample_valid():
    """有効なサンプルはTrueを返す"""
    sample = DataSample(
        original_text="元の文",
        sns_text="SNS文",
        emojis=["😊", "🎉"],
        emoji_string="😊 🎉",
    )
    assert validate_sample(sample) is True


def test_validate_sample_invalid_empty_sns_text():
    """SNSテキストが空の場合はFalseを返す"""
    sample = DataSample(
        original_text="元の文",
        sns_text="",
        emojis=["😊"],
        emoji_string="😊",
    )
    assert validate_sample(sample) is False


def test_validate_sample_invalid_emoji_count_zero():
    """絵文字が0個の場合はFalseを返す"""
    sample = DataSample(
        original_text="元の文",
        sns_text="SNS文",
        emojis=[],
        emoji_string="",
    )
    assert validate_sample(sample, min_count=1) is False


def test_validate_sample_invalid_emoji_count_exceeds_max():
    """絵文字が最大数を超える場合はFalseを返す"""
    sample = DataSample(
        original_text="元の文",
        sns_text="SNS文",
        emojis=["😊", "🎉", "✨", "💕", "🔥", "⭐"],
        emoji_string="😊 🎉 ✨ 💕 🔥 ⭐",
    )
    assert validate_sample(sample, max_count=5) is False


def test_validate_sample_invalid_non_emoji():
    """絵文字でない文字が含まれる場合はFalseを返す"""
    sample = DataSample(
        original_text="元の文",
        sns_text="SNS文",
        emojis=["😊", "abc"],
        emoji_string="😊 abc",
    )
    assert validate_sample(sample) is False


# =============================================================================
# save/load/append テスト
# =============================================================================


def test_save_and_load_dataset(tmp_path: Path):
    """save_datasetとload_datasetが正しく動作する"""
    samples = [
        DataSample("元1", "SNS1", ["😊"], "😊"),
        DataSample("元2", "SNS2", ["🎉", "✨"], "🎉 ✨"),
    ]
    path = tmp_path / "test.jsonl"

    save_dataset(samples, path)
    loaded = load_dataset(path)

    assert len(loaded) == 2
    assert loaded[0].original_text == "元1"
    assert loaded[1].emojis == ["🎉", "✨"]


def test_append_sample(tmp_path: Path):
    """append_sampleが正しく追記する"""
    path = tmp_path / "test.jsonl"

    sample1 = DataSample("元1", "SNS1", ["😊"], "😊")
    sample2 = DataSample("元2", "SNS2", ["🎉"], "🎉")

    append_sample(sample1, path)
    append_sample(sample2, path)

    loaded = load_dataset(path)
    assert len(loaded) == 2


def test_count_existing_samples(tmp_path: Path):
    """count_existing_samplesが正しくカウントする"""
    path = tmp_path / "test.jsonl"

    # ファイルが存在しない場合
    assert count_existing_samples(path) == 0

    # サンプルを追加
    samples = [
        DataSample("元1", "SNS1", ["😊"], "😊"),
        DataSample("元2", "SNS2", ["🎉"], "🎉"),
        DataSample("元3", "SNS3", ["✨"], "✨"),
    ]
    save_dataset(samples, path)

    assert count_existing_samples(path) == 3


def test_load_dataset_nonexistent_file(tmp_path: Path):
    """存在しないファイルを読み込むと空リストを返す"""
    path = tmp_path / "nonexistent.jsonl"
    loaded = load_dataset(path)
    assert loaded == []


# =============================================================================
# generate_dataset テスト（同期版）
# =============================================================================


def test_generate_dataset_creates_jsonl(tmp_path: Path):
    """generate_datasetがJSONLファイルを作成する"""
    client = FakeClient()
    sentences = ["今日は楽しい", "明日は晴れる"]
    out = tmp_path / "dataset.jsonl"

    samples = generate_dataset(
        client, sentences, output_path=out, request_delay=0, show_progress=False
    )
    assert len(samples) == 2
    assert out.exists()

    loaded = load_dataset(out)
    assert isinstance(loaded[0], DataSample)
    assert loaded[0].emojis == ["😊", "🎉"]


def test_generate_dataset_handles_errors(tmp_path: Path):
    """generate_datasetがエラーを適切に処理する"""
    client = FakeClient(fail_indices={1})  # 2番目のサンプルでエラー
    sentences = ["文1", "文2", "文3"]
    out = tmp_path / "dataset.jsonl"

    samples = generate_dataset(
        client, sentences, output_path=out, request_delay=0, show_progress=False
    )

    # エラーが発生した1件を除いて2件成功
    assert len(samples) == 2


def test_generate_dataset_resume(tmp_path: Path):
    """generate_datasetがresumeオプションで続きから再開する"""
    out = tmp_path / "dataset.jsonl"

    # 最初に2件生成
    client1 = FakeClient()
    samples1 = generate_dataset(
        client1,
        ["文1", "文2"],
        output_path=out,
        request_delay=0,
        show_progress=False,
        resume=False,
    )
    assert len(samples1) == 2

    # 追加で3件（resume=True、既存2件をスキップ）
    client2 = FakeClient()
    samples2 = generate_dataset(
        client2,
        ["文1", "文2", "文3", "文4", "文5"],
        output_path=out,
        request_delay=0,
        show_progress=False,
        resume=True,
    )

    # 既存2件 + 新規3件 = 5件
    assert len(samples2) == 5


def test_generate_dataset_no_resume(tmp_path: Path):
    """resume=Falseの場合、既存ファイルをクリアする"""
    out = tmp_path / "dataset.jsonl"

    # 最初に2件生成
    client1 = FakeClient()
    generate_dataset(
        client1,
        ["文1", "文2"],
        output_path=out,
        request_delay=0,
        show_progress=False,
        resume=False,
    )

    # resume=Falseで再実行
    client2 = FakeClient()
    samples2 = generate_dataset(
        client2,
        ["文A", "文B", "文C"],
        output_path=out,
        request_delay=0,
        show_progress=False,
        resume=False,
    )

    # 新規3件のみ
    assert len(samples2) == 3


# =============================================================================
# generate_dataset_async テスト（非同期版）
# =============================================================================


@pytest.mark.asyncio
async def test_generate_dataset_async_creates_jsonl(tmp_path: Path):
    """generate_dataset_asyncがJSONLファイルを作成する"""
    client = FakeAsyncClient()
    sentences = ["今日は楽しい", "明日は晴れる", "来週は旅行"]
    out = tmp_path / "dataset_async.jsonl"

    samples = await generate_dataset_async(
        client, sentences, output_path=out, show_progress=False
    )

    assert len(samples) == 3
    assert out.exists()

    loaded = load_dataset(out)
    assert len(loaded) == 3


@pytest.mark.asyncio
async def test_generate_dataset_async_handles_errors(tmp_path: Path):
    """generate_dataset_asyncがエラーを適切に処理する"""
    client = FakeAsyncClient(fail_indices={1})  # 2番目のサンプルでエラー
    sentences = ["文1", "文2", "文3"]
    out = tmp_path / "dataset_async.jsonl"

    samples = await generate_dataset_async(
        client, sentences, output_path=out, show_progress=False
    )

    # エラーが発生した1件を除いて2件成功
    assert len(samples) == 2


@pytest.mark.asyncio
async def test_generate_dataset_async_preserves_order(tmp_path: Path):
    """generate_dataset_asyncが元の順序を保持する"""
    client = FakeAsyncClient()
    sentences = ["文1", "文2", "文3", "文4", "文5"]
    out = tmp_path / "dataset_async.jsonl"

    samples = await generate_dataset_async(
        client, sentences, output_path=out, show_progress=False
    )

    # インデックス順にソートされているか確認
    for i, sample in enumerate(samples):
        assert sample.original_text == f"文{i + 1}"
