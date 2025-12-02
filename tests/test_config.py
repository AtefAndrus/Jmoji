from pathlib import Path

import pytest

from src.config import load_config


def test_load_config(tmp_path: Path):
    """YAMLファイルを正しく読み込めることを確認"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """
data:
  max_samples: 100
  random_seed: 42
training:
  batch_size: 16
""",
        encoding="utf-8",
    )

    cfg = load_config(config_file)
    assert cfg["data"]["max_samples"] == 100
    assert cfg["data"]["random_seed"] == 42
    assert cfg["training"]["batch_size"] == 16


def test_load_config_empty_file(tmp_path: Path):
    """空のYAMLファイルはNoneを返す"""
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("", encoding="utf-8")

    cfg = load_config(config_file)
    assert cfg is None


def test_load_config_file_not_found():
    """存在しないファイルはFileNotFoundErrorを発生"""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")
