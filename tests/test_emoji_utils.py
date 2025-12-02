from src.data.emoji_utils import (
    extract_emojis,
    get_all_emojis,
    is_valid_emoji,
    normalize_skin_tone,
)


def test_normalize_skin_tone_removes_modifier():
    assert normalize_skin_tone("👋🏻") == "👋"


def test_normalize_skin_tone_multiple_modifiers():
    """複数の肌色修飾子を含むテキスト"""
    assert normalize_skin_tone("👋🏻👍🏽") == "👋👍"


def test_normalize_skin_tone_no_modifier():
    """肌色修飾子がない場合はそのまま返す"""
    assert normalize_skin_tone("👋👍") == "👋👍"


def test_extract_emojis_order_and_limit():
    text = "楽しい😊🎉✨最高！"
    emojis = extract_emojis(text, max_count=2)
    assert emojis == ["😊", "🎉"]
    assert all(is_valid_emoji(e) for e in emojis)


def test_extract_emojis_empty_text():
    """空文字列からは空リストを返す"""
    assert extract_emojis("") == []


def test_extract_emojis_no_emoji():
    """絵文字がないテキストからは空リストを返す"""
    assert extract_emojis("絵文字なしのテキスト") == []


def test_extract_emojis_only_emojis():
    """絵文字のみのテキスト"""
    emojis = extract_emojis("😊🎉✨")
    assert emojis == ["😊", "🎉", "✨"]


def test_get_all_emojis_returns_set():
    """get_all_emojisがセットを返すことを確認"""
    all_emojis = get_all_emojis()
    assert isinstance(all_emojis, set)
    assert len(all_emojis) > 1000  # 絵文字は1000種類以上ある
    assert "😊" in all_emojis


def test_is_valid_emoji():
    """絵文字の有効性チェック"""
    assert is_valid_emoji("😊") is True
    assert is_valid_emoji("🎉") is True
    assert is_valid_emoji("a") is False
    assert is_valid_emoji("あ") is False
    assert is_valid_emoji("") is False
