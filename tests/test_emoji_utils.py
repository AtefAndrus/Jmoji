from src.data.emoji_utils import extract_emojis, is_valid_emoji, normalize_skin_tone


def test_normalize_skin_tone_removes_modifier():
    assert normalize_skin_tone("👋🏻") == "👋"


def test_extract_emojis_order_and_limit():
    text = "楽しい😊🎉✨最高！"
    emojis = extract_emojis(text, max_count=2)
    assert emojis == ["😊", "🎉"]
    assert all(is_valid_emoji(e) for e in emojis)
