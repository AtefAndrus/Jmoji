from src.data.emoji_utils import (
    extract_emojis,
    filter_samples_by_top_emojis,
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


# filter_samples_by_top_emojis テスト
class TestFilterSamplesByTopEmojis:
    """filter_samples_by_top_emojis関数のテスト"""

    def test_basic_filtering(self):
        """基本的なフィルタリング動作"""
        samples = [
            {"text": "a", "emoji_string": "😊 🎉"},
            {"text": "b", "emoji_string": "😊 ✨"},
            {"text": "c", "emoji_string": "🎉 ✨"},
            {"text": "d", "emoji_string": "😊 😊 😊"},  # 😊が多い
        ]
        # top_n=2: 😊(4回), 🎉(2回)がTop-2
        filtered, counts, top_emojis = filter_samples_by_top_emojis(samples, top_n=2)

        assert "😊" in top_emojis
        assert "🎉" in top_emojis
        assert len(top_emojis) == 2
        # ✨を含むサンプルはフィルタされる
        assert len(filtered) == 2
        assert filtered[0]["text"] == "a"
        assert filtered[1]["text"] == "d"

    def test_all_samples_pass(self):
        """すべてのサンプルがTop-Nに収まる場合"""
        samples = [
            {"text": "a", "emoji_string": "😊"},
            {"text": "b", "emoji_string": "🎉"},
        ]
        filtered, counts, top_emojis = filter_samples_by_top_emojis(samples, top_n=10)

        assert len(filtered) == 2
        assert len(top_emojis) == 2

    def test_empty_samples(self):
        """空のサンプルリスト"""
        filtered, counts, top_emojis = filter_samples_by_top_emojis([], top_n=10)

        assert filtered == []
        assert len(counts) == 0
        assert len(top_emojis) == 0

    def test_samples_without_emoji_key(self):
        """emoji_keyが存在しないサンプル"""
        samples = [
            {"text": "a"},
            {"text": "b", "emoji_string": "😊"},
        ]
        filtered, counts, top_emojis = filter_samples_by_top_emojis(samples, top_n=10)

        # emoji_stringがないサンプルはフィルタされる（emoji_strが空）
        assert len(filtered) == 1
        assert filtered[0]["text"] == "b"

    def test_custom_emoji_key(self):
        """カスタムemoji_keyを使用"""
        samples = [
            {"text": "a", "emojis": "😊 🎉"},
            {"text": "b", "emojis": "😊"},
        ]
        filtered, counts, top_emojis = filter_samples_by_top_emojis(
            samples, top_n=10, emoji_key="emojis"
        )

        assert len(filtered) == 2
        assert counts["😊"] == 2
        assert counts["🎉"] == 1

    def test_empty_emoji_string(self):
        """空のemoji_stringを持つサンプル"""
        samples = [
            {"text": "a", "emoji_string": ""},
            {"text": "b", "emoji_string": "😊"},
        ]
        filtered, counts, top_emojis = filter_samples_by_top_emojis(samples, top_n=10)

        # 空文字列のサンプルはフィルタされる
        assert len(filtered) == 1
        assert filtered[0]["text"] == "b"

    def test_counts_accuracy(self):
        """フィルタ後のカウントが正確か"""
        samples = [
            {"text": "a", "emoji_string": "😊 😊 🎉"},
            {"text": "b", "emoji_string": "😊 ✨"},  # ✨でフィルタされる
            {"text": "c", "emoji_string": "🎉 🎉"},
        ]
        # top_n=2: 😊(3回), 🎉(3回)
        filtered, counts, top_emojis = filter_samples_by_top_emojis(samples, top_n=2)

        # bがフィルタされるので、残りはa,c
        assert len(filtered) == 2
        # フィルタ後のカウント: 😊(2), 🎉(3)
        assert counts["😊"] == 2
        assert counts["🎉"] == 3
