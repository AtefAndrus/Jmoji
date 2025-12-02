from src.data.text_preprocessor import (
    DEFAULT_NSFW_KEYWORDS,
    extract_sentences,
    filter_safe_sentences,
    is_safe_sentence,
    normalize_text,
)


def test_normalize_text():
    assert normalize_text("ＡＢ ｃ　　d") == "AB c d"


def test_normalize_text_strips_whitespace():
    """前後の空白が除去されることを確認"""
    assert normalize_text("  テスト  ") == "テスト"


def test_normalize_text_empty():
    """空文字列は空文字列を返す"""
    assert normalize_text("") == ""


def test_extract_sentences_filters_length_and_japanese():
    text = "今日は雨です。12345! 明日は晴れるかな？"
    sents = extract_sentences(text, min_len=5, max_len=30)
    # 期待される文が含まれることを確認
    assert "今日は雨です。" in sents
    assert "明日は晴れるかな？" in sents
    # 数字のみ文は含まれない（日本語文字がない）
    assert all("12345" not in s for s in sents)
    # 合計2文のみ
    assert len(sents) == 2


def test_extract_sentences_min_length_filter():
    """最小長でフィルタされることを確認"""
    text = "短い。これは十分に長い文です。"
    sents = extract_sentences(text, min_len=10, max_len=100)
    assert "短い。" not in sents
    assert "これは十分に長い文です。" in sents


def test_extract_sentences_max_length_filter():
    """最大長でフィルタされることを確認"""
    text = "短い文。" + "あ" * 100 + "。"
    sents = extract_sentences(text, min_len=1, max_len=50)
    assert "短い文。" in sents
    assert len(sents) == 1  # 長い文は除外


def test_extract_sentences_empty_text():
    """空文字列からは空リストを返す"""
    assert extract_sentences("") == []


def test_extract_sentences_no_japanese():
    """日本語がない文は除外される"""
    text = "Hello World! This is English."
    sents = extract_sentences(text, min_len=1, max_len=100)
    assert sents == []


# NSFWフィルタのテスト
def test_is_safe_sentence_with_default_keywords():
    """デフォルトキーワードでNSFW文を検出"""
    assert is_safe_sentence("今日は良い天気です。") is True
    assert is_safe_sentence("殺人事件が発生した。") is False
    assert is_safe_sentence("ポルノグラフィーの歴史") is False


def test_is_safe_sentence_with_custom_keywords():
    """カスタムキーワードでNSFW文を検出"""
    custom_keywords = {"テスト", "危険"}
    assert is_safe_sentence("これはテストです。", custom_keywords) is False
    assert is_safe_sentence("安全な文章です。", custom_keywords) is True
    assert is_safe_sentence("危険な行為", custom_keywords) is False


def test_filter_safe_sentences():
    """NSFWフィルタが複数文を正しくフィルタリング"""
    sentences = [
        "今日は良い天気です。",
        "殺人事件のニュース。",
        "明日は雨かもしれない。",
        "ポルノサイトへのリンク。",
    ]
    result = filter_safe_sentences(sentences)
    assert len(result) == 2
    assert "今日は良い天気です。" in result
    assert "明日は雨かもしれない。" in result


def test_filter_safe_sentences_with_custom_keywords():
    """カスタムキーワードでのフィルタリング"""
    sentences = ["テストA", "テストB", "本番"]
    custom_keywords = {"テスト"}
    result = filter_safe_sentences(sentences, custom_keywords)
    assert result == ["本番"]


def test_default_nsfw_keywords_is_set():
    """DEFAULT_NSFW_KEYWORDSがセットであること"""
    assert isinstance(DEFAULT_NSFW_KEYWORDS, set)
    assert len(DEFAULT_NSFW_KEYWORDS) > 0
    assert "殺人" in DEFAULT_NSFW_KEYWORDS
