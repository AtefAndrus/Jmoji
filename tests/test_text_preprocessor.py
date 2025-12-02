from src.data.text_preprocessor import extract_sentences, normalize_text


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
