from src.data.text_preprocessor import extract_sentences, normalize_text


def test_normalize_text():
    assert normalize_text("ＡＢ ｃ　　d") == "AB c d"


def test_extract_sentences_filters_length_and_japanese():
    text = "今日は雨です。12345! 明日は晴れるかな？"  # 短い数字文は除外
    sents = extract_sentences(text, min_len=5, max_len=30)
    assert "今日は雨です。" in sents
    # 数字のみ文は含まれない
    assert all("12345" not in s for s in sents)
