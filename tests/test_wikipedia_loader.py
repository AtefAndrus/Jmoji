from src.data.wikipedia_loader import load_wikipedia_sentences


def test_load_wikipedia_sentences_uses_preprocess(monkeypatch):
    dummy_items = [{"text": "今日は雨です。明日は晴れるでしょう。"}]

    class DummyDS:
        def shuffle(self, seed=0, buffer_size=0):
            return self

        def __iter__(self):
            return iter(dummy_items)

    def fake_load_dataset(*args, **kwargs):
        return DummyDS()

    monkeypatch.setattr("src.data.wikipedia_loader.load_dataset", fake_load_dataset)

    sentences = load_wikipedia_sentences(max_samples=1, min_len=1)
    assert sentences == ["今日は雨です。"]
