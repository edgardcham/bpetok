"""Test benchmark and metric calculations."""

import pytest

from bpetok import metrics


class FakeTokenizer:
    """Minimal tokenizer stub that returns predetermined tokens."""

    def __init__(self, tokens: list[int]):
        self.tokens = tokens

    def encode(self, text: str) -> list[int]:
        return self.tokens

    def decode(self, ids: list[int]) -> str:
        return "x" * len(ids)


def _mock_perf_counter(step: float = 0.01):
    """Return a perf_counter function that advances by `step` each call."""
    state = {"value": 0.0}

    def perf_counter() -> float:
        current = state["value"]
        state["value"] += step
        return current

    return perf_counter


def test_measure_encoding_speed(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = FakeTokenizer(tokens=[1, 2, 3])
    monkeypatch.setattr(metrics.time, "perf_counter", _mock_perf_counter(step=0.1))
    speed = metrics.measure_encoding_speed(tokenizer, "hello", warmup=0, trials=2)
    assert speed == pytest.approx(len(tokenizer.tokens) / 0.1)


def test_measure_decoding_speed(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = FakeTokenizer(tokens=[1, 2, 3, 4])
    ids = [1, 2, 3, 4]
    monkeypatch.setattr(metrics.time, "perf_counter", _mock_perf_counter(step=0.05))
    speed = metrics.measure_decoding_speed(tokenizer, ids, warmup=0, trials=3)
    assert speed == pytest.approx(len(ids) / 0.05)


def test_average_tokens_per_word() -> None:
    assert metrics.average_tokens_per_word(num_tokens=10, num_words=5) == 2
    assert metrics.average_tokens_per_word(num_tokens=10, num_words=0) == 10


def test_compression_ratio() -> None:
    assert metrics.compression_ratio(50, 100) == 0.5
    assert metrics.compression_ratio(50, 0) == 50


def test_baseline_token_counts() -> None:
    counts = metrics.baseline_token_counts("hello world")
    assert counts["character"] == 11
    assert counts["whitespace"] == 2
