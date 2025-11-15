"""Benchmark measurements and performance metrics."""

from __future__ import annotations

import time
from typing import Mapping, Sequence

from .encode_decode import Tokenizer

SECONDS_EPS = 1e-12


def measure_encoding_speed(
    tokenizer: Tokenizer,
    text: str,
    warmup: int = 1,
    trials: int = 3,
) -> float:
    """
    Return tokens/second for encoding.

    Args:
        tokenizer: Tokenizer to measure.
        text: Text to encode.
        warmup: Number of warmup trials.
        trials: Number of trials to measure.
    """
    for _ in range(warmup):
        tokenizer.encode(text)
    total_tokens = 0
    total_time = 0.0
    for _ in range(max(trials, 1)):
        start = time.perf_counter()
        ids = tokenizer.encode(text)
        total_time += time.perf_counter() - start
        total_tokens += len(ids)
    return total_tokens / max(total_time, SECONDS_EPS)


def measure_decoding_speed(
    tokenizer: Tokenizer,
    ids: Sequence[int],
    warmup: int = 1,
    trials: int = 3,
) -> float:
    """
    Return tokens/second for decoding.

    Args:
        tokenizer: Tokenizer to measure.
        ids: IDs to decode.
        warmup: Number of warmup trials.
        trials: Number of trials to measure.
    """
    for _ in range(warmup):
        tokenizer.decode(ids)
    total_time = 0.0
    for _ in range(max(trials, 1)):
        start = time.perf_counter()
        tokenizer.decode(ids)
        total_time += time.perf_counter() - start
    return (len(ids) * max(trials, 1)) / max(total_time, SECONDS_EPS)


def average_tokens_per_word(num_tokens: int, num_words: int) -> float:
    """
    Compute average tokens per word (guarding against zero words).

    Args:
        num_tokens: Number of tokens.
        num_words: Number of words.

    Returns:
        Average tokens per word.
    """
    return num_tokens / max(num_words, 1)


def compression_ratio(num_tokens: int, baseline_tokens: int) -> float:
    """
    Ratio of learned-token count to a baseline token count.

    Args:
        num_tokens: Number of tokens.
        baseline_tokens: Baseline token count.

    Returns:
        Compression ratio.
    """
    return num_tokens / max(baseline_tokens, 1)


def baseline_token_counts(text: str) -> Mapping[str, int]:
    """
    Return baseline token counts for character-level and whitespace-level tokenizers.

    Args:
        text: Text to count tokens in.

    Returns:
        Mapping of token to count.
    """
    return {"character": len(text), "whitespace": len(text.split())}
