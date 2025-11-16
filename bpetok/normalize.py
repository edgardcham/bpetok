"""Text and byte normalization, pretokenization, and splitting."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from .model import TokenizerConfig

VISIBLE_SPACE = "_"
WHITESPACE_RE = re.compile(r"\s+")
WORD_OR_PUNCT_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def normalize_text(text: str, config: TokenizerConfig) -> str:
    """
    Normalize text using the given configuration.

    Args:
        text: Text to normalize.
        config: Tokenizer configuration.

    Returns:
        Normalized text.
    """
    if config.unicode_normalization != "none":
        text = unicodedata.normalize(config.unicode_normalization, text)
    if config.strip_accents:
        text = _strip_accents(text)
    if config.lowercase:
        text = text.lower()
    text = WHITESPACE_RE.sub(" ", text.strip())
    if config.add_visible_space:
        text = text.replace(" ", f"{VISIBLE_SPACE}")
    return text


def _strip_accents(text: str) -> str:
    """
    Strip accents from the given text.

    Args:
        text: Text to strip accents from.

    Returns:
        Text with accents stripped. Example: "café" -> "cafe"
    """
    decomposed = unicodedata.normalize("NFD", text)
    return "".join([c for c in decomposed if not unicodedata.combining(c)])


def split_visible_spaces(text: str) -> Iterable[str]:
    """
    Split a normalized string into tokens while keeping the visible-space markers.

    Args:
        text: Normalized string to split.

    Returns:
        Iterable of tokens.
    """
    for segment in text.split(VISIBLE_SPACE):
        if not segment:
            yield VISIBLE_SPACE
            continue
        yield VISIBLE_SPACE
        yield from WORD_OR_PUNCT_RE.findall(segment)


def pretokenize_characters(text: str, config: TokenizerConfig) -> list[str]:
    """
    Use normalization + regex splitting to produce character-mode symbols..

    Args:
        text: Normalized string to pretokenize.
        config: Tokenizer configuration.

    Returns:
        List of pretokenized characters.
    """
    normalized = normalize_text(text, config)
    tokens: list[str] = []
    first = True
    for chunk in normalized.split(VISIBLE_SPACE):
        if not chunk:
            continue
        if not first:
            # Insert a visible-space marker between chunks, but not at the start.
            tokens.append(VISIBLE_SPACE)
        first = False
        tokens.extend(list(chunk))  # break words into chars
    return tokens


def text_to_byte_symbols(text: str) -> list[str]:
    """
    Convert UTF-8 text into byte symbols (byte-level BPE).

    Each byte is mapped to a single Unicode codepoint via BYTE_TO_CHAR so that
    merged tokens can be flattened back to the underlying byte sequence.
    """
    byte_values = text.encode("utf-8")
    return [BYTE_TO_CHAR[b] for b in byte_values]


def byte_symbols_to_text(symbols: Iterable[str]) -> str:
    """
    Inverse of text_to_byte_symbols.

    Args:
        symbols: Iterable of single-character byte symbols.

    Returns:
        Text decoded from byte symbols.
    """
    byte_values = bytes(CHAR_TO_BYTE[ch] for ch in symbols)
    return byte_values.decode("utf-8", errors="strict")


def _build_byte_tables() -> tuple[list[str], dict[str, int]]:
    # Map each byte value to a single-character symbol. For readability we
    # simply reuse the same codepoint; this is reversible because we only ever
    # feed these symbols into CHAR_TO_BYTE.
    byte_to_char = [chr(b) for b in range(256)]
    char_to_byte = {char: byte for byte, char in enumerate(byte_to_char)}
    return byte_to_char, char_to_byte


BYTE_TO_CHAR, CHAR_TO_BYTE = _build_byte_tables()
