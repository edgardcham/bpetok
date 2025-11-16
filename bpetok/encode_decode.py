"""Encoding and decoding logic for trained BPE tokenizer."""

from __future__ import annotations

from typing import Iterable

from .model import MergeRule, TokenizerConfig, Vocabulary, load_model
from .normalize import (
    VISIBLE_SPACE,
    byte_symbols_to_text,
    normalize_text,
    pretokenize_characters,
    text_to_byte_symbols,
)


class Tokenizer:
    """Lightweight runtime wrapper around a trained tokenizer model."""

    def __init__(self, config: TokenizerConfig, vocab: Vocabulary, merges: list[MergeRule]):
        self.config = config
        self.vocab = vocab
        self.merges = merges
        self.merge_ranks = {rule.as_tuple(): rule.rank for rule in merges}
        self.unknown_id = self.vocab.get_id(config.specials[0])

    @classmethod
    def load(cls, path) -> "Tokenizer":
        config, vocab, merges = load_model(path)
        return cls(config, vocab, merges)

    def _pretokenize(self, text: str) -> list[str]:
        """
        Pre-tokenize the text according to the model kind.
        Mirrors training pretokenization.

        Args:
            text: Text to pretokenize.

        Returns:
            List of pretokenized symbols.
        """
        if self.config.model_kind == "character_bpe":
            normalized = normalize_text(text, self.config)
            return pretokenize_characters(normalized, self.config)
        return text_to_byte_symbols(text)

    def _apply_merges(self, tokens: list[str]) -> list[str]:
        """
        Apply the merges to the tokens.

        Args:
            tokens: List of tokens to apply the merges to.

        Returns:
            List of merged tokens.
        """
        merged = tokens[:]
        ranks = self.merge_ranks
        i = 0
        while i < len(merged) - 1:
            pair = (merged[i], merged[i + 1])
            if pair in ranks:
                merged_token = "".join(pair)
                merged[i : i + 2] = [merged_token]
                if i > 0:
                    i -= 1
            else:
                i += 1
        return merged

    def encode(self, text: str, *, strict: bool = False) -> list[int]:
        """
        Encode the text into a list of token IDs.

        Args:
            text: Text to encode.
            strict: If True, raise KeyError when a token is missing.

        Returns:
            List of token IDs.
        """
        tokens = self._pretokenize(text)
        tokens = self._apply_merges(tokens)
        encoded: list[int] = []
        for token in tokens:
            token_id = self.vocab.token_to_id.get(token)
            if token_id is None:
                if strict:
                    raise KeyError(token)
                token_id = self.unknown_id
            encoded.append(token_id)
        return encoded

    def decode(self, ids: Iterable[int]) -> str:
        """
        Decode a list of token IDs into a string.

        Args:
            ids: List of token IDs to decode.

        Returns:
            Decoded string.
        """
        tokens = [self.vocab.get_token(idx) for idx in ids]
        if self.config.model_kind == "character_bpe":
            text = "".join(tokens)
            if self.config.add_visible_space:
                text = text.replace(VISIBLE_SPACE, " ")
            return text
        # Byte-level: flatten merged tokens into the underlying byte symbols.
        flat_symbols = [ch for tok in tokens for ch in tok]
        text = byte_symbols_to_text(flat_symbols)
        return text
