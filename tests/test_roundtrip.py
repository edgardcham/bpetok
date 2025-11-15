"""Test round-trip encoding/decoding accuracy."""

import pytest

from bpetok.encode_decode import Tokenizer
from bpetok.model import TokenizerConfig, Vocabulary
from bpetok.normalize import BYTE_TO_CHAR


@pytest.fixture
def byte_tokenizer() -> Tokenizer:
    """Return a byte-level tokenizer with a full byte vocabulary."""
    config = TokenizerConfig(model_kind="byte_bpe", add_visible_space=False)
    vocab_tokens = config.specials + list(BYTE_TO_CHAR)
    vocab = Vocabulary.from_tokens(vocab_tokens)
    return Tokenizer(config, vocab, merges=[])


@pytest.mark.parametrize(
    "text",
    [
        "Hello world",
        "Hello, world!!! How are you?",
        "😀 😃 😄 😁 😂 🤣",
        "Привет мир",
        "こんにちは 世界",
        "",
        "a",
    ],
)
def test_roundtrip_various_inputs(byte_tokenizer: Tokenizer, text: str) -> None:
    """Ensure decode(encode(text)) == text for a variety of inputs."""
    ids = byte_tokenizer.encode(text)
    decoded = byte_tokenizer.decode(ids)
    assert decoded == text
