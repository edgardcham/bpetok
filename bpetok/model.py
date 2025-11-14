"""Data classes, configuration, and file I/O for BPE tokenizer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, Field


class TokenizerConfig(BaseModel):
    # Different versions of BPE use different tokenization strategies
    model_kind: Literal["character_bpe", "byte_bpe"] = "character_bpe"

    # Training stops when vocab hits target size or pairs fall below min frequency
    target_vocabulary_size: int = 16000
    minimum_pair_frequency: int = 2

    # Optional runtime cap: stop after at most this many merge operations even if other criteria are not met
    max_merges: int | None = None

    # Random seed for reproducibility
    random_seed: int = 28
    # Emit trainer progress every N merges for deterministic logs
    progress_every: int = 100

    # Normalization toggles must match between training and encoding
    unicode_normalization: Literal["NFC", "NFKC", "NFD", "NFKD", "none"] = "NFC"
    strip_accents: bool = False
    lowercase: bool = False
    add_visible_space: bool = True

    # Reserved tokens keep fixed IDs; Field avoids mutable default pitfalls
    specials: list[str] = Field(default_factory=lambda: ["<unknown>", "<begin>", "<end>"])

    # Paths to training data and validation set
    data_path: Path = Path("./data/train.txt")
    valid_path: Path = Path("./data/valid.txt")
    output_dir: Path = Path("./out")
    model_path: Path = Path("./out/model.json")

    @property
    def special_id_map(self) -> dict[str, int]:
        """Deterministic mapping so reserved tokens never enter merge operations."""
        return {token: idx for idx, token in enumerate(self.specials)}


@dataclass
class Vocabulary:
    """Tracks tokens <-> ids and keeps reserved specials pinned to the lowest indices"""

    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: list[str] = field(default_factory=list)

    @classmethod
    def from_tokens(cls, tokens: Iterable[str]) -> "Vocabulary":
        """
        Build vocab where each token gets its position as the id (i.e. token_to_id[token] == id_to_token.index(token))

        Args:
            tokens: Iterable of tokens to add to the vocabulary.

        Returns:
            Vocabulary object built from the tokens.
        """
        vocab = cls()
        for token in tokens:
            vocab.add_token(token)
        return vocab

    def add_token(self, token: str) -> int:
        """
        Add a token to the vocabulary.

        Args:
            token: Token to add to the vocabulary.

        Returns:
            ID of the added token.
        """
        if token in self.token_to_id:
            return self.token_to_id[token]
        idx = len(self.id_to_token)
        self.id_to_token.append(token)
        self.token_to_id[token] = idx
        return idx

    def __len__(self) -> int:
        """
        Get the number of tokens in the vocabulary.

        Returns:
            Number of tokens in the vocabulary.
        """
        return len(self.id_to_token)

    def get_id(self, token: str) -> int:
        """
        Get the ID of a token.

        Args:
            token: Token to get the ID for.

        Returns:
            ID of the token.
        """
        if token in self.token_to_id:
            return self.token_to_id[token]
        raise ValueError(f"Token {token} not in vocabulary")

    def get_token(self, idx: int) -> str:
        """
        Get the token for a given ID.

        Args:
            id: ID to get the token for.

        Returns:
            Token for the given ID.
        """
        if 0 <= idx < len(self.id_to_token):
            return self.id_to_token[idx]
        raise ValueError(f"ID {idx} not in vocabulary")

    def to_json(self) -> dict:
        """
        Convert vocabulary to a JSON payload.

        Returns:
            Dictionary containing the vocabulary data.
        """
        return {"tokens": self.id_to_token}

    @classmethod
    def from_json(cls, payload: dict) -> "Vocabulary":
        """
        Build vocabulary from a JSON payload.

        Args:
            payload: Dictionary containing the vocabulary data.

        Returns:
            Vocabulary object built from the JSON payload.
        """
        return cls.from_tokens(payload["tokens"])


def save_vocab(path: Path, vocab: Vocabulary) -> None:
    """
    Save vocabulary to a file in JSON format.

    Args:
        path: Path to the file where the vocabulary will be saved.
        vocab: Vocabulary object to be saved.

    Returns:
        None
    """
    # Ensure the directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write the vocabulary to the file
    path.write_text(json.dumps(vocab.to_json(), ensure_ascii=False, indent=2))


def load_vocab(path: Path) -> Vocabulary:
    """
    Load vocabulary from a file in JSON format.

    Args:
        path: Path to the file where the vocabulary is saved.

    Returns:
        Vocabulary object loaded from the file.
    """
    # Ensure file exists
    path.resolve(strict=True)
    # Read the file
    return Vocabulary.from_json(json.loads(path.read_text()))


@dataclass(frozen=True)
class MergeRule:
    """
    Represents a merge like ('A', 'B') -> new symbol with a fixed rank.

    Args:
        left: Left symbol of the merge.
        right: Right symbol of the merge.
        rank: Rank of the merge.
    """

    left: str
    right: str
    rank: int

    def as_tuple(self) -> tuple[str, str]:
        """
        Get the merge rule as a tuple.

        Returns:
            Tuple containing the left and right symbols of the merge.
        """
        return self.left, self.right


def save_merges(path: Path, merges: list[MergeRule]) -> None:
    """
    Save merges to a file in text format.

    Args:
        path: Path to the file where the merges will be saved.
        merges: List of merge rules to be saved.

    Returns:
        None
    """
    # Ensure the directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"{rule.left} {rule.right}" for rule in merges]
    # Write the merges to the file
    path.write_text("\n".join(lines))


def load_merges(path: Path) -> list[MergeRule]:
    """
    Load merges from a file in text format.

    Args:
        path: Path to the file where the merges are saved.

    Returns:
        List of merge rules loaded from the file.
    """
    # Ensure file exists
    path.resolve(strict=True)
    # Read the file
    merges: list[MergeRule] = []
    for rank, line in enumerate(path.read_text().splitlines()):
        left, right = line.strip().split()
        merges.append(MergeRule(left=left, right=right, rank=rank))
    return merges


def save_model(
    path: Path, config: TokenizerConfig, vocab: Vocabulary, merges: list[MergeRule]
) -> None:
    """
    Save model to a file in JSON format.

    Args:
        path: Path to the file where the model will be saved.
        config: Tokenizer configuration.
        vocab: Vocabulary object.
        merges: List of merge rules.

    Returns:
        None
    """

    payload = {
        "config": config.model_dump(),
        "vocab": vocab.to_json(),
        "merges": [rule.as_tuple() for rule in merges],
    }
    # Ensure the directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write the model to the file
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def load_model(path: Path) -> tuple[TokenizerConfig, Vocabulary, list[MergeRule]]:
    """
    Load model from a file in JSON format.

    Args:
        path: Path to the file where the model is saved.

    Returns:
        Tuple containing the tokenizer configuration, vocabulary, and merge rules.
    """
    # Ensure file exists
    path.resolve(strict=True)
    # Read the file
    payload = json.loads(path.read_text())
    # Load the tokenizer configuration
    config = TokenizerConfig.model_validate(payload["config"])
    # Load the vocabulary
    vocab = Vocabulary.from_json(payload["vocab"])
    # Load the merge rules
    merges = [
        MergeRule(left=left, right=right, rank=rank)
        for rank, (left, right) in enumerate(payload["merges"])
    ]
    return config, vocab, merges
