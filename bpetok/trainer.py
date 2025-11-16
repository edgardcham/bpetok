"""BPE training algorithm - learns merge rules from corpus."""

from __future__ import annotations

from typing import Iterable

from .encode_decode import Tokenizer as RuntimeTokenizer
from .model import (
    MergeRule,
    TokenizerConfig,
    Vocabulary,
    save_merges,
    save_model,
    save_vocab,
)
from .normalize import VISIBLE_SPACE, normalize_text, pretokenize_characters, text_to_byte_symbols
from .utils import build_pair_stats, select_best_pair, update_pair_stats_for_merge


class TokenIndexer:
    """Maps string tokens to integer IDs for faster training."""

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: list[str] = []

    def get_or_add(self, token: str) -> int:
        existing = self.token_to_id.get(token)
        if existing is not None:
            return existing
        idx = len(self.id_to_token)
        self.token_to_id[token] = idx
        self.id_to_token.append(token)
        return idx

    def encode_sequence(self, sequence: list[str]) -> list[int]:
        return [self.get_or_add(tok) for tok in sequence]


def train(config: TokenizerConfig) -> tuple[Vocabulary, list[MergeRule]]:
    """
    Entry point called by CLI: trains a tokenizer and returns the vocab + merges

    Args:
        config: Tokenizer configuration.

    Returns:
        Tuple containing the vocabulary and merge rules.
    """
    raw_sequences = list(_load_corpus(config))
    indexer = TokenIndexer()
    sequences = [indexer.encode_sequence(seq) for seq in raw_sequences]
    vocab = Vocabulary.from_tokens(config.specials)
    initial_tokens = {token for seq in raw_sequences for token in seq}
    for token in sorted(initial_tokens):
        vocab.add_token(token)

    merge_rules: list[MergeRule] = []

    pair_counts, pair_locations, pair_heap = build_pair_stats(sequences)
    max_merges = config.max_merges or (config.target_vocabulary_size - len(vocab))
    for merge_idx in range(max_merges):
        best_pair, best_count = select_best_pair(pair_counts, pair_heap)
        if best_count < config.minimum_pair_frequency:
            break
        # Guard against stale stats: best_pair must have some tracked locations.
        locations = pair_locations.get(best_pair)
        if not locations:
            # Fallback: rebuild pair stats once from scratch.
            pair_counts, pair_locations, pair_heap = build_pair_stats(sequences)
            best_pair, best_count = select_best_pair(pair_counts, pair_heap)
            if best_count < config.minimum_pair_frequency:
                break
            locations = pair_locations.get(best_pair, set())

        left_id, right_id = best_pair
        left_symbol = indexer.id_to_token[left_id]
        right_symbol = indexer.id_to_token[right_id]
        new_symbol = left_symbol + right_symbol
        new_token_id = indexer.get_or_add(new_symbol)
        vocab.add_token(new_symbol)
        merge_rules.append(MergeRule(left=left_symbol, right=right_symbol, rank=merge_idx))
        locations = pair_locations.pop(best_pair, set())
        update_pair_stats_for_merge(
            sequences,
            best_pair,
            locations,
            pair_counts,
            pair_locations,
            pair_heap,
            new_token_id,
        )
        if config.debug_first_merges and merge_idx < 5:
            print("Top pairs after merge", merge_idx + 1)
            for pair, count in pair_counts.most_common(5):
                print(pair, count)
        if (merge_idx + 1) % config.progress_every == 0:
            print(
                f"[merge {merge_idx + 1}] "
                f"best pair=({left_symbol!r}, {right_symbol!r}) "
                f"freq={best_count} vocab_size={len(vocab)}"
            )
    output_dir = config.output_dir.resolve()
    save_vocab(output_dir / "vocab.json", vocab)
    save_merges(output_dir / "merges.txt", merge_rules)
    save_model(config.model_path, config, vocab, merge_rules)

    # Validation
    tokenizer = RuntimeTokenizer(config, vocab, merge_rules)
    _evaluate_on_validation(config, tokenizer)

    return vocab, merge_rules


def _load_corpus(config: TokenizerConfig) -> Iterable[list[str]]:
    """
    Yield token sequences according to the configured model kind.

    Args:
        config: Tokenizer configuration.

    Yields:
        Token sequences.
    """
    path = config.data_path.resolve(strict=True)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if config.model_kind == "character_bpe":
                yield pretokenize_characters(line, config)
            else:
                yield text_to_byte_symbols(line)


def _evaluate_on_validation(config: TokenizerConfig, tokenizer: RuntimeTokenizer) -> None:
    """
    Evaluate the tokenizer on the validation set.

    Args:
        config: Tokenizer configuration.
        vocab: Vocabulary.
        merge_rules: List of merge rules.
    """
    valid_path = config.valid_path
    if not valid_path.exists():
        print("No validation file found. Skipping evaluation.")
        return
    mismatches = 0
    skipped = 0
    total = 0
    with valid_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            total += 1
            try:
                ids = tokenizer.encode(line, strict=True)
            except KeyError:
                skipped += 1
                continue
            decoded = tokenizer.decode(ids)

            if config.model_kind == "character_bpe":
                reference = normalize_text(line, config)
                if config.add_visible_space:
                    reference = reference.replace(VISIBLE_SPACE, " ")
            else:
                reference = line

            if decoded != reference:
                mismatches += 1

    checked = total - skipped
    if checked == 0:
        print("[validation] skipped all lines (out-of-vocabulary characters)")
    elif mismatches:
        print(f"[validation] {mismatches}/{checked} lines failed round-trip "
              f"(skipped {skipped} due to OOV tokens)")
    else:
        print(f"[validation] {checked}/{checked} lines round-tripped successfully "
              f"(skipped {skipped} due to OOV tokens)")
