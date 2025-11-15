"""Utility functions for efficient pair counting and heap management."""

from __future__ import annotations

from collections import Counter, defaultdict

Pair = tuple[str, str]
Location = tuple[int, int]


def build_pair_stats(
    sequences: list[list[str]],
) -> tuple[Counter[tuple[str, str]], dict[tuple[str, str], set[tuple[int, int]]]]:
    """
    Build pair stats for the sequences.

    Args:
        sequences: List of token sequences.

    Returns:
        Tuple containing the pair counts and locations.
    """
    counts: Counter[tuple[str, str]] = Counter()
    locations: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)
    for seq_idx, seq in enumerate(sequences):
        for token_idx, (left, right) in enumerate(zip(seq, seq[1:])):
            pair = (left, right)
            counts[pair] += 1
            locations[pair].add((seq_idx, token_idx))
    return counts, locations


def select_best_pair(pair_counts: Counter[tuple[str, str]]) -> tuple[tuple[str, str], int]:
    """
    Select the best pair to merge.

    Args:
        pair_counts: Counter of pairs.

    Returns:
        Tuple containing the best pair and its count.
    """
    if not pair_counts:
        return ("", ""), 0
    best_pair, best_count = max(pair_counts.items(), key=lambda item: item[1])
    return best_pair, best_count


def update_pair_stats_for_merge(
    sequences: list[list[str]],
    pair: tuple[str, str],
    locations: set[tuple[int, int]],
    pair_counts: Counter[tuple[str, str]],
    pair_locations: dict[tuple[str, str], set[tuple[int, int]]],
) -> None:
    """
    Update pair stats for the merge.

    Args:
        sequences: List of token sequences.
        pair: Tuple containing the pair to merge.
        locations: Set of locations of the pair.
        pair_counts: Counter of pairs.
        pair_locations: Dict of locations of pairs.
    """
    merged_symbol = "".join(pair)
    for seq_idx, token_idx in sorted(locations, reverse=True):
        seq = sequences[seq_idx]
        if token_idx >= len(seq) - 1:
            continue
        if (seq[token_idx], seq[token_idx + 1]) != pair:
            continue

        # remove the pair itself
        pair_counts[pair] -= 1
        pair_locations[pair].discard((seq_idx, token_idx))

        left_neighbor = seq[token_idx - 1] if token_idx > 0 else None
        right_neighbor = seq[token_idx + 2] if token_idx + 2 < len(seq) else None

        if left_neighbor:
            old_left_pair = (left_neighbor, seq[token_idx])
            pair_counts[old_left_pair] -= 1
            pair_locations[old_left_pair].discard((seq_idx, token_idx - 1))

        if right_neighbor:
            old_right_pair = (seq[token_idx + 1], right_neighbor)
            pair_counts[old_right_pair] -= 1
            pair_locations[old_right_pair].discard((seq_idx, token_idx + 1))

        seq[token_idx : token_idx + 2] = [merged_symbol]

        if left_neighbor:
            new_left_pair = (left_neighbor, merged_symbol)
            pair_counts[new_left_pair] += 1
            pair_locations[new_left_pair].add((seq_idx, token_idx - 1))

        if right_neighbor:
            new_right_pair = (merged_symbol, right_neighbor)
            pair_counts[new_right_pair] += 1
            pair_locations[new_right_pair].add((seq_idx, token_idx))
