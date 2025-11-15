"""Utility functions for efficient pair counting and heap management."""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict

Pair = tuple[str, str]
Location = tuple[int, int]


def _push_heap(
    heap: list[tuple[int, Pair]],
    pair_counts: Counter[Pair],
    pair: Pair,
) -> None:
    """Push the current count for `pair` onto the heap if positive."""
    count = pair_counts.get(pair, 0)
    if count > 0:
        heapq.heappush(heap, (-count, pair))


def build_pair_stats(
    sequences: list[list[str]],
) -> tuple[Counter[Pair], dict[Pair, set[Location]], list[tuple[int, Pair]]]:
    """
    Build pair stats for the sequences.

    Args:
        sequences: List of token sequences.

    Returns:
        Tuple containing the pair counts, locations, and a max-heap.
    """
    counts: Counter[Pair] = Counter()
    locations: dict[Pair, set[Location]] = defaultdict(set)
    for seq_idx, seq in enumerate(sequences):
        for token_idx, (left, right) in enumerate(zip(seq, seq[1:])):
            pair = (left, right)
            counts[pair] += 1
            locations[pair].add((seq_idx, token_idx))
    heap: list[tuple[int, Pair]] = [(-count, pair) for pair, count in counts.items()]
    heapq.heapify(heap)
    return counts, locations, heap


def select_best_pair(
    pair_counts: Counter[Pair],
    heap: list[tuple[int, Pair]],
) -> tuple[Pair, int]:
    """
    Select the best pair to merge using a max-heap with lazy invalidation.

    Args:
        pair_counts: Counter of pairs.
        heap: Max-heap storing (-count, pair) entries.

    Returns:
        Tuple containing the best pair and its count.
    """
    while heap:
        neg_count, pair = heap[0]
        count = -neg_count
        if pair_counts.get(pair, 0) == count and count > 0:
            return pair, count
        heapq.heappop(heap)
    return ("", ""), 0


def update_pair_stats_for_merge(
    sequences: list[list[str]],
    pair: Pair,
    locations: set[Location],
    pair_counts: Counter[Pair],
    pair_locations: dict[Pair, set[Location]],
    heap: list[tuple[int, Pair]],
) -> None:
    """
    Update pair stats for the merge.

    Args:
        sequences: List of token sequences.
        pair: Tuple containing the pair to merge.
        locations: Set of locations of the pair.
        pair_counts: Counter of pairs.
        pair_locations: Dict of locations of pairs.
        heap: Max-heap storing (-count, pair) entries.
    """
    merged_symbol = "".join(pair)
    for seq_idx, token_idx in sorted(locations, reverse=True):
        seq = sequences[seq_idx]
        if token_idx >= len(seq) - 1:
            continue
        if (seq[token_idx], seq[token_idx + 1]) != pair:
            pair_locations[pair].discard((seq_idx, token_idx))
            continue

        pair_counts[pair] -= 1
        _push_heap(heap, pair_counts, pair)

        left_neighbor = seq[token_idx - 1] if token_idx > 0 else None
        right_neighbor = seq[token_idx + 2] if token_idx + 2 < len(seq) else None

        if left_neighbor:
            old_left_pair = (left_neighbor, seq[token_idx])
            left_coord = (seq_idx, token_idx - 1)
            pair_counts[old_left_pair] -= 1
            pair_locations[old_left_pair].discard(left_coord)
            if pair_counts[old_left_pair] <= 0:
                pair_locations.pop(old_left_pair, None)
            _push_heap(heap, pair_counts, old_left_pair)

        if right_neighbor:
            old_right_pair = (seq[token_idx + 1], right_neighbor)
            right_coord = (seq_idx, token_idx + 1)
            pair_counts[old_right_pair] -= 1
            pair_locations[old_right_pair].discard(right_coord)
            if pair_counts[old_right_pair] <= 0:
                pair_locations.pop(old_right_pair, None)
            _push_heap(heap, pair_counts, old_right_pair)

        seq[token_idx : token_idx + 2] = [merged_symbol]

        if left_neighbor:
            new_left_pair = (left_neighbor, merged_symbol)
            new_left_coord = (seq_idx, token_idx - 1)
            pair_counts[new_left_pair] += 1
            pair_locations[new_left_pair].add(new_left_coord)
            _push_heap(heap, pair_counts, new_left_pair)

        if right_neighbor:
            new_right_pair = (merged_symbol, right_neighbor)
            new_right_coord = (seq_idx, token_idx)
            pair_counts[new_right_pair] += 1
            pair_locations[new_right_pair].add(new_right_coord)
            _push_heap(heap, pair_counts, new_right_pair)

    if pair_counts[pair] <= 0:
        pair_locations.pop(pair, None)
