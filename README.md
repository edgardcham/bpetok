# BPETok - Byte Pair Encoding Tokenizer

A minimal, educational implementation of Byte Pair Encoding (BPE) tokenization from scratch. This project prioritizes learning and clarity over speed.

DeepWiki available [here](https://deepwiki.com/edgardcham/bpetok/).

## Project Goals

- Build a BPE tokenizer from raw text
- Encode/decode any string with perfect round-trip accuracy
- Provide a tiny, easy-to-use CLI
- Include comprehensive benchmarks: tokens/second, average tokens per word, compression ratio
- Maintain unit tests with reproducible configuration using Pydantic

## How to Use This Guide

1. **Skim the intuition**: read the short BPE primer below so you know *why* merges work before touching code.
2. **Follow the numbered sections**: each major heading mirrors a module in the repo (normalization, trainer, encoder, CLI).
3. **Experiment after every section**: run the CLI commands or notebook snippets that are suggested so you immediately see the theory in action.
4. **Measure and reflect**: use the metrics/tests sections to confirm your understanding; tweak the config knobs and watch how behavior changes.

Treat this README as a workbook—flip back to it whenever you add a new feature so the implementation and explanation stay in sync.

## BPE in 60 Seconds

BPE starts with the smallest possible alphabet (characters or raw bytes). It repeatedly replaces the most frequent adjacent pair with a new symbol. High-frequency patterns like “th” or “ing” quickly become single tokens, which compresses text and reduces out-of-vocabulary issues. Once you learn the merge list, encoding is just “apply the same merges, in order, to new text”; decoding walks the steps back.

## BPE Implementation Approaches

### 1. Classic Character-Based BPE

The traditional approach that forms the foundation of subword tokenization:

1. Split text into words (separated by spaces and punctuation)
2. Treat each word as a sequence of characters
3. Apply merge rules over pairs of adjacent characters to create longer subword pieces
4. Handle new or rare words by decomposing them into known pieces (no out-of-vocabulary issues)

**Normalization & Pretokenization:**
- Normalize Unicode to standard form (NFC)
- Convert to lowercase (reduces vocabulary size)
- Collapse whitespace to single spaces
- Split tokens by word character sequences; treat punctuation as individual tokens
- Add visible space marker (e.g., `▁`) to make spaces explicit

### 2. Byte-Level BPE (GPT-2 Style)

A more robust approach that handles any text:

1. Convert text to bytes using UTF-8
2. Map each byte to a visible symbol using a reversible table
3. No word splitting - spaces are just another byte
4. Guarantees perfect handling of any Unicode text

**Preprocessing:**
- Convert text directly to UTF-8 bytes
- Map bytes to printable symbols for inspection
- No normalization or lowercasing needed

## Training Algorithm

Think of the trainer as a friendly loop with only three responsibilities: start from tiny pieces, count what appears together, and merge the obvious winners. The mechanics below look formal, but the story is “keep gluing the most common neighbors until adding new glue no longer helps.”

### Initialization

**Character-based (#1):** Each pre-split word becomes a list of characters. An end-of-word marker `</w>` can distinguish word boundaries.

**Byte-based (#2):** Start from 256 single-byte symbols (one per possible byte value).

### Core Training Loop

1. Count how often adjacent pairs of symbols appear across the entire training corpus
2. Find the most frequent pair (X, Y)
3. Create a new symbol Z = XY and merge every occurrence of (X, Y) into Z
4. Update pair counts affected by that change (only local neighbors)
5. Repeat until desired vocabulary size is reached or most frequent pair falls below minimum frequency

### Outputs

- **merges.txt**: One merge per line, in the exact order learned
- **vocab.json**: Mapping from each symbol to numeric identifier, including special tokens like `<unknown>`, `<begin>`, `<end>` with reserved IDs
- **model.json**: (Optional) Single file containing vocab, merge list, and config

Why keep all three?
- `merges.txt` mirrors papers/blog posts and is easy to diff as you iterate.
- `vocab.json` is what the encoder/decoder use at runtime.
- `model.json` binds config + vocab + merges together so a single file can reproduce a training run later.

### Performance Tips

- Store each word/byte sequence as a list of integers for speed
- Use max-heap keyed by pair frequency for O(log n) retrieval
- Handle stale heap entries lazily (verify frequency when popping)
- When merging X Y → Z, only update pair counts for local neighbors
- **First version: correctness beats speed.** Profile, then optimize.

## Encoding and Decoding

### Encoding Process

**Character-based:**
1. Split text using same normalization as training
2. Represent words as character sequences
3. Repeatedly apply merge rules in learned order
4. Map symbols to numeric identifiers

**Byte-based:**
1. Convert text to UTF-8 bytes
2. Map to visible byte symbols
3. Apply merge rules greedily
4. Map to numeric identifiers

### Decoding Process

1. Map numeric identifiers back to symbols
2. Split merged symbols back into characters (#1) or bytes (#2)
3. Join characters into text, or convert bytes back to UTF-8
4. Revert visible space markers back to spaces

**Goal:** `decode(encode(text)) == text` for all inputs

## Configuration

```python
from pathlib import Path
from typing import Literal

class TokenizerConfig(BaseModel):
    model_kind: Literal["character_bpe", "byte_bpe"] = "character_bpe"
    target_vocabulary_size: int = 16000
    minimum_pair_frequency: int = 2
    max_merges: int | None = None
    random_seed: int = 13
    progress_every: int = 100
    unicode_normalization: Literal["NFC", "NFKC", "NFD", "NFKD", "none"] = "NFC"
    strip_accents: bool = False
    lowercase: bool = False
    add_visible_space: bool = True
    specials: list[str] = ["<unknown>", "<begin>", "<end>"]
    debug_first_merges: bool = False
    data_path: Path = Path("./data/train.txt")
    valid_path: Path = Path("./data/valid.txt")
    output_dir: Path = Path("./out")
    model_path: Path = Path("./out/model.json")
```

- **Training termination:** Training stops as soon as the vocab reaches `target_vocabulary_size`, the top pair frequency dips below `minimum_pair_frequency`, or the optional `max_merges` limit is hit—whichever happens first—so runs remain bounded and merges stay meaningful.
- **Deterministic progress reporting:** `random_seed` controls all stochastic work, and `progress_every` ensures trainer logs fire on repeatable intervals for the same seed/input.
- **Debugging:** `debug_first_merges` controls whether training prints the top few pairs and counts after the first merges—useful for learning and debugging, but off by default for performance.
- **Normalization toggles:** `unicode_normalization`, `strip_accents`, `lowercase`, and `add_visible_space` must be identical during training and inference; they implement the normalization/pretokenization rules described earlier to guarantee round-trip decoding.
- **Special tokens:** `specials` orders the reserved strings; the implementation derives an ID map like `{"<unknown>": 0, "<begin>": 1, "<end>": 2}` so these IDs stay fixed, skip merge operations, and serialize consistently.
- **Paths:** `data_path`, `valid_path`, `output_dir`, and `model_path` centralize where corpora are read and artifacts written, matching the default project tree.

> **Try this:** flip one flag at a time (e.g., set `lowercase=True`). Retrain, run the round-trip tests, and note exactly what changed. This habit builds intuition about how preprocessing choices ripple through the tokenizer.

## Model Artifacts at a Glance

| File                | What it contains                                                                 | Why it matters                                                                                                               |
|---------------------|-----------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| `vocab.json`        | The `Vocabulary` data structure: ordered list of tokens plus the implied token→id map. | Encoding/decoding lookups pull from here. Specials sit at the front so `<unknown>` stays id 0, `<begin>` id 1, etc.          |
| `merges.txt`        | Line-by-line list of `left right` pairs; each line number is the merge’s rank.    | Applying merges in this exact order is how you reproduce training at inference time.                                        |
| `model.json`        | Bundle of `config`, `vocab`, and `merges`.                                        | Single-file checkpoint for the CLI/API; guarantees you can reload the tokenizer without tracking multiple paths.            |

**Vocabulary recap:** think of it as two synchronized views of the same table—`token_to_id` (dict for fast lookups) and `id_to_token` (list to turn ids back into strings). Building it via `Vocabulary.from_tokens(config.specials)` seeds the reserved slots, then the trainer keeps appending new symbols as merges are learned.

**MergeRule recap:** each merge is stored as a `MergeRule(left="t", right="h", rank=42)`. The `rank` tells the encoder “apply this after the previous 42 merges have already been considered,” which keeps the behavior deterministic even if multiple pairs tie for frequency.

## Project Structure

```
bpe-tokenizer/
  bpetok/
    __init__.py
    model.py          # Data classes, file I/O, configuration
    normalize.py      # Text/byte normalization and splitting
    trainer.py        # BPE learning loop
    encode_decode.py  # Encode/decode logic
    metrics.py        # Measurements and benchmarks
    cli.py            # Command-line interface
    utils.py          # Counters, heaps, fast pair updates
  tests/
    test_roundtrip.py
    test_merges_order.py
    test_metrics.py
  examples/
    quick_start.ipynb
  data/
    train.txt
    valid.txt
  out/
    vocab.json
    merges.txt
    model.json
  pyproject.toml
  README.md
  LICENSE
```

## CLI Usage

Each command mirrors a phase in the learning journey:
- `train`: learn merges from scratch
- `encode`: apply merges to raw text
- `decode`: turn token IDs back into readable text
- `stats`: inspect throughput/quality numbers

```bash
# Train a tokenizer
bpetok train --data data/train.txt --valid data/valid.txt --out out/ --config bpe.yaml

# Encode text
bpetok encode --model out/model.json --in sample.txt --out sample.tok

# Decode tokens
bpetok decode --model out/model.json --in sample.tok --out sample.txt

# Display statistics
bpetok stats --model out/model.json --in sample.txt
```

## Python API

```python
from bpetok import Tokenizer

# Load trained model
tok = Tokenizer.load("out/model.json")

# Encode text to token IDs
ids = tok.encode("Hello, world!")

# Decode token IDs back to text
text = tok.decode(ids)
```

## Testing Strategy

### Round-Trip Tests

Test perfect reconstruction on:
- Plain English text
- Heavy punctuation
- Emojis and Unicode symbols
- Multiple languages (especially for byte-level)
- Edge cases: empty strings, single characters

### Core Functionality Tests

- Space handling with and without visible space markers
- Special symbols maintain fixed identifiers and never get merged
- Reproducibility: same config + seed → identical `merges.txt`
- Byte-level: random Unicode strings round-trip perfectly

## Benchmarks

### Performance Metrics

- **Encoding speed**: tokens per second
- **Decoding speed**: tokens per second
- **Average tokens per word**: lower is better
- **Compression ratio** compared to baselines:
  - Character-level tokenizer (every character is a token)
  - Whitespace word tokenizer

### Quality Metrics

- Round-trip accuracy: `decode(encode(x)) == x` (should be 100%)
- Vocabulary coverage on validation set
- Stability: retrain with same config/seed produces identical results

## Common Pitfalls

1. **Inconsistent normalization**: Using different normalization rules between training and inference breaks decoding
2. **Unpinned special tokens**: Forgetting to reserve fixed IDs for special symbols
3. **Inefficient pair counting**: Recomputing all pair counts from scratch every step is unbearably slow
4. **Ambiguous space handling**: Be explicit about how spaces are represented and processed
5. **Byte-to-string conversion errors**: Ensure proper UTF-8 encoding/decoding for byte-level approach

## Dataset

Primary training dataset: **WikiText-2**

A collection of high-quality Wikipedia articles, ideal for learning subword tokenization patterns.

## Step-by-Step Learning Path

1. **Warm-up (Concepts):** read the “BPE in 60 Seconds” section, then grab a notebook and manually merge a toy corpus by hand.
2. **Normalization lab:** implement the toggles in `normalize.py`, then feed the same sentence through every combination to see how token counts change.
3. **Trainer focus:** instrument `trainer.py` with prints every `progress_every` merges; confirm the merge list matches your expectations for simple corpora.
4. **Encoder/decoder pairing:** after wiring up `encode_decode.py`, write tiny scripts that call `decode(encode(text))` for tricky Unicode strings.
5. **Benchmarks & metrics:** use `metrics.py` + CLI `stats` to compare your tokenizer against naïve baselines (character-level, whitespace).
6. **Stretch goals:** re-train with the byte-level mode, add new specials, or plug in a different dataset. Keep notes about what each change taught you.

Throughout, remember the mantra:
1. **Clarity** over cleverness
2. **Correctness** over optimization
3. **Understanding** over black-box solutions

Profile second. Make it work, make it right, then make it fast.

## License

MIT License - See LICENSE file for details
