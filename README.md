# BPETok - Byte Pair Encoding Tokenizer

A minimal, educational implementation of Byte Pair Encoding (BPE) tokenization from scratch. This project prioritizes learning and clarity over speed.

## Project Goals

- Build a BPE tokenizer from raw text
- Encode/decode any string with perfect round-trip accuracy
- Provide a tiny, easy-to-use CLI
- Include comprehensive benchmarks: tokens/second, average tokens per word, compression ratio
- Maintain unit tests with reproducible configuration using Pydantic

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
class TokenizerConfig(BaseModel):
    model_kind: Literal["character_bpe", "byte_bpe"] = "character_bpe"
    target_vocabulary_size: int = 16000
    minimum_pair_frequency: int = 2
    lowercase: bool = False
    add_visible_space: bool = True
    specials: list[str] = ["<unknown>", "<begin>", "<end>"]
    random_seed: int = 13
```

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

## Learning Path

This project is designed for educational purposes. The implementation prioritizes:

1. **Clarity** over cleverness
2. **Correctness** over optimization
3. **Understanding** over black-box solutions

Profile first, optimize second. Make it work, make it right, then make it fast.

## License

MIT License - See LICENSE file for details

---

**Ready to build your tokenizer from scratch?** Start with the character-based approach, validate with comprehensive tests, then graduate to byte-level BPE.
