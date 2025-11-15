"""Test merge order consistency and reproducibility."""

from pathlib import Path

from bpetok.model import TokenizerConfig
from bpetok.trainer import train

SAMPLE_LINES = [
    "hello world",
    "hello there general kenobi",
    "world peace now",
    "byte pair encoding demo",
]


def _write_corpus(tmp_path: Path) -> Path:
    data_path = tmp_path / "train.txt"
    data_path.write_text("\n".join(SAMPLE_LINES), encoding="utf-8")
    return data_path


def test_merge_order_is_deterministic(tmp_path: Path) -> None:
    """Training twice with the same config should yield identical merge order."""
    data_path = _write_corpus(tmp_path)

    def run(out_dir: Path):
        cfg = TokenizerConfig(
            data_path=data_path,
            valid_path=data_path,
            output_dir=out_dir,
            model_path=out_dir / "model.json",
            target_vocabulary_size=200,
            max_merges=10,
            progress_every=1000,
        )
        _, merges = train(cfg)
        return [(m.left, m.right) for m in merges]

    merges_first = run(tmp_path / "out_first")
    merges_second = run(tmp_path / "out_second")

    assert merges_first == merges_second


def test_special_tokens_are_not_merged(tmp_path: Path) -> None:
    """Ensure specials never appear inside learned merges."""
    data_path = _write_corpus(tmp_path)
    out_dir = tmp_path / "out_specials"
    cfg = TokenizerConfig(
        data_path=data_path,
        valid_path=data_path,
        output_dir=out_dir,
        model_path=out_dir / "model.json",
        target_vocabulary_size=200,
        max_merges=15,
    )
    _, merges = train(cfg)
    special_set = set(cfg.specials)
    for merge in merges:
        assert merge.left not in special_set
        assert merge.right not in special_set
