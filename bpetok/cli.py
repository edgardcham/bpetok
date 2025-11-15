"""Command-line interface for BPE tokenizer."""

import json
from pathlib import Path
from typing import Optional

import typer

from .encode_decode import Tokenizer as RuntimeTokenizer
from .model import TokenizerConfig
from .trainer import train as train_model

app = typer.Typer(help="BPETok - Byte Pair Encoding Tokenizer")


def _read_config_file(path: Path) -> dict:
    """
    Read a config file.

    Args:
        path: Path to config file.

    Returns:
        dict: Config as a dictionary.
    """
    text = path.read_text()
    if path.suffix in {".json"}:
        return json.loads(text)
    if path.suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise typer.BadParameter(
                "YAML config requested but PyYAML is not installed. "
                "Install it or provide a JSON config."
            ) from exc
        return yaml.safe_load(text)
    raise typer.BadParameter(f"Unsupported config format: {path.suffix}")


def _load_config(
    config_path: Optional[Path], data: Path, valid: Optional[Path], out: Path, model: Optional[Path]
) -> TokenizerConfig:
    """
    Load a tokenizer config from a file or default values.

    Args:
        config_path: Path to config file.
        data: Path to training data.
        valid: Path to validation data.
        out: Output directory.
        model: Path where model.json should be written (defaults to out/model.json).

    Returns:
        TokenizerConfig: Tokenizer config.
    """
    base = (
        TokenizerConfig.model_validate(_read_config_file(config_path))
        if config_path
        else TokenizerConfig()
    )
    updates = {
        "data_path": data,
        "output_dir": out,
        "model_path": model or (out / "model.json"),
    }
    if valid:
        updates["valid_path"] = valid
    return base.model_copy(update=updates)


@app.command()
def train(
    data: Path = typer.Option(..., exists=True, dir_okay=False, help="Path to training data"),
    valid: Optional[Path] = typer.Option(
        None, exists=True, dir_okay=False, help="Path to validation data"
    ),
    out: Path = typer.Option(Path("out/"), help="Output directory"),
    config: Optional[Path] = typer.Option(None, help="Path to config JSON/YAML file"),
    model_path: Optional[Path] = typer.Option(
        None, help="Path where model.json should be written (defaults to out/model.json)"
    ),
):
    """Train a BPE tokenizer on the provided corpus."""
    cfg = _load_config(config, data, valid, out, model_path)
    typer.echo(f"Training tokenizer on {cfg.data_path} …")
    vocab, merges = train_model(cfg)
    typer.echo(f"Training complete. Vocab size: {len(vocab)}, merges learned: {len(merges)}")


@app.command()
def encode(
    model: Path = typer.Option(..., exists=True, dir_okay=False, help="Path to trained model.json"),
    input: Path = typer.Option(..., "--in", exists=True, dir_okay=False, help="Input text file"),
    output: Path = typer.Option(..., "--out", dir_okay=False, help="Output file for token IDs"),
):
    """Encode text to token IDs."""
    tokenizer = RuntimeTokenizer.load(model)
    text = input.read_text(encoding="utf-8")
    ids = tokenizer.encode(text)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(" ".join(str(idx) for idx in ids), encoding="utf-8")
    typer.echo(f"Wrote {len(ids)} tokens to {output}")


@app.command()
def decode(
    model: Path = typer.Option(..., exists=True, dir_okay=False, help="Path to trained model.json"),
    input: Path = typer.Option(..., "--in", exists=True, dir_okay=False, help="Input token file"),
    output: Path = typer.Option(..., "--out", dir_okay=False, help="Output text file"),
):
    """Decode token IDs back to text."""
    tokenizer = RuntimeTokenizer.load(model)
    raw = input.read_text(encoding="utf-8").strip()
    ids = [int(part) for part in raw.split()] if raw else []
    text = tokenizer.decode(ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    typer.echo(f"Wrote decoded text to {output}")


@app.command()
def stats(
    model: Path = typer.Option(..., exists=True, dir_okay=False, help="Path to trained model.json"),
    input: Path = typer.Option(
        ..., "--in", exists=True, dir_okay=False, help="Input text file for stats"
    ),
):
    """Display tokenizer statistics and basic metrics."""
    tokenizer = RuntimeTokenizer.load(model)
    text = input.read_text(encoding="utf-8")
    ids = tokenizer.encode(text)
    tokens_per_word = len(ids) / max(len(text.split()), 1)
    typer.echo(f"Vocab size: {len(tokenizer.vocab)}")
    typer.echo(f"Merges learned: {len(tokenizer.merges)}")
    typer.echo(f"Input length: {len(text)} chars, tokens produced: {len(ids)}")
    typer.echo(f"Avg tokens per word: {tokens_per_word:.2f}")


if __name__ == "__main__":
    app()
