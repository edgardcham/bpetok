"""Command-line interface for BPE tokenizer."""

import typer

app = typer.Typer(help="BPETok - Byte Pair Encoding Tokenizer")


@app.command()
def train(
    data: str = typer.Option(..., help="Path to training data"),
    valid: str = typer.Option(None, help="Path to validation data"),
    out: str = typer.Option("out/", help="Output directory"),
    config: str = typer.Option(None, help="Path to config YAML file"),
):
    """Train a BPE tokenizer on the provided corpus."""
    typer.echo("TODO: Implement training")


@app.command()
def encode(
    model: str = typer.Option(..., help="Path to trained model"),
    input: str = typer.Option(..., "--in", help="Input text file"),
    output: str = typer.Option(..., "--out", help="Output token file"),
):
    """Encode text to token IDs."""
    typer.echo("TODO: Implement encoding")


@app.command()
def decode(
    model: str = typer.Option(..., help="Path to trained model"),
    input: str = typer.Option(..., "--in", help="Input token file"),
    output: str = typer.Option(..., "--out", help="Output text file"),
):
    """Decode token IDs back to text."""
    typer.echo("TODO: Implement decoding")


@app.command()
def stats(
    model: str = typer.Option(..., help="Path to trained model"),
    input: str = typer.Option(..., "--in", help="Input text file for stats"),
):
    """Display tokenizer statistics and benchmarks."""
    typer.echo("TODO: Implement stats")


if __name__ == "__main__":
    app()
