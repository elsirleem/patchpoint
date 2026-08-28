"""patchpoint CLI: build the dataset, index a repo, evaluate, and compare retrievers."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from patchpoint.data.checkout import index_repo_at_sha
from patchpoint.data.dataset import build_dataset
from patchpoint.demo_data import DEMO_FILES, DEMO_ISSUE
from patchpoint.eval.compare import compare_runs
from patchpoint.eval.harness import run_eval
from patchpoint.retrievers.bm25 import BM25Retriever
from patchpoint.retrievers.dense import HashingDenseRetriever
from patchpoint.retrievers.hybrid import RRFHybridRetriever
from patchpoint.schemas import Dataset

app = typer.Typer(help="Rank a repo's files by how likely each is to need changing for an issue.")

DATASETS_DIR = Path("data") / "datasets"

RETRIEVERS = {
    "bm25": lambda: BM25Retriever(),
    "dense": lambda: HashingDenseRetriever(),
    "hybrid": lambda: RRFHybridRetriever([BM25Retriever(), HashingDenseRetriever()]),
}


@app.command()
def build(
    repo: str = typer.Option(..., help="owner/name, e.g. pallets/flask"),
    max_examples: int = typer.Option(200, "--max", help="max examples to mine"),
    max_gold_files: int = typer.Option(
        20, "--max-gold-files", help="drop examples touching more files than this"
    ),
) -> None:
    """Mine issue/fix pairs from `repo`'s history and write a dataset file."""
    dataset = build_dataset(repo, max_examples=max_examples, max_gold_files=max_gold_files)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATASETS_DIR / f"{repo.replace('/', '__')}.json"
    out_path.write_text(dataset.model_dump_json(indent=2))
    typer.echo(f"wrote {len(dataset.examples)} examples to {out_path}")


@app.command()
def index(
    repo: str = typer.Option(..., help="owner/name, e.g. pallets/flask"),
    sha: str = typer.Option(..., help="base_sha to index — never HEAD"),
) -> None:
    """Checkout `repo` at `sha` and cache its file contents for retrieval.

    Mainly a manual/debug entry point — `eval` calls the same underlying function
    itself, once per example, since each example has its own base_sha.
    """
    files = index_repo_at_sha(repo, sha)
    typer.echo(f"indexed {len(files)} files from {repo}@{sha[:10]}")


@app.command("eval")
def eval_cmd(
    retriever: str = typer.Option(..., help=f"one of {list(RETRIEVERS)}"),
    split: str = typer.Option("dev", help="'dev' or 'test' — test is touched once, at the end"),
    repo: str = typer.Option(..., help="owner/name, e.g. pallets/flask"),
    top_k: int = 10,
) -> None:
    """Run a retriever over a dataset split and write results/<run_id>/."""
    dataset_path = DATASETS_DIR / f"{repo.replace('/', '__')}.json"
    dataset = Dataset.model_validate_json(dataset_path.read_text())

    run_id = f"{retriever}-{split}-{dataset.repo.replace('/', '__')}"
    run_dir = run_eval(RETRIEVERS[retriever](), dataset, split, run_id, top_k=top_k)
    typer.echo(f"wrote results to {run_dir}")


@app.command("compare")
def compare_cmd(
    a: str = typer.Option(..., help="first run_id under results/"),
    b: str = typer.Option(..., help="second run_id under results/"),
    metric: str = typer.Option("recall@10"),
) -> None:
    """Compare two runs' results/ with the paired bootstrap test."""
    result = compare_runs(a, b, metric)
    typer.echo(json.dumps(result, indent=2))


@app.command()
def demo() -> None:
    """Offline end-to-end smoke test on synthetic data. No network access required."""
    for name, make_retriever in RETRIEVERS.items():
        retriever = make_retriever()
        retriever.index(DEMO_FILES)
        ranked = retriever.query(DEMO_ISSUE, top_k=3)
        typer.echo(f"{name}: {[sf.path for sf in ranked]}")


if __name__ == "__main__":
    app()
