# patchpoint

Given a GitHub issue in plain English, rank the repository's files by how likely each is to
need changing. Evaluated on real issue/fix pairs mined from git history.

The deliverable is the measurement, not the app: every retriever is scored with a bootstrap
confidence interval against a chronological dev/test split, never a single number.

## Setup

    pip install -e ".[dev]"
    make demo   # offline smoke test, no network needed

## Usage

    python -m patchpoint.cli build   --repo pallets/flask --max 200
    python -m patchpoint.cli index   --repo pallets/flask --sha <sha>
    python -m patchpoint.cli eval    --retriever hybrid --split dev --repo pallets/flask
    python -m patchpoint.cli compare --a hybrid --b bm25 --metric recall@10
