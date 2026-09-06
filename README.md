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

## Results

Dataset: 199 issue/fix pairs mined from `pallets/flask`, chronologically split into 158 dev /
41 test. Examples touching more than 20 files were dropped at build time (one outlier — a
100-file docs rewrite — removed), since no top-*k* retriever can meaningfully cover a change
of that size; see `data/dataset.py`.

**Dev (n=158):**

| retriever | recall@10 | hit@10 |
|---|---|---|
| bm25 | 0.694 | 0.899 |
| hybrid (bm25 + embeddings, RRF) | 0.687 | 0.937 |
| dense (embeddings only) | 0.451 | 0.690 |

**Test (n=41, evaluated once, at the end):**

| retriever | recall@10 | hit@10 |
|---|---|---|
| bm25 | 0.712 | 0.927 |
| hybrid | 0.597 | 0.829 |

**Paired bootstrap comparisons** (`eval/bootstrap.py`, 10,000 resamples):

| comparison | split | metric | diff | 95% CI | p |
|---|---|---|---|---|---|
| bm25 vs hybrid | dev | recall@10 | +0.007 | [-0.045, 0.059] | 0.79 (n.s.) |
| bm25 vs hybrid | dev | hit@10 | -0.038 | [-0.095, 0.013] | 0.20 (n.s.) |
| bm25 vs hybrid | test | recall@10 | +0.132 | [0.037, 0.231] | 0.006 |
| bm25 vs hybrid | test | hit@10 | +0.103 | [0.000, 0.231] | 0.11 (n.s.) |
| bm25 vs dense | dev | recall@10 | +0.244 | [0.171, 0.314] | ≈0 |
| hybrid vs dense | dev | recall@10 | +0.236 | [0.183, 0.293] | ≈0 |

**Reading it honestly**: bm25 and the RRF hybrid are statistically indistinguishable on dev.
On test, bm25 pulls ahead on recall@10 — a real, significant difference, not noise dressed up
as a finding, though test's smaller n (41 vs 158) means it carries a wider CI and less
certainty than the dev result. Both splits agree on the part that matters: fusing in dense
embeddings does not reliably beat bm25 alone on this dataset, and both clearly beat the dense
retriever used by itself. An earlier run with a hashing-based embedding stand-in (see
`retrievers/dense.py`) showed hybrid losing to bm25 by a wide, significant margin — that result
no longer holds once the embeddings are real, which is itself worth noting as a caution about
trusting a fusion result built on a weak or fake second signal.

## Chunking

Whole-file embedding truncates anything past the model's max sequence length, so files were
split before embedding and scored by their best-matching chunk (max over chunks, per file) —
see `retrievers/chunked_embedding.py`. Two chunkers were compared: fixed-size line windows
(`chunking/fixed_window.py`) and one chunk per top-level function/class
(`chunking/ast_chunk.py`, falling back to fixed-window for non-Python files or parse failures).

**Dev (n=158):**

| retriever | recall@10 | hit@10 |
|---|---|---|
| dense-fixed (chunked, fixed-window) | 0.706 | 0.899 |
| dense-ast (chunked, AST) | 0.629 | 0.873 |
| dense (whole-file, no chunking) | 0.451 | 0.690 |

| comparison | metric | diff | 95% CI | p |
|---|---|---|---|---|
| dense-fixed vs dense (whole-file) | recall@10 | +0.255 | [0.198, 0.314] | ≈0 |
| dense-fixed vs dense-ast | recall@10 | +0.077 | [0.038, 0.117] | 0.0002 |
| dense-fixed vs dense-ast | hit@10 | +0.025 | [-0.013, 0.063] | 0.26 (n.s.) |
| dense-fixed vs bm25 | recall@10 | +0.012 | [-0.047, 0.069] | 0.70 (n.s.) |

Chunking at all is an unambiguous win over embedding whole files. Between the two chunkers,
**fixed-window beats AST chunking on recall@10** — the opposite of what was originally assumed
here (see git history). This isn't a truncation story: AST chunks are actually smaller on
average (median 875 vs 1998 characters) and less often exceed the model's ~256-token limit (48%
of chunks vs 89%, sampled directly from the flask checkout). The best-grounded hypothesis is
that AST chunking produced more, smaller chunks per file (181 vs 120 across the `.py` files
sampled) — under max-score aggregation, more independent chunks per file means more chances for
one small, low-context chunk to score spuriously high, adding noise to the ranking. That's a
hypothesis supported by the chunk-count numbers, not a proven mechanism.

The other headline: chunked dense embeddings alone are now statistically tied with bm25
(p=0.70) — a real embedding retriever matching the classical baseline, once it isn't crippled
by silent truncation.

## Failure taxonomy

Grounded in the actual bm25 predictions rather than guessed categories — every example cited
is a real complete miss (hit@10 = 0) from
`results/bm25-{dev,test}-pallets__flask/predictions.jsonl`. 19 of 199 dev+test examples missed
entirely; patterns below, roughly in order of frequency:

1. **Housekeeping files with no content signal.** `CHANGES.rst`, `setup.py`, `tox.ini`, and
   `.travis.yml` show up in roughly a third of all misses (e.g. #2586, #3279, #3941, #4043,
   #4053, #4502) — usually alongside real code files that *were* found. These files get
   touched by nearly every PR as a matter of process, but their content has no lexical
   relationship to the issue that prompted the change; a changelog entry doesn't exist until
   the fix is written. This looks like a property of the dataset's gold-file definition more
   than a retriever weakness — worth a note if reporting recall@k as "how good is retrieval,"
   since a chunk of the ceiling is structurally unreachable.

2. **Right neighborhood, wrong file**, mostly in `docs/`. E.g. #1521 ("better docs for app
   context") has gold `docs/appcontext.rst`, but bm25's top 5 were other doc pages
   (`docs/patterns/celery.rst`, `docs/design.rst`, ...) — it found "this is about docs" but not
   which one, likely because several doc pages share enough vocabulary to compete on pure term
   overlap. Same pattern in #3320, #3981, #4328.

3. **Vocabulary mismatch between symptom and implementation.** E.g. #3225 ("suspicious
   `del <local_variable>` in last line of function") has gold `flask/cli.py`, but the issue
   text never mentions "cli" — it describes a code smell the reporter noticed, not the
   subsystem it lives in. Same for #4570 (a mypy lint error) → `src/flask/cli.py`. This is the
   core limitation of term-overlap retrieval: it can't bridge "this behavior is wrong" to "this
   is the file that causes it" without shared words.

4. **Structural/cross-cutting issues with no single right file.** #2287 ("DRY up the test
   suite using pytest fixtures") has 12 gold files across `tests/` — the issue asks for a
   pattern change across the whole suite, not a localized fix, so no top-10 list was ever
   going to score well here regardless of retriever quality.

## Future work

Deliberately out of scope for now, not overlooked:

- **Agentic retriever** (`retrievers/agentic.py`) — structure only; needs a real model wired
  in and `PRICING` populated from verified provider docs before use. Also needs a real design
  decision the other retrievers didn't: an agent picks files rather than scoring all of them,
  so mapping that onto the existing `top_k`/`ScoredFile` interface isn't free.
- **`hybrid` still fuses bm25 with the whole-file embedding retriever**, not the (stronger,
  chunked) `dense-fixed` — the chunking comparison above happened after `hybrid` was wired up,
  and re-running that fusion with the better dense signal hasn't been done yet.
- **A second repository** — only `pallets/flask` has been evaluated so far.
