"""Agentic retriever: an LLM that can read files/grep and decide what to change.

A manual tool-use loop (not the SDK's beta tool_runner) because termination is
tied to a specific tool call (submit_answer) plus a turn cap, and cost has to be
accumulated per call — control the runner doesn't expose. Model IDs and pricing
below came from Anthropic's own current documentation (via the claude-api
skill), not memory — see CLAUDE.md invariant 7.

Unlike the other retrievers, this one doesn't score every file — it explores
and decides. recall@k/hit@k only care about top-k set membership, not score
values, so the model's answer order is preserved via descending synthetic
scores; the scores themselves aren't meaningful outside this file's ranking.

Every call costs real money, so a run is cached like anything else in
CLAUDE.md's external-call rule — keyed by (model, the exact set of indexed
files, issue text, top_k). cost_usd() only counts cache misses in the current
process: a fully-cached re-run costs ~$0, which is the point of caching.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from patchpoint.retrievers.base import Retriever
from patchpoint.schemas import ScoredFile

_CACHE_DIR = Path(".cache") / "agentic"
_MAX_TURNS = 6
_MAX_TOKENS = 1024
_MAX_READ_CHARS = 4000
_MAX_GREP_RESULTS = 20

# Source: Anthropic's current pricing docs (via the claude-api skill's model
# reference, cached 2026-06-24) — https://docs.claude.com/en/docs/about-claude/pricing
# Add a model here only once its rate has been checked against current docs.
PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5": {"input_per_mtok": 1.00, "output_per_mtok": 5.00},
}

_SYSTEM_PROMPT = """\
You are picking which files in a software repository are most likely to need
changing to resolve a GitHub issue. You are given the issue and a list of every
file path in the repository (not contents). Use read_file and grep to
investigate specific files or search for relevant terms before deciding — don't
guess from paths alone if you're unsure. When ready, call submit_answer with an
ordered list of file paths, most likely first."""

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": (
            "Read the full contents of one file in the repository, by its path "
            "exactly as it appears in the file list you were given."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "grep",
        "description": (
            "Search every file in the repository for lines matching a regular "
            "expression. Returns up to 20 matches with path and line number."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "submit_answer",
        "description": (
            "Submit your final answer: the file paths most likely to need "
            "changing, ordered from most to least likely. Call this to finish."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"files": {"type": "array", "items": {"type": "string"}}},
            "required": ["files"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def _get_client():  # type: ignore[no-untyped-def]
    import anthropic

    return anthropic.Anthropic()


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Compute (not estimate) cost from PRICING. Raises if the model isn't priced yet."""
    if model_id not in PRICING:
        raise KeyError(
            f"{model_id!r} not in PRICING — add verified rates from provider docs first"
        )
    rates = PRICING[model_id]
    return (
        input_tokens / 1_000_000 * rates["input_per_mtok"]
        + output_tokens / 1_000_000 * rates["output_per_mtok"]
    )


def _files_hash(files: dict[str, str]) -> str:
    h = hashlib.sha256()
    for path in sorted(files):
        h.update(path.encode("utf-8"))
        h.update(b"\0")
        h.update(files[path].encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


class AgenticRetriever(Retriever):
    def __init__(self, model_id: str = "claude-haiku-4-5") -> None:
        self.model_id = model_id
        self._files: dict[str, str] = {}
        self._files_key = ""
        self._run_input_tokens = 0
        self._run_output_tokens = 0

    def index(self, files: dict[str, str]) -> None:
        self._files = files
        self._files_key = _files_hash(files)

    def cost_usd(self) -> float:
        return estimate_cost(self.model_id, self._run_input_tokens, self._run_output_tokens)

    def query(self, issue_text: str, top_k: int = 10) -> list[ScoredFile]:
        cache_path = self._cache_path(issue_text, top_k)
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
        else:
            files, input_tokens, output_tokens = self._run_agent(issue_text, top_k)
            cached = {"files": files, "input_tokens": input_tokens, "output_tokens": output_tokens}
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cached))
            # Only a cache miss reflects real spend in this process — see module
            # docstring; a fully-cached re-run should report ~$0.
            self._run_input_tokens += input_tokens
            self._run_output_tokens += output_tokens

        ranked = cached["files"][:top_k]
        return [ScoredFile(path=p, score=1.0 - i * 0.001) for i, p in enumerate(ranked)]

    def _cache_path(self, issue_text: str, top_k: int) -> Path:
        key = hashlib.sha256(
            f"{self.model_id}\0{self._files_key}\0{issue_text}\0{top_k}".encode()
        ).hexdigest()
        return _CACHE_DIR / f"{key}.json"

    def _run_agent(self, issue_text: str, top_k: int) -> tuple[list[str], int, int]:
        client = _get_client()
        file_list = "\n".join(sorted(self._files))
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    f"Issue:\n{issue_text}\n\n"
                    f"Repository files ({len(self._files)}):\n{file_list}\n\n"
                    f"Pick up to {top_k} files."
                ),
            }
        ]

        input_tokens = 0
        output_tokens = 0

        for _ in range(_MAX_TURNS):
            response = client.messages.create(
                model=self.model_id,
                max_tokens=_MAX_TOKENS,
                system=_SYSTEM_PROMPT,
                tools=_TOOLS,
                messages=messages,
            )
            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            submit = next((b for b in tool_use_blocks if b.name == "submit_answer"), None)
            if submit is not None:
                files = submit.input.get("files", [])
                return [f for f in files if isinstance(f, str)], input_tokens, output_tokens

            if not tool_use_blocks:
                # Model stopped without ever submitting — an honest empty answer,
                # not an error: this is real agent behavior under this harness.
                return [], input_tokens, output_tokens

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in tool_use_blocks:
                content, is_error = self._execute_tool(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": content,
                        "is_error": is_error,
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        return [], input_tokens, output_tokens  # turn limit hit without submit_answer

    def _execute_tool(self, name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
        if name == "read_file":
            path = tool_input.get("path", "")
            content = self._files.get(path)
            if content is None:
                return f"no such file: {path!r}", True
            return content[:_MAX_READ_CHARS], False

        if name == "grep":
            pattern = tool_input.get("pattern", "")
            try:
                regex = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                return f"invalid regex: {e}", True
            matches = self._grep(regex)
            return json.dumps(matches), False

        return f"unknown tool: {name!r}", True

    def _grep(self, regex: re.Pattern[str]) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for path, content in self._files.items():
            for lineno, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    matches.append({"path": path, "line": lineno, "text": line.strip()[:200]})
                    if len(matches) >= _MAX_GREP_RESULTS:
                        return matches
        return matches
