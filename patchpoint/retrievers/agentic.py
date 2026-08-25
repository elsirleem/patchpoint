"""Agentic retriever: an LLM that can read files/grep and decide what to change.

Structure only for Day 0 — no model wired in. PRICING must be filled in and dated
from each provider's current pricing page before this is used; do not fill it from
memory (see CLAUDE.md invariant 7).
"""

from __future__ import annotations

from patchpoint.retrievers.base import Retriever
from patchpoint.schemas import ScoredFile

# {model_id: {"input_per_mtok": float, "output_per_mtok": float}} in USD.
# TODO: populate from provider docs before this retriever is used; leave empty until then.
PRICING: dict[str, dict[str, float]] = {}


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


class AgenticRetriever(Retriever):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._files: dict[str, str] = {}

    def index(self, files: dict[str, str]) -> None:
        self._files = files

    def query(self, issue_text: str, top_k: int = 10) -> list[ScoredFile]:
        raise NotImplementedError("agent loop not wired in yet — see CLAUDE.md status")
