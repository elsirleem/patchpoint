"""AST-aware chunking (e.g. Python functions/classes as chunk boundaries).

Not yet implemented — see CLAUDE.md status. FixedWindowChunker is the Day 0
default; the target comparison is recall@k, AST vs fixed window.
"""

from __future__ import annotations

from patchpoint.chunking.base import Chunk


class ASTChunker:
    def chunk(self, path: str, content: str) -> list[Chunk]:
        raise NotImplementedError("AST chunking not implemented yet — see CLAUDE.md status")
