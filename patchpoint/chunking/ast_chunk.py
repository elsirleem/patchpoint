"""AST-aware chunking: one chunk per top-level function/class definition.

Python-only, since ast.parse only understands Python — for any non-.py file, or
a .py file that fails to parse (old flask history has some Python 2 syntax),
falls back to FixedWindowChunker so every file still gets chunked somehow.
Without that fallback, files that happen to parse would get fine-grained
function-level chunks while files that don't get one giant whole-file chunk,
which would confound a fixed-window-vs-AST comparison with a fallback-rate
difference rather than a real chunking-strategy difference.

Nested functions/classes aren't chunked separately — the enclosing top-level
def/class is one chunk, matching the granularity described in this module's own
original stub docstring ("functions/classes as chunk boundaries").
"""

from __future__ import annotations

import ast

from patchpoint.chunking.base import Chunk
from patchpoint.chunking.fixed_window import FixedWindowChunker

_fallback = FixedWindowChunker()


class ASTChunker:
    def chunk(self, path: str, content: str) -> list[Chunk]:
        if not path.endswith(".py"):
            return _fallback.chunk(path, content)

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return _fallback.chunk(path, content)

        top_level_defs = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        ]
        if not top_level_defs:
            return _fallback.chunk(path, content)

        lines = content.splitlines()
        chunks: list[Chunk] = []
        cursor = 1  # 1-indexed: next line not yet claimed by a chunk

        def add_chunk(start: int, end: int) -> None:
            text = "\n".join(lines[start - 1 : end])
            if text.strip():
                chunks.append(Chunk(path=path, start_line=start, end_line=end, text=text))

        for node in top_level_defs:
            start, end = node.lineno, node.end_lineno or node.lineno
            if start > cursor:
                # module-level code before this def: imports, docstring, constants
                add_chunk(cursor, start - 1)
            add_chunk(start, end)
            cursor = end + 1

        if cursor <= len(lines):
            add_chunk(cursor, len(lines))

        return chunks
