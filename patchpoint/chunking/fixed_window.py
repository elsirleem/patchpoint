"""Splits a file into fixed-size, overlapping line windows.

The Day 0 default chunker — see patchpoint.chunking.ast_chunk for the planned
AST-aware replacement.
"""

from __future__ import annotations

from patchpoint.chunking.base import Chunk


class FixedWindowChunker:
    def __init__(self, window_lines: int = 60, overlap_lines: int = 10) -> None:
        self.window_lines = window_lines
        self.overlap_lines = overlap_lines

    def chunk(self, path: str, content: str) -> list[Chunk]:
        lines = content.splitlines()
        if not lines:
            return []

        step = max(1, self.window_lines - self.overlap_lines)
        chunks = []
        for start in range(0, len(lines), step):
            end = min(start + self.window_lines, len(lines))
            chunks.append(
                Chunk(
                    path=path,
                    start_line=start + 1,
                    end_line=end,
                    text="\n".join(lines[start:end]),
                )
            )
            if end == len(lines):
                break
        return chunks
