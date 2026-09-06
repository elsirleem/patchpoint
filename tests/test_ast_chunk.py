from patchpoint.chunking.ast_chunk import ASTChunker

CODE = '''"""Module docstring."""
import os


def foo():
    return 1


class Bar:
    def method(self):
        return 2
'''


def test_ast_chunk_one_chunk_per_top_level_def() -> None:
    chunks = ASTChunker().chunk("m.py", CODE)

    # module preamble (docstring + import) + foo() + Bar (with its method inside)
    assert len(chunks) == 3
    assert "Module docstring" in chunks[0].text
    assert "def foo" in chunks[1].text
    assert "class Bar" in chunks[2].text
    assert "def method" in chunks[2].text  # nested method stays inside its class's chunk


def test_ast_chunk_covers_every_non_blank_line_exactly_once() -> None:
    # Purely blank/whitespace gaps between defs (e.g. the two blank lines
    # separating foo() from Bar) are intentionally dropped — an empty chunk has
    # no embedding value — so coverage is checked over non-blank lines only.
    chunks = ASTChunker().chunk("m.py", CODE)
    lines = CODE.splitlines()
    non_blank_lines = {i for i, line in enumerate(lines, start=1) if line.strip()}

    covered: set[int] = set()
    for c in chunks:
        chunk_lines = set(range(c.start_line, c.end_line + 1))
        assert not (covered & chunk_lines), "chunks should not overlap"
        covered |= chunk_lines

    assert non_blank_lines <= covered


def test_ast_chunk_falls_back_to_fixed_window_for_non_python_files() -> None:
    content = "\n".join(f"line {i}" for i in range(100))
    chunks = ASTChunker().chunk("notes.rst", content)
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 100
    assert len(chunks) > 1  # fixed-window splits a 100-line file into multiple windows


def test_ast_chunk_falls_back_on_syntax_error() -> None:
    invalid_python = "def broken(:\n    pass\n"
    chunks = ASTChunker().chunk("m.py", invalid_python)
    assert chunks  # didn't crash, produced something via fallback
    assert chunks[0].start_line == 1


def test_ast_chunk_falls_back_when_no_top_level_defs() -> None:
    content = "x = 1\ny = 2\n"
    chunks = ASTChunker().chunk("m.py", content)
    assert chunks[0].text.strip() != ""
