from patchpoint.chunking.fixed_window import FixedWindowChunker


def test_fixed_window_covers_all_lines() -> None:
    content = "\n".join(f"line {i}" for i in range(150))
    chunker = FixedWindowChunker(window_lines=60, overlap_lines=10)
    chunks = chunker.chunk("f.py", content)
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 150


def test_fixed_window_empty_file() -> None:
    chunker = FixedWindowChunker()
    assert chunker.chunk("f.py", "") == []
