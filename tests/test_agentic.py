from patchpoint.retrievers import agentic
from patchpoint.retrievers.agentic import AgenticRetriever


class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _ToolUseBlock:
    type = "tool_use"

    def __init__(self, name: str, input: dict, id: str) -> None:
        self.name = name
        self.input = input
        self.id = id


class _TextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _Response:
    def __init__(self, content: list, input_tokens: int = 10, output_tokens: int = 5) -> None:
        self.content = content
        self.usage = _Usage(input_tokens, output_tokens)


class _ScriptedMessages:
    def __init__(self, responses: list[_Response]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses: list[_Response]) -> None:
        self.messages = _ScriptedMessages(responses)


def _files() -> dict[str, str]:
    return {"auth.py": "def login(): pass", "billing.py": "def charge(): pass"}


def test_agentic_returns_submitted_files_in_order(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    client = _FakeClient(
        [_Response([_ToolUseBlock("submit_answer", {"files": ["auth.py", "billing.py"]}, "t1")])]
    )
    monkeypatch.setattr(agentic, "_get_client", lambda: client)

    retriever = AgenticRetriever()
    retriever.index(_files())
    ranked = retriever.query("login is broken", top_k=10)

    assert [f.path for f in ranked] == ["auth.py", "billing.py"]
    assert ranked[0].score > ranked[1].score  # order preserved via descending scores


def test_agentic_executes_read_file_before_submitting(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    client = _FakeClient(
        [
            _Response([_ToolUseBlock("read_file", {"path": "auth.py"}, "t1")]),
            _Response([_ToolUseBlock("submit_answer", {"files": ["auth.py"]}, "t2")]),
        ]
    )
    monkeypatch.setattr(agentic, "_get_client", lambda: client)

    retriever = AgenticRetriever()
    retriever.index(_files())
    ranked = retriever.query("login is broken", top_k=10)

    assert [f.path for f in ranked] == ["auth.py"]
    assert client.messages.calls == 2


def test_agentic_returns_empty_when_model_never_submits(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    client = _FakeClient([_Response([_TextBlock("I'm not sure.")])])
    monkeypatch.setattr(agentic, "_get_client", lambda: client)

    retriever = AgenticRetriever()
    retriever.index(_files())
    assert retriever.query("login is broken") == []


def test_agentic_stops_at_turn_limit_without_submit(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    # Always asks to read a file, never submits — should stop at _MAX_TURNS, not loop forever.
    responses = [
        _Response([_ToolUseBlock("read_file", {"path": "auth.py"}, f"t{i}")])
        for i in range(agentic._MAX_TURNS)
    ]
    client = _FakeClient(responses)
    monkeypatch.setattr(agentic, "_get_client", lambda: client)

    retriever = AgenticRetriever()
    retriever.index(_files())
    assert retriever.query("login is broken") == []
    assert client.messages.calls == agentic._MAX_TURNS


def test_agentic_caches_and_only_charges_once(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    client = _FakeClient(
        [_Response([_ToolUseBlock("submit_answer", {"files": ["auth.py"]}, "t1")], 100, 50)]
    )
    monkeypatch.setattr(agentic, "_get_client", lambda: client)

    retriever = AgenticRetriever()
    retriever.index(_files())
    retriever.query("login is broken")
    cost_after_first = retriever.cost_usd()
    retriever.query("login is broken")  # identical call — should hit the cache
    cost_after_second = retriever.cost_usd()

    assert client.messages.calls == 1
    assert cost_after_first == cost_after_second
    assert cost_after_first > 0


def test_agentic_unknown_tool_returns_error_result(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    client = _FakeClient(
        [
            _Response([_ToolUseBlock("delete_repo", {}, "t1")]),
            _Response([_ToolUseBlock("submit_answer", {"files": []}, "t2")]),
        ]
    )
    monkeypatch.setattr(agentic, "_get_client", lambda: client)

    retriever = AgenticRetriever()
    retriever.index(_files())
    assert retriever.query("anything") == []  # didn't crash on the bad tool call
