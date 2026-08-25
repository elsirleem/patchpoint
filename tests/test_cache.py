from patchpoint.data.cache import cached_json


def test_cached_json_avoids_second_call(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    calls = []

    @cached_json("ns")
    def fn(x: str) -> dict:
        calls.append(x)
        return {"x": x}

    assert fn("a") == {"x": "a"}
    assert fn("a") == {"x": "a"}
    assert calls == ["a"]
