from fastapi.testclient import TestClient

from patchpoint import api
from patchpoint.api import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_rank_returns_files_ranked_against_head(monkeypatch) -> None:
    monkeypatch.setattr(api, "resolve_head_sha", lambda repo: "deadbeef")
    monkeypatch.setattr(
        api,
        "index_repo_at_sha",
        lambda repo, sha: {
            "auth.py": "def login(username, password): authenticate user session",
            "billing.py": "def invoice(order): compute total price",
        },
    )

    response = client.post(
        "/rank", json={"repo": "a/b", "issue_text": "login authenticate session", "top_k": 2}
    )

    assert response.status_code == 200
    paths = [f["path"] for f in response.json()]
    assert paths[0] == "auth.py"
