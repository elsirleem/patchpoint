from fastapi.testclient import TestClient

from patchpoint.api import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_rank_not_implemented_yet() -> None:
    response = client.post("/rank", json={"repo": "a/b", "issue_text": "x"})
    assert response.status_code == 501
