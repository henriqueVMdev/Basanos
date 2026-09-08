import pytest

import server

# Flask recusa registrar rota depois da 1ª requisição: registra no import.


@server.app.route("/__test_boom")
def _boom():
    raise RuntimeError("estourou")


@server.app.route("/__test_huge")
def _huge():
    raise RuntimeError("x" * 5000)


@pytest.fixture
def client():
    return server.app.test_client()


def test_unexpected_error_becomes_json_not_html(client):
    r = client.get("/__test_boom")
    assert r.status_code == 500
    assert r.get_json() == {"error": "estourou"}


def test_error_response_never_leaks_the_traceback(client):
    body = client.get("/__test_boom").get_json()
    assert set(body) == {"error"}          # sem "traceback" na resposta
    assert 'File "' not in body["error"]


def test_huge_upstream_message_is_truncated(client):
    body = client.get("/__test_huge").get_json()
    assert len(body["error"]) == 500


def test_http_errors_keep_their_status_and_stay_json(client):
    r = client.get("/api/rota-que-nao-existe")
    assert r.status_code == 404
    assert "error" in r.get_json()


def test_routes_still_return_their_own_validation_errors(client):
    # 400/404 explícitos das rotas não passam pelo handler
    assert client.post("/api/filter", json={}).status_code == 400
    assert client.post("/api/load", json={"filename": "nao_existe.csv"}).status_code == 404
    assert client.get("/api/files").status_code == 200
