"""Suíte de referência escrita à mão: só serve para provar que cada mutante de mutacoes.py é detectável.

Não faz parte do experimento com a LLM. Valores esperados calculados à mão a partir da docstring de compute_gae.
"""
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ambiente.app import criar_app
from coletor.rollout import ClienteAmbiente, ErroAmbiente, ErroContrato, coletar_rollout


@pytest.fixture
def http():
    with TestClient(criar_app()) as cliente:
        yield cliente


def novo(http, comprimento=5, max_passos=10):
    r = http.post("/episodios", json={"comprimento": comprimento, "max_passos": max_passos})
    assert r.status_code == 201
    return r.json()["id"]


def test_ciclo_de_vida(http):
    eid = novo(http, comprimento=2, max_passos=10)
    assert http.post(f"/episodios/{eid}/passos", json={"acao": 0}).json() == {"observacao": 0, "recompensa": -1.0, "terminated": False, "truncated": False}
    http.post(f"/episodios/{eid}/passos", json={"acao": 1})
    assert http.post(f"/episodios/{eid}/passos", json={"acao": 1}).json() == {"observacao": 2, "recompensa": 10.0, "terminated": True, "truncated": False}
    assert http.post(f"/episodios/{eid}/passos", json={"acao": 1}).status_code == 409
    assert http.delete(f"/episodios/{eid}").status_code == 204
    assert http.get(f"/episodios/{eid}").status_code == 404


def test_truncagem_exata(http):
    eid = novo(http, comprimento=5, max_passos=2)
    assert http.post(f"/episodios/{eid}/passos", json={"acao": 1}).json()["truncated"] is False
    assert http.post(f"/episodios/{eid}/passos", json={"acao": 1}).json()["truncated"] is True


@pytest.mark.parametrize("acao", [2, -1, True, "1"])
def test_acao_invalida(http, acao):
    assert http.post(f"/episodios/{novo(http)}/passos", json={"acao": acao}).status_code == 422


def test_id_inexistente(http):
    assert http.post("/episodios/nao-existe/passos", json={"acao": 1}).status_code == 404


def test_rollout_com_termino(http):
    r = coletar_rollout(ClienteAmbiente(http, 2, 3), lambda o: 1, float, 3, gamma=0.5, gae_lambda=1.0)
    np.testing.assert_array_equal(r.next_values, [1.0, 2.0, 1.0])
    np.testing.assert_array_equal(r.terminated, [False, True, False])
    np.testing.assert_array_equal(r.truncated, [False, False, True])
    np.testing.assert_allclose(r.advantages, [4.0, 9.0, -0.5])
    np.testing.assert_allclose(r.returns, [4.0, 10.0, -0.5])


def test_rollout_com_truncagem_usa_observacao_final(http):
    r = coletar_rollout(ClienteAmbiente(http, 5, 2), lambda o: 1, float, 3, gamma=0.5, gae_lambda=1.0)
    np.testing.assert_array_equal(r.next_values, [1.0, 2.0, 1.0])
    np.testing.assert_array_equal(r.terminated, [False, False, False])
    np.testing.assert_array_equal(r.truncated, [False, True, True])
    np.testing.assert_allclose(r.advantages, [-1.0, -1.0, -0.5])


def test_rollout_remove_episodios(http):
    criados = []
    original = http.request
    def espiao(metodo, url, **kw):
        resp = original(metodo, url, **kw)
        if metodo == "POST" and url == "/episodios":
            criados.append(resp.json()["id"])
        return resp
    http.request = espiao
    coletar_rollout(ClienteAmbiente(http, 2, 3), lambda o: 1, float, 3, gamma=0.5, gae_lambda=1.0)
    assert len(criados) == 2
    assert all(original("GET", f"/episodios/{i}").status_code == 404 for i in criados)


def cliente_falso(handler):
    return ClienteAmbiente(httpx.Client(transport=httpx.MockTransport(handler), base_url="http://amb"), 5, 10)


def test_erro_5xx_vira_erro_ambiente():
    with pytest.raises(ErroAmbiente) as e:
        cliente_falso(lambda req: httpx.Response(503, text="fora")).criar_episodio()
    assert e.value.status == 503


def test_404_vira_erro_ambiente():
    with pytest.raises(ErroAmbiente) as e:
        cliente_falso(lambda req: httpx.Response(404, json={"detail": "x"})).passo("x", 1)
    assert e.value.status == 404


def test_timeout_vira_erro_ambiente():
    def handler(req):
        raise httpx.ReadTimeout("lento", request=req)
    with pytest.raises(ErroAmbiente) as e:
        cliente_falso(handler).criar_episodio()
    assert e.value.status is None


@pytest.mark.parametrize("corpo", [
    {"observacao": 1, "recompensa": -1.0, "terminated": "false", "truncated": False},
    {"observacao": True, "recompensa": -1.0, "terminated": False, "truncated": False},
    {"observacao": 1, "recompensa": -1.0, "terminated": False},
])
def test_contrato_violado(corpo):
    with pytest.raises(ErroContrato):
        cliente_falso(lambda req: httpx.Response(200, json=corpo)).passo("x", 1)


def test_status_de_sucesso_inesperado():
    with pytest.raises(ErroContrato):
        cliente_falso(lambda req: httpx.Response(200, json={"id": "a", "observacao": 0})).criar_episodio()


def test_objetivo_no_ultimo_passo_permitido_marca_os_dois(http):
    eid = novo(http, comprimento=2, max_passos=2)
    http.post(f"/episodios/{eid}/passos", json={"acao": 1})
    assert http.post(f"/episodios/{eid}/passos", json={"acao": 1}).json() == {"observacao": 2, "recompensa": 10.0, "terminated": True, "truncated": True}
