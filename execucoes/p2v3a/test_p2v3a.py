import pytest
import httpx
import numpy as np
from fastapi.testclient import TestClient

from ambiente.app import criar_app
from coletor.rollout import (
    ClienteAmbiente, ErroAmbiente, ErroContrato, coletar_rollout
)


@pytest.fixture
def app_client():
    """Fixture que cria um app novo para cada teste."""
    return TestClient(criar_app())


# ============================================================================
# TESTES DE VALIDAÇÃO DE ENTRADA (422)
# ============================================================================

def test_post_episodios_422_comprimento_abaixo_minimo(app_client):
    """POST /episodios com comprimento < 2 retorna 422."""
    resposta = app_client.post("/episodios", json={"comprimento": 1, "max_passos": 10})
    assert resposta.status_code == 422


def test_post_episodios_422_comprimento_acima_maximo(app_client):
    """POST /episodios com comprimento > 20 retorna 422."""
    resposta = app_client.post("/episodios", json={"comprimento": 21, "max_passos": 10})
    assert resposta.status_code == 422


def test_post_episodios_422_max_passos_abaixo_minimo(app_client):
    """POST /episodios com max_passos < 1 retorna 422."""
    resposta = app_client.post("/episodios", json={"comprimento": 10, "max_passos": 0})
    assert resposta.status_code == 422


def test_post_episodios_422_max_passos_acima_maximo(app_client):
    """POST /episodios com max_passos > 500 retorna 422."""
    resposta = app_client.post("/episodios", json={"comprimento": 10, "max_passos": 501})
    assert resposta.status_code == 422


# ============================================================================
# TESTES DE ERRO HTTP 5XX
# ============================================================================

def test_post_episodios_500_erro_interno():
    """POST /episodios com HTTP 500 lança ErroAmbiente com status=500."""
    def handler_500(request):
        return httpx.Response(500, text="Internal Server Error")
    
    cliente = ClienteAmbiente(
        httpx.Client(transport=httpx.MockTransport(handler_500), base_url="http://ambiente"),
        comprimento=10, max_passos=10
    )
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.criar_episodio()
    assert exc_info.value.status == 500


def test_post_passos_500_erro_interno():
    """POST /episodios/{id}/passos com HTTP 500 lança ErroAmbiente com status=500."""
    def handler_500(request):
        return httpx.Response(500, text="Internal Server Error")
    
    cliente = ClienteAmbiente(
        httpx.Client(transport=httpx.MockTransport(handler_500), base_url="http://ambiente"),
        comprimento=10, max_passos=10
    )
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.passo("fake-id", 0)
    assert exc_info.value.status == 500


# ============================================================================
# TESTES DE TIMEOUT
# ============================================================================

def test_post_episodios_timeout():
    """POST /episodios com timeout lança ErroAmbiente com status=None."""
    def handler_timeout(request):
        raise httpx.ReadTimeout("timeout", request=request)
    
    cliente = ClienteAmbiente(
        httpx.Client(transport=httpx.MockTransport(handler_timeout), base_url="http://ambiente"),
        comprimento=10, max_passos=10
    )
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.criar_episodio()
    assert exc_info.value.status is None


def test_post_passos_timeout():
    """POST /episodios/{id}/passos com timeout lança ErroAmbiente com status=None."""
    def handler_timeout(request):
        raise httpx.ReadTimeout("timeout", request=request)
    
    cliente = ClienteAmbiente(
        httpx.Client(transport=httpx.MockTransport(handler_timeout), base_url="http://ambiente"),
        comprimento=10, max_passos=10
    )
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.passo("fake-id", 0)
    assert exc_info.value.status is None


# ============================================================================
# TESTES DE FALHA DE CONEXÃO
# ============================================================================

def test_post_episodios_conexao_recusada():
    """POST /episodios com conexão recusada lança ErroAmbiente com status=None."""
    def handler_error(request):
        raise httpx.ConnectError("conexão recusada", request=request)
    
    cliente = ClienteAmbiente(
        httpx.Client(transport=httpx.MockTransport(handler_error), base_url="http://ambiente"),
        comprimento=10, max_passos=10
    )
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.criar_episodio()
    assert exc_info.value.status is None


# ============================================================================
# TESTES DE VIOLAÇÃO DE CONTRATO - POST /episodios
# ============================================================================

def test_post_episodios_contrato_id_ausente():
    """POST /episodios sem 'id' na resposta lança ErroContrato."""
    def handler_sem_id(request):
        return httpx.Response(201, json={"observacao": 0, "passos": 0, "encerrado": False})
    
    cliente = ClienteAmbiente(
        httpx.Client(transport=httpx.MockTransport(handler_sem_id), base_url="http://ambiente"),
        comprimento=10, max_passos=10
    )
    with pytest.raises(ErroContrato):
        cliente.criar_episodio()


def test_post_episodios_contrato_observacao_ausente():
    """POST /episodios sem 'observacao' na resposta lança ErroContrato."""
    def handler_sem_obs(request):
        return httpx.Response(201, json={"id": "123", "passos": 0, "encerrado": False})
    
    cliente = ClienteAmbiente(
        httpx.Client(transport=httpx.MockTransport(handler_sem_obs), base_url="http://ambiente"),
        comprimento=10, max_passos=10
    )
    with pytest.raises(ErroContrato):
        cliente.criar_episodio()


def test_post_episodios_contrato_id_tipo_errado():
    """POST /episodios com 'id' como int lança ErroContrato."""
    def handler_id_int(request):
        return httpx.Response(201, json={"id": 123, "observacao": 0, "passos": 0, "encerrado": False})
    
    cliente = ClienteAmbiente(
        httpx.Client(transport=httpx.MockTransport(handler_id_int), base_url="http://ambiente"),
        comprimento=10, max_passos=10
    )
    with pytest.raises(ErroContrato):
        cliente.criar_episodio()


def test_post_episodios_contrato_observacao_tipo_bool():
    """POST /episodios com 'observacao' como bool lança ErroContrato."""
    def handler_obs_bool(request):
        return httpx.Response(201, json={"id": "123", "observacao": True, "passos": 0, "encerrado": False})
    
    cliente = ClienteAmbiente(
        httpx.Client(transport=httpx.MockTransport(handler_obs_bool), base_url="http://ambiente"),
        comprimento=10, max_passos=10
    )
    with pytest.raises(ErroContrato):
        cliente.criar_episodio()


# ============================================================================
# TESTES DE VIOLAÇÃO DE CONTRATO - POST /episodios/{id}/passos
# ============================================================================

def test_post_passos_contrato_observacao_ausente():
    """POST /episodios/{id}/passos sem 'observacao' lança ErroContrato."""
    def handler_sem_obs(request):
        return httpx.Response(200, json={"recompensa": -1.0, "terminated": False, "truncated": False})
    
    cliente = ClienteAmbiente(
        httpx.Client(transport=httpx.MockTransport(handler_sem_obs), base_url="http://ambiente"),
        comprimento=10, max_passos=10
    )
    with pytest.raises(ErroContrato):
        cliente.passo("fake-id", 0)


def test_post_passos_contrato_recompensa_ausente():
    """POST /episodios/{id}/passos sem 'recompensa' lança ErroContrato."""
    def handler_sem_recompensa(request):
        return httpx.Response(200, json={"observacao": 1, "terminated": False, "truncated": False})
    
    cliente = ClienteAmbiente(
        httpx.Client(transport=httpx.MockTransport(handler_sem_recompensa), base_url="http://ambiente"),
        comprimento=10, max_passos=10
    )
    with pytest.raises(ErroContrato):
        cliente.passo("fake-id", 0)


def test_post_passos_contrato_terminated_ausente():
    """POST /episodios/{id}/passos sem 'terminated' lança ErroContrato."""
    def handler_sem_terminated(request):
        return httpx.Response(200, json={"observacao": 1, "recompensa": -1.0, "truncated": False})
    
    cliente = ClienteAmbiente(
        httpx.Client(transport=httpx.MockTransport(handler_sem_terminated), base_url="http://ambiente"),
        comprimento=10, max_passos=10
    )
    with pytest.raises(ErroContrato):
        cliente.passo("fake-id", 0)


def test_post_passos_contrato_truncated_ausente():
    """POST /episodios/{id}/passos sem 'truncated' lança ErroContrato."""
    def handler_sem_truncated(request):
        return httpx.Response(200, json={"observacao": 1, "recompensa": -1.0, "terminated": False})
    
    cliente = ClienteAmbiente(
        httpx.Client(transport=httpx.MockTransport(handler_sem_truncated), base_url="http://ambiente"),
        comprimento=10, max_passos=10
    )
    with pytest.raises(ErroContrato):
        cliente.passo("fake-id", 0)


def test_post_passos_contrato_recompensa_tipo_bool():
    """POST /episodios/{id}/passos com 'recompensa' como bool lança ErroContrato."""
    def handler_recompensa_bool(request):
        return httpx.Response(200, json={
            "observacao": 1, "recompensa": True, "terminated": False, "truncated": False
        })
    
    cliente = ClienteAmbiente(
        httpx.Client(transport=httpx.MockTransport(handler_recompensa_bool), base_url="http://ambiente"),
        comprimento=10, max_passos=10
    )
    with pytest.raises(ErroContrato):
        cliente.passo("fake-id", 0)


def test_post_passos_contrato_terminated_tipo_int():
    """POST /episodios/{id}/passos com 'terminated' como int lança ErroContrato."""
    def handler_terminated_int(request):
        return httpx.Response(200, json={
            "observacao": 1, "recompensa": -1.0, "terminated": 1, "truncated": False
        })
    
    cliente = ClienteAmbiente(
        httpx.Client(transport=httpx.MockTransport(handler_terminated_int), base_url="http://ambiente"),
        comprimento=10, max_passos=10
    )
    with pytest.raises(ErroContrato):
        cliente.passo("fake-id", 0)


# ============================================================================
# TESTES DE ERRO HTTP 4XX
# ============================================================================

def test_post_passos_404_episodio_inexistente(app_client):
    """POST /episodios/{id}/passos com episódio inexistente lança ErroAmbiente status=404."""
    cliente = ClienteAmbiente(app_client, comprimento=10, max_passos=10)
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.passo("episodio-inexistente", 0)
    assert exc_info.value.status == 404


def test_post_passos_409_episodio_encerrado(app_client):
    """POST /episodios/{id}/passos com episódio terminado lança ErroAmbiente status=409."""
    cliente = ClienteAmbiente(app_client, comprimento=2, max_passos=10)
    episodio_id, obs = cliente.criar_episodio()
    # Passo 0: obs 0 -> 1
    cliente.passo(episodio_id, 1)
    # Passo 1: obs 1 -> 2, terminated=True
    cliente.passo(episodio_id, 1)
    # Tentar dar mais um passo deve falhar com 409
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.passo(episodio_id, 0)
    assert exc_info.value.status == 409


def test_delete_episodio_404_inexistente(app_client):
    """DELETE /episodios/{id} com episódio inexistente lança ErroAmbiente status=404."""
    cliente = ClienteAmbiente(app_client, comprimento=10, max_passos=10)
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.encerrar("episodio-inexistente")
    assert exc_info.value.status == 404


# ============================================================================
# TESTES DE REGRAS DE FRONTEIRA
# ============================================================================

def test_terminated_next_values_usa_obs_final(app_client):
    """Quando terminated=True, next_values[t] usa obs final do episódio, não do novo.
    
    Cenário: comprimento=2, n_passos=2, política sempre ação 1 (direita).
    - Passo 0: obs 0 -> 1, recompensa -1, terminated=False
    - Passo 1: obs 1 -> 2, recompensa +10, terminated=True
    next_values[1] = valor(2), não valor(0) do novo episódio.
    """
    cliente = ClienteAmbiente(app_client, comprimento=2, max_passos=10)
    
    valor = lambda obs: float(obs)
    politica = lambda obs: 1
    
    rollout = coletar_rollout(cliente, politica, valor, n_passos=2, gamma=0.99, gae_lambda=0.95)
    
    assert rollout.observacoes.tolist() == [0, 1]
    assert rollout.acoes.tolist() == [1, 1]
    assert rollout.recompensas.tolist() == [-1.0, 10.0]
    assert rollout.values.tolist() == [0.0, 1.0]
    assert rollout.next_values.tolist() == [1.0, 2.0]
    np.testing.assert_array_equal(rollout.terminated, [False, True])
    np.testing.assert_array_equal(rollout.truncated, [False, True])


def test_truncated_next_values_usa_obs_final(app_client):
    """Quando truncated=True em meio ao rollout, next_values usa obs final, não a do novo.
    
    Cenário: comprimento=10, max_passos=2, n_passos=3, política sempre ação 0.
    - Passo 0: obs 0 -> 0, recompensa -1, truncated=False
    - Passo 1: obs 0 -> 0, recompensa -1, truncated=True (passos >= max_passos)
      Novo episódio criado.
    - Passo 2: obs 0 -> 0, recompensa -1, truncated=False
    next_values[1] usa obs final do passo 1, não obs inicial do novo episódio.
    """
    cliente = ClienteAmbiente(app_client, comprimento=10, max_passos=2)
    
    valor = lambda obs: float(obs) + 1.0
    politica = lambda obs: 0
    
    rollout = coletar_rollout(cliente, politica, valor, n_passos=3, gamma=0.99, gae_lambda=0.95)
    
    assert rollout.observacoes.tolist() == [0, 0, 0]
    assert rollout.recompensas.tolist() == [-1.0, -1.0, -1.0]
    assert rollout.values.tolist() == [1.0, 1.0, 1.0]
    assert rollout.next_values.tolist() == [1.0, 1.0, 1.0]
    np.testing.assert_array_equal(rollout.truncated, [False, True, True])


def test_ultimo_passo_truncated_quando_nao_terminado(app_client):
    """O último passo é marcado truncated se não terminou naturalmente.
    
    Cenário: comprimento=10, max_passos=10, n_passos=2, política sempre ação 0.
    - Passo 0: obs 0 -> 0, terminated=False, truncated=False (naturalmente)
    - Passo 1: obs 0 -> 0, terminated=False, truncated=False (naturalmente)
    Mas rollout.truncated[1] é marcado True (rollout interrompe).
    """
    cliente = ClienteAmbiente(app_client, comprimento=10, max_passos=10)
    
    valor = lambda obs: 0.5
    politica = lambda obs: 0
    
    rollout = coletar_rollout(cliente, politica, valor, n_passos=2, gamma=0.99, gae_lambda=0.95)
    
    np.testing.assert_array_equal(rollout.terminated, [False, False])
    np.testing.assert_array_equal(rollout.truncated, [False, True])


def test_episodios_criados_removidos_apos_rollout(app_client):
    """Todos os episódios criados durante rollout são removidos após seu término.
    
    Registra os IDs criados durante a coleta e verifica que GET retorna 404 para cada.
    """
    cliente = ClienteAmbiente(app_client, comprimento=2, max_passos=10)
    
    ids_criados = []
    original_criar = cliente.criar_episodio
    
    def criar_com_registro():
        id_episodio, obs = original_criar()
        ids_criados.append(id_episodio)
        return id_episodio, obs
    
    cliente.criar_episodio = criar_com_registro
    
    valor = lambda obs: 0.0
    politica = lambda obs: 1
    
    rollout = coletar_rollout(cliente, politica, valor, n_passos=3, gamma=0.99, gae_lambda=0.95)
    
    # Todos os episódios criados devem ter sido removidos
    for episodio_id in ids_criados:
        resposta = app_client.get(f"/episodios/{episodio_id}")
        assert resposta.status_code == 404
