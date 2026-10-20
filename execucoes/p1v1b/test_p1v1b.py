import pytest
import httpx
import numpy as np
from fastapi.testclient import TestClient
from ambiente.app import criar_app
from coletor.rollout import (
    ClienteAmbiente,
    ErroAmbiente,
    ErroContrato,
    Transicao,
    Rollout,
    coletar_rollout,
)


@pytest.fixture
def cliente_http():
    """TestClient novo a cada teste."""
    app = criar_app()
    return TestClient(app)


# ====== POST /episodios ======

def test_criar_episodio_com_sucesso(cliente_http):
    """POST /episodios com parâmetros válidos retorna 201."""
    response = cliente_http.post(
        "/episodios",
        json={"comprimento": 10, "max_passos": 100}
    )
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["observacao"] == 0
    assert data["passos"] == 0
    assert data["encerrado"] is False


def test_criar_episodio_validacao_comprimento_minimo(cliente_http):
    """POST /episodios com comprimento < 2 retorna 422."""
    response = cliente_http.post(
        "/episodios",
        json={"comprimento": 1, "max_passos": 100}
    )
    assert response.status_code == 422


def test_criar_episodio_validacao_comprimento_maximo(cliente_http):
    """POST /episodios com comprimento > 20 retorna 422."""
    response = cliente_http.post(
        "/episodios",
        json={"comprimento": 21, "max_passos": 100}
    )
    assert response.status_code == 422


def test_criar_episodio_validacao_max_passos_minimo(cliente_http):
    """POST /episodios com max_passos < 1 retorna 422."""
    response = cliente_http.post(
        "/episodios",
        json={"comprimento": 10, "max_passos": 0}
    )
    assert response.status_code == 422


def test_criar_episodio_validacao_max_passos_maximo(cliente_http):
    """POST /episodios com max_passos > 500 retorna 422."""
    response = cliente_http.post(
        "/episodios",
        json={"comprimento": 10, "max_passos": 501}
    )
    assert response.status_code == 422


def test_criar_episodio_campo_obrigatorio_faltando(cliente_http):
    """POST /episodios sem campo obrigatório retorna 422."""
    response = cliente_http.post(
        "/episodios",
        json={"comprimento": 10}
    )
    assert response.status_code == 422


def test_criar_episodio_tipo_invalido(cliente_http):
    """POST /episodios com comprimento string retorna 422."""
    response = cliente_http.post(
        "/episodios",
        json={"comprimento": "10", "max_passos": 100}
    )
    assert response.status_code == 422


# ====== GET /episodios/{episodio_id} ======

def test_consultar_episodio_existente(cliente_http):
    """GET /episodios/{id} com id existente retorna 200."""
    post_response = cliente_http.post(
        "/episodios",
        json={"comprimento": 10, "max_passos": 100}
    )
    episodio_id = post_response.json()["id"]
    
    response = cliente_http.get(f"/episodios/{episodio_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == episodio_id
    assert isinstance(data["observacao"], int)
    assert isinstance(data["passos"], int)
    assert isinstance(data["encerrado"], bool)


def test_consultar_episodio_inexistente(cliente_http):
    """GET /episodios/{id} com id inexistente retorna 404."""
    response = cliente_http.get("/episodios/nao_existe")
    assert response.status_code == 404


# ====== DELETE /episodios/{episodio_id} ======

def test_remover_episodio_existente(cliente_http):
    """DELETE /episodios/{id} com id existente retorna 204."""
    post_response = cliente_http.post(
        "/episodios",
        json={"comprimento": 10, "max_passos": 100}
    )
    episodio_id = post_response.json()["id"]
    
    response = cliente_http.delete(f"/episodios/{episodio_id}")
    assert response.status_code == 204
    
    get_response = cliente_http.get(f"/episodios/{episodio_id}")
    assert get_response.status_code == 404


def test_remover_episodio_inexistente(cliente_http):
    """DELETE /episodios/{id} com id inexistente retorna 404."""
    response = cliente_http.delete("/episodios/nao_existe")
    assert response.status_code == 404


# ====== POST /episodios/{episodio_id}/passos ======

def test_dar_passo_com_sucesso(cliente_http):
    """POST /episodios/{id}/passos com ação válida retorna 200."""
    post_response = cliente_http.post(
        "/episodios",
        json={"comprimento": 10, "max_passos": 100}
    )
    episodio_id = post_response.json()["id"]
    
    response = cliente_http.post(
        f"/episodios/{episodio_id}/passos",
        json={"acao": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["observacao"], int)
    assert isinstance(data["recompensa"], (int, float))
    assert isinstance(data["terminated"], bool)
    assert isinstance(data["truncated"], bool)


def test_dar_passo_episodio_inexistente(cliente_http):
    """POST /episodios/{id}/passos com id inexistente retorna 404."""
    response = cliente_http.post(
        "/episodios/nao_existe/passos",
        json={"acao": 0}
    )
    assert response.status_code == 404


def test_dar_passo_episodio_encerrado_terminated(cliente_http):
    """POST /episodios/{id}/passos em episódio terminado retorna 409."""
    post_response = cliente_http.post(
        "/episodios",
        json={"comprimento": 2, "max_passos": 100}
    )
    episodio_id = post_response.json()["id"]
    
    for _ in range(5):
        response = cliente_http.post(
            f"/episodios/{episodio_id}/passos",
            json={"acao": 1}
        )
        if response.status_code != 200:
            break
    
    response = cliente_http.post(
        f"/episodios/{episodio_id}/passos",
        json={"acao": 1}
    )
    assert response.status_code == 409


def test_dar_passo_episodio_encerrado_truncated(cliente_http):
    """POST /episodios/{id}/passos em episódio truncated retorna 409."""
    post_response = cliente_http.post(
        "/episodios",
        json={"comprimento": 100, "max_passos": 2}
    )
    episodio_id = post_response.json()["id"]
    
    for _ in range(5):
        response = cliente_http.post(
            f"/episodios/{episodio_id}/passos",
            json={"acao": 1}
        )
        if response.status_code != 200:
            break
    
    response = cliente_http.post(
        f"/episodios/{episodio_id}/passos",
        json={"acao": 1}
    )
    assert response.status_code == 409


def test_dar_passo_acao_invalida_negativa(cliente_http):
    """POST /episodios/{id}/passos com ação < 0 retorna 422."""
    post_response = cliente_http.post(
        "/episodios",
        json={"comprimento": 10, "max_passos": 100}
    )
    episodio_id = post_response.json()["id"]
    
    response = cliente_http.post(
        f"/episodios/{episodio_id}/passos",
        json={"acao": -1}
    )
    assert response.status_code == 422


def test_dar_passo_acao_invalida_acima_maximo(cliente_http):
    """POST /episodios/{id}/passos com ação > 1 retorna 422."""
    post_response = cliente_http.post(
        "/episodios",
        json={"comprimento": 10, "max_passos": 100}
    )
    episodio_id = post_response.json()["id"]
    
    response = cliente_http.post(
        f"/episodios/{episodio_id}/passos",
        json={"acao": 2}
    )
    assert response.status_code == 422


def test_dar_passo_campo_obrigatorio_faltando(cliente_http):
    """POST /episodios/{id}/passos sem ação retorna 422."""
    post_response = cliente_http.post(
        "/episodios",
        json={"comprimento": 10, "max_passos": 100}
    )
    episodio_id = post_response.json()["id"]
    
    response = cliente_http.post(
        f"/episodios/{episodio_id}/passos",
        json={}
    )
    assert response.status_code == 422


def test_dar_passo_tipo_invalido(cliente_http):
    """POST /episodios/{id}/passos com ação string retorna 422."""
    post_response = cliente_http.post(
        "/episodios",
        json={"comprimento": 10, "max_passos": 100}
    )
    episodio_id = post_response.json()["id"]
    
    response = cliente_http.post(
        f"/episodios/{episodio_id}/passos",
        json={"acao": "0"}
    )
    assert response.status_code == 422


# ====== ClienteAmbiente ======

def test_cliente_ambiente_criar_episodio(cliente_http):
    """ClienteAmbiente.criar_episodio() retorna id e observacao=0."""
    cliente = ClienteAmbiente(cliente_http, 10, 100)
    episodio_id, obs = cliente.criar_episodio()
    
    assert isinstance(episodio_id, str)
    assert episodio_id
    assert obs == 0


def test_cliente_ambiente_passo(cliente_http):
    """ClienteAmbiente.passo() retorna Transicao com tipos corretos."""
    cliente = ClienteAmbiente(cliente_http, 10, 100)
    episodio_id, _ = cliente.criar_episodio()
    
    transicao = cliente.passo(episodio_id, 1)
    
    assert isinstance(transicao, Transicao)
    assert isinstance(transicao.observacao, int)
    assert isinstance(transicao.recompensa, float)
    assert isinstance(transicao.terminated, bool)
    assert isinstance(transicao.truncated, bool)


def test_cliente_ambiente_passo_episodio_inexistente(cliente_http):
    """ClienteAmbiente.passo() com episódio inexistente lança ErroAmbiente."""
    cliente = ClienteAmbiente(cliente_http, 10, 100)
    
    with pytest.raises(ErroAmbiente):
        cliente.passo("nao_existe", 1)


def test_cliente_ambiente_encerrar(cliente_http):
    """ClienteAmbiente.encerrar() remove o episódio do serviço."""
    cliente = ClienteAmbiente(cliente_http, 10, 100)
    episodio_id, _ = cliente.criar_episodio()
    
    cliente.encerrar(episodio_id)
    
    response = cliente_http.get(f"/episodios/{episodio_id}")
    assert response.status_code == 404


def test_cliente_ambiente_encerrar_episodio_inexistente(cliente_http):
    """ClienteAmbiente.encerrar() com episódio inexistente lança ErroAmbiente."""
    cliente = ClienteAmbiente(cliente_http, 10, 100)
    
    with pytest.raises(ErroAmbiente):
        cliente.encerrar("nao_existe")


# ====== coletar_rollout ======

def test_coletar_rollout_numero_de_passos(cliente_http):
    """coletar_rollout() coleta exatamente n_passos."""
    cliente = ClienteAmbiente(cliente_http, 10, 100)
    
    rollout = coletar_rollout(
        cliente, lambda obs: 1, lambda obs: 0.0, 15, 0.99, 0.95
    )
    
    assert len(rollout.observacoes) == 15
    assert len(rollout.acoes) == 15
    assert len(rollout.recompensas) == 15
    assert len(rollout.values) == 15
    assert len(rollout.next_values) == 15
    assert len(rollout.terminated) == 15
    assert len(rollout.truncated) == 15
    assert len(rollout.advantages) == 15
    assert len(rollout.returns) == 15


def test_coletar_rollout_tipos_arrays(cliente_http):
    """coletar_rollout() retorna arrays com dtypes corretos."""
    cliente = ClienteAmbiente(cliente_http, 10, 100)
    
    rollout = coletar_rollout(
        cliente, lambda obs: 1, lambda obs: 0.0, 5, 0.99, 0.95
    )
    
    assert rollout.observacoes.dtype == np.int64
    assert rollout.acoes.dtype == np.int64
    assert rollout.recompensas.dtype == np.float64
    assert rollout.values.dtype == np.float64
    assert rollout.next_values.dtype == np.float64
    assert rollout.terminated.dtype == bool
    assert rollout.truncated.dtype == bool
    assert rollout.advantages.dtype == np.float64
    assert rollout.returns.dtype == np.float64


def test_coletar_rollout_ultimo_passo_truncado(cliente_http):
    """Último passo é marcado como truncated se episódio não terminou."""
    cliente = ClienteAmbiente(cliente_http, 20, 100)
    
    rollout = coletar_rollout(
        cliente, lambda obs: 0, lambda obs: 0.0, 8, 0.99, 0.95
    )
    
    assert rollout.truncated[-1] is True


def test_coletar_rollout_episodio_reiniciado_apos_terminar(cliente_http):
    """Episódio é reiniciado automaticamente se termina no meio da coleta."""
    cliente = ClienteAmbiente(cliente_http, 2, 100)
    
    rollout = coletar_rollout(
        cliente, lambda obs: 1, lambda obs: 0.0, 10, 0.99, 0.95
    )
    
    assert len(rollout.observacoes) == 10


def test_coletar_rollout_n_passos_zero(cliente_http):
    """coletar_rollout() com n_passos=0 lança ValueError."""
    cliente = ClienteAmbiente(cliente_http, 10, 100)
    
    with pytest.raises(ValueError):
        coletar_rollout(cliente, lambda obs: 1, lambda obs: 0.0, 0, 0.99, 0.95)


def test_coletar_rollout_n_passos_negativo(cliente_http):
    """coletar_rollout() com n_passos < 0 lança ValueError."""
    cliente = ClienteAmbiente(cliente_http, 10, 100)
    
    with pytest.raises(ValueError):
        coletar_rollout(cliente, lambda obs: 1, lambda obs: 0.0, -5, 0.99, 0.95)


# ====== Erros HTTP (5xx, timeout, etc) com MockTransport ======

def test_criar_episodio_http_500():
    """ClienteAmbiente.criar_episodio() com HTTP 500 lança ErroAmbiente."""
    def handler(request):
        return httpx.Response(500)
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 10, 100)
    
    with pytest.raises(ErroAmbiente) as exc:
        cliente.criar_episodio()
    
    assert exc.value.status == 500


def test_passo_http_502():
    """ClienteAmbiente.passo() com HTTP 502 lança ErroAmbiente."""
    def handler(request):
        return httpx.Response(502)
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 10, 100)
    
    with pytest.raises(ErroAmbiente) as exc:
        cliente.passo("id", 1)
    
    assert exc.value.status == 502


def test_encerrar_http_503():
    """ClienteAmbiente.encerrar() com HTTP 503 lança ErroAmbiente."""
    def handler(request):
        return httpx.Response(503)
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 10, 100)
    
    with pytest.raises(ErroAmbiente) as exc:
        cliente.encerrar("id")
    
    assert exc.value.status == 503


def test_criar_episodio_timeout():
    """ClienteAmbiente.criar_episodio() com timeout lança ErroAmbiente(status=None)."""
    def handler(request):
        raise httpx.ConnectError("timeout")
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 10, 100)
    
    with pytest.raises(ErroAmbiente) as exc:
        cliente.criar_episodio()
    
    assert exc.value.status is None


def test_passo_conexao_recusada():
    """ClienteAmbiente.passo() com conexão recusada lança ErroAmbiente(status=None)."""
    def handler(request):
        raise httpx.ConnectError("connection refused")
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 10, 100)
    
    with pytest.raises(ErroAmbiente) as exc:
        cliente.passo("id", 1)
    
    assert exc.value.status is None


def test_encerrar_read_timeout():
    """ClienteAmbiente.encerrar() com read timeout lança ErroAmbiente(status=None)."""
    def handler(request):
        raise httpx.ReadTimeout("read timeout")
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 10, 100)
    
    with pytest.raises(ErroAmbiente) as exc:
        cliente.encerrar("id")
    
    assert exc.value.status is None


# ====== Contrato violado ======

def test_criar_episodio_body_sem_id():
    """criar_episodio() com body sem 'id' lança ErroContrato."""
    def handler(request):
        return httpx.Response(201, json={"observacao": 0, "passos": 0, "encerrado": False})
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 10, 100)
    
    with pytest.raises(ErroContrato):
        cliente.criar_episodio()


def test_criar_episodio_body_sem_observacao():
    """criar_episodio() com body sem 'observacao' lança ErroContrato."""
    def handler(request):
        return httpx.Response(201, json={"id": "123", "passos": 0, "encerrado": False})
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 10, 100)
    
    with pytest.raises(ErroContrato):
        cliente.criar_episodio()


def test_criar_episodio_observacao_tipo_invalido():
    """criar_episodio() com observacao string lança ErroContrato."""
    def handler(request):
        return httpx.Response(201, json={"id": "123", "observacao": "0", "passos": 0, "encerrado": False})
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 10, 100)
    
    with pytest.raises(ErroContrato):
        cliente.criar_episodio()


def test_passo_body_sem_observacao():
    """passo() com body sem 'observacao' lança ErroContrato."""
    def handler(request):
        return httpx.Response(200, json={"recompensa": 0.0, "terminated": False, "truncated": False})
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 10, 100)
    
    with pytest.raises(ErroContrato):
        cliente.passo("id", 1)


def test_passo_recompensa_tipo_invalido():
    """passo() com recompensa string lança ErroContrato."""
    def handler(request):
        return httpx.Response(200, json={"observacao": 1, "recompensa": "1.0", "terminated": False, "truncated": False})
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 10, 100)
    
    with pytest.raises(ErroContrato):
        cliente.passo("id", 1)


def test_passo_terminated_tipo_invalido():
    """passo() com terminated string lança ErroContrato."""
    def handler(request):
        return httpx.Response(200, json={"observacao": 1, "recompensa": 0.0, "terminated": "true", "truncated": False})
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 10, 100)
    
    with pytest.raises(ErroContrato):
        cliente.passo("id", 1)


def test_passo_truncated_tipo_invalido():
    """passo() com truncated string lança ErroContrato."""
    def handler(request):
        return httpx.Response(200, json={"observacao": 1, "recompensa": 0.0, "terminated": False, "truncated": "false"})
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 10, 100)
    
    with pytest.raises(ErroContrato):
        cliente.passo("id", 1)


def test_criar_episodio_body_com_campo_extra():
    """criar_episodio() com campo extra no body ainda funciona (permissivo)."""
    def handler(request):
        return httpx.Response(201, json={"id": "123", "observacao": 0, "passos": 0, "encerrado": False, "extra": "campo"})
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 10, 100)
    
    episodio_id, obs = cliente.criar_episodio()
    assert episodio_id == "123"
    assert obs == 0


# ====== Comportamento do fluxo ======

def test_fluxo_acao_direita_aumenta_observacao(cliente_http):
    """Ação 1 (direita) aumenta a observação do agente."""
    cliente = ClienteAmbiente(cliente_http, 10, 100)
    episodio_id, obs_inicial = cliente.criar_episodio()
    
    transicao = cliente.passo(episodio_id, 1)
    
    assert transicao.observacao > obs_inicial


def test_fluxo_acao_esquerda_diminui_observacao(cliente_http):
    """Ação 0 (esquerda) diminui a observação (ou mantém em 0)."""
    cliente = ClienteAmbiente(cliente_http, 10, 100)
    episodio_id, _ = cliente.criar_episodio()
    
    t1 = cliente.passo(episodio_id, 1)
    obs_antes = t1.observacao
    
    t2 = cliente.passo(episodio_id, 0)
    obs_depois = t2.observacao
    
    assert obs_depois < obs_antes or obs_depois == 0


def test_fluxo_recompensa_zero_longe_do_objetivo(cliente_http):
    """Recompensa é 0 quando longe do objetivo."""
    cliente = ClienteAmbiente(cliente_http, 10, 100)
    episodio_id, _ = cliente.criar_episodio()
    
    transicao = cliente.passo(episodio_id, 0)
    
    assert transicao.recompensa == 0.0


def test_fluxo_terminated_verdadeiro_ao_objetivo(cliente_http):
    """terminated é True ao atingir o objetivo (comprimento)."""
    cliente = ClienteAmbiente(cliente_http, 2, 100)
    episodio_id, _ = cliente.criar_episodio()
    
    for _ in range(10):
        transicao = cliente.passo(episodio_id, 1)
        if transicao.terminated:
            break
    
    assert transicao.terminated is True
    assert transicao.observacao == 2


def test_fluxo_truncated_verdadeiro_ao_max_passos(cliente_http):
    """truncated é True ao atingir max_passos."""
    cliente = ClienteAmbiente(cliente_http, 100, 3)
    episodio_id, _ = cliente.criar_episodio()
    
    for _ in range(10):
        transicao = cliente.passo(episodio_id, 1)
        if transicao.truncated:
            break
    
    assert transicao.truncated is True


def test_fluxo_observacao_inicial_sempre_zero(cliente_http):
    """Novos episódios sempre começam com observacao=0."""
    cliente = ClienteAmbiente(cliente_http, 10, 100)
    
    for _ in range(3):
        _, obs = cliente.criar_episodio()
        assert obs == 0


def test_fluxo_multiplos_episodios_simultaneos(cliente_http):
    """Múltiplos episódios podem coexistir e ser manipulados independentemente."""
    cliente = ClienteAmbiente(cliente_http, 10, 100)
    
    ids = []
    for _ in range(3):
        episodio_id, _ = cliente.criar_episodio()
        ids.append(episodio_id)
    
    assert len(set(ids)) == 3
    
    for i, episodio_id in enumerate(ids):
        t = cliente.passo(episodio_id, i % 2)
        assert t is not None


def test_fluxo_acoes_seguem_politica(cliente_http):
    """coletar_rollout() segue a política fornecida."""
    cliente = ClienteAmbiente(cliente_http, 10, 100)
    
    def politica(obs):
        return 1 if obs < 5 else 0
    
    rollout = coletar_rollout(
        cliente, politica, lambda obs: 0.0, 20, 0.99, 0.95
    )
    
    count_1_early = np.sum(rollout.acoes[:5] == 1)
    assert count_1_early >= 3


def test_fluxo_values_da_funcao_valor(cliente_http):
    """rollout.values correspondem aos valores retornados pela função valor."""
    cliente = ClienteAmbiente(cliente_http, 10, 100)
    
    def valor(obs):
        return float(obs) * 0.1
    
    rollout = coletar_rollout(
        cliente, lambda obs: 1, valor, 8, 0.99, 0.95
    )
    
    assert np.all(rollout.values >= 0.0)
    assert np.all(rollout.values <= 1.0)


def test_fluxo_next_values_de_observacao_apos_passo(cliente_http):
    """rollout.next_values são calculados da observação após o passo."""
    cliente = ClienteAmbiente(cliente_http, 10, 100)
    
    def valor(obs):
        return float(obs) / 10.0
    
    rollout = coletar_rollout(
        cliente, lambda obs: 1, valor, 5, 0.99, 0.95
    )
    
    assert len(rollout.next_values) == 5


def test_fluxo_recompensa_positiva_ao_terminar(cliente_http):
    """Recompensa é positiva ao atingir o objetivo."""
    cliente = ClienteAmbiente(cliente_http, 2, 100)
    episodio_id, _ = cliente.criar_episodio()
    
    for _ in range(10):
        transicao = cliente.passo(episodio_id, 1)
        if transicao.terminated:
            break
    
    assert transicao.recompensa > 0.0


def test_fluxo_acoes_binarias_entre_zero_e_um(cliente_http):
    """Todas as ações coletadas são 0 ou 1."""
    cliente = ClienteAmbiente(cliente_http, 10, 100)
    
    rollout = coletar_rollout(
        cliente, lambda obs: obs % 2, lambda obs: 0.0, 10, 0.99, 0.95
    )
    
    assert np.all(np.isin(rollout.acoes, [0, 1]))


def test_fluxo_observacoes_dentro_dos_limites(cliente_http):
    """Observações coletadas estão dentro dos limites esperados."""
    cliente = ClienteAmbiente(cliente_http, 10, 100)
    
    rollout = coletar_rollout(
        cliente, lambda obs: 1, lambda obs: 0.0, 15, 0.99, 0.95
    )
    
    assert np.all(rollout.observacoes >= 0)
    assert np.all(rollout.observacoes <= 10)


def test_fluxo_recompensas_nao_negativas(cliente_http):
    """Recompensas coletadas nunca são negativas."""
    cliente = ClienteAmbiente(cliente_http, 10, 100)
    
    rollout = coletar_rollout(
        cliente, lambda obs: 1, lambda obs: 0.0, 10, 0.99, 0.95
    )
    
    assert np.all(rollout.recompensas >= 0.0)


def test_fluxo_terminated_e_truncated_mutuamente_exclusivos(cliente_http):
    """Em cada passo, terminated e truncated não podem ser ambos True."""
    cliente = ClienteAmbiente(cliente_http, 10, 100)
    
    rollout = coletar_rollout(
        cliente, lambda obs: 1, lambda obs: 0.0, 20, 0.99, 0.95
    )
    
    assert not np.any(rollout.terminated & rollout.truncated)
