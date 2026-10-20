# Raciocínio (máx 25 linhas)

1. **Chamadas HTTP**: POST /episodios (201, corpo JSON), POST /passos (200, corpo JSON), DELETE /episodios (204, sem corpo).

2. **Falhas por chamada**:
   - POST /episodios: HTTP 422 (comprimento ∉ [2,20], max_passos ∉ [1,500], tipos errados), 500, timeout, conexão, status ≠201, campos ausentes/tipos errados (id, observacao, passos, encerrado).
   - POST /passos: HTTP 404 (episódio não existe), 409 (já encerrado), 422 (acao ∉ [0,1], tipo errado), 500, timeout, conexão, status ≠200, campos ausentes/tipos errados.
   - DELETE: HTTP 404 (não existe), 500, timeout, conexão.

3. **Limites de schema**: comprimento 2-20 (StrictInt), max_passos 1-500, acao 0-1.

4. **Regras de fronteira integradas**: 
   - next_values[t] usa observação FINAL do passo (mesmo se terminated/truncated), não do episódio seguinte.
   - Episódios removidos após terminar/truncar (exceto último passo) → GET posterior retorna 404.
   - Último passo sempre truncated=True se não terminou.
   - Recompensa +10 se terminated, -1 caso contrário.

5. **Cenários calculados**:
   - Cenário A (term no meio): comprimento=3, max_passos=10, n_passos=5, politica=right. Passo 2 termina (obs: 0→1→2→3, rewards: -1,-1,+10, term: F,F,T). Novo criado passos 3-4.
   - Cenário B (trunc no meio): comprimento=10, max_passos=2, n_passos=5, politica=right. Passos 0-1 truncam (obs: 0→1→2, rewards: -1,-1), novo criado, passos 2-3 truncam, novo criado, passo 4 último.
   - Campo bool tipo: terminated não deve aceitar 1/0, precisa bool.

```python
import httpx
import numpy as np
import pytest
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
def app_cliente():
    """Fixture que cria um novo TestClient para cada teste."""
    return TestClient(criar_app())


def mock_handler(status_code, body=None, exception=None):
    """Factory para criar handlers Mock que retornam status/corpo ou lançam exceção."""
    def handler(request: httpx.Request) -> httpx.Response | None:
        if exception:
            raise exception
        return httpx.Response(
            status_code=status_code,
            content=body or b"",
            headers={"content-type": "application/json"},
        )
    return handler


# ===== Testes de validação de entrada (HTTP 422) =====

def test_criar_episodio_comprimento_minimo_invalido(app_cliente):
    """HTTP 422 quando comprimento < 2."""
    cliente = ClienteAmbiente(app_cliente, comprimento=1, max_passos=10)
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.criar_episodio()
    assert exc_info.value.status == 422


def test_criar_episodio_comprimento_maximo_invalido(app_cliente):
    """HTTP 422 quando comprimento > 20."""
    cliente = ClienteAmbiente(app_cliente, comprimento=21, max_passos=10)
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.criar_episodio()
    assert exc_info.value.status == 422


def test_criar_episodio_max_passos_minimo_invalido(app_cliente):
    """HTTP 422 quando max_passos < 1."""
    cliente = ClienteAmbiente(app_cliente, comprimento=5, max_passos=0)
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.criar_episodio()
    assert exc_info.value.status == 422


def test_criar_episodio_max_passos_maximo_invalido(app_cliente):
    """HTTP 422 quando max_passos > 500."""
    cliente = ClienteAmbiente(app_cliente, comprimento=5, max_passos=501)
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.criar_episodio()
    assert exc_info.value.status == 422


# ===== Testes de falhas HTTP (4xx, 5xx) =====

def test_passo_episodio_nao_existe(app_cliente):
    """HTTP 404 ao dar passo em episódio inexistente."""
    cliente = ClienteAmbiente(app_cliente, comprimento=5, max_passos=10)
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.passo("episodio_falso", acao=1)
    assert exc_info.value.status == 404


def test_passo_episodio_terminado(app_cliente):
    """HTTP 409 ao dar passo em episódio já terminado."""
    cliente = ClienteAmbiente(app_cliente, comprimento=2, max_passos=10)
    episodio_id, obs = cliente.criar_episodio()
    # Episódio começa em posição 0, comprimento 2. Vai direita: 0→1, depois 1→2 (termina)
    cliente.passo(episodio_id, acao=1)  # 0 → 1
    cliente.passo(episodio_id, acao=1)  # 1 → 2, terminated=True
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.passo(episodio_id, acao=1)
    assert exc_info.value.status == 409


def test_passo_episodio_truncado(app_cliente):
    """HTTP 409 ao dar passo em episódio já truncado (max_passos atingido)."""
    cliente = ClienteAmbiente(app_cliente, comprimento=10, max_passos=1)
    episodio_id, obs = cliente.criar_episodio()
    cliente.passo(episodio_id, acao=0)  # 1 passo, max_passos=1 → truncated=True
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.passo(episodio_id, acao=0)
    assert exc_info.value.status == 409


def test_remover_episodio_nao_existe(app_cliente):
    """HTTP 404 ao deletar episódio inexistente."""
    cliente = ClienteAmbiente(app_cliente, comprimento=5, max_passos=10)
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.encerrar("episodio_falso")
    assert exc_info.value.status == 404


def test_criar_episodio_http_500(app_cliente):
    """HTTP 500 (erro interno do servidor) ao criar episódio."""
    transport = httpx.MockTransport(mock_handler(500, b"Internal Server Error"))
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.criar_episodio()
    assert exc_info.value.status == 500


def test_passo_http_500(app_cliente):
    """HTTP 500 ao dar passo."""
    cliente = ClienteAmbiente(app_cliente, comprimento=5, max_passos=10)
    episodio_id, _ = cliente.criar_episodio()
    
    def handler_passos(request: httpx.Request) -> httpx.Response:
        if "/passos" in request.url.path:
            return httpx.Response(500, content=b"Server Error")
        return httpx.Response(201, json={"id": episodio_id, "observacao": 0, "passos": 0, "encerrado": False})
    
    transport = httpx.MockTransport(handler_passos)
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente_mock = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente_mock.passo(episodio_id, acao=1)
    assert exc_info.value.status == 500


# ===== Testes de timeout e falha de conexão =====

def test_criar_episodio_timeout():
    """ReadTimeout ao criar episódio."""
    request = httpx.Request("POST", "http://ambiente/episodios")
    
    def handler(req):
        raise httpx.ReadTimeout("timeout", request=request)
    
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.criar_episodio()
    assert exc_info.value.status is None


def test_passo_timeout():
    """ReadTimeout ao dar passo."""
    request = httpx.Request("POST", "http://ambiente/episodios/abc/passos")
    
    def handler(req):
        raise httpx.ReadTimeout("timeout", request=request)
    
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.passo("abc", acao=1)
    assert exc_info.value.status is None


def test_criar_episodio_conexao_falha():
    """ConnectError ao criar episódio."""
    request = httpx.Request("POST", "http://ambiente/episodios")
    
    def handler(req):
        raise httpx.ConnectError("conexão recusada", request=request)
    
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.criar_episodio()
    assert exc_info.value.status is None


def test_passo_conexao_falha():
    """ConnectError ao dar passo."""
    request = httpx.Request("POST", "http://ambiente/episodios/abc/passos")
    
    def handler(req):
        raise httpx.ConnectError("conexão recusada", request=request)
    
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.passo("abc", acao=1)
    assert exc_info.value.status is None


# ===== Testes de violação de contrato (ErroContrato) =====

def test_criar_episodio_status_incorreto():
    """Status 200 em vez de 201 ao criar episódio."""
    transport = httpx.MockTransport(
        mock_handler(200, b'{"id": "abc", "observacao": 0, "passos": 0, "encerrado": false}')
    )
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroContrato):
        cliente.criar_episodio()


def test_passo_status_incorreto():
    """Status 201 em vez de 200 ao dar passo."""
    request = httpx.Request("POST", "http://ambiente/episodios/abc/passos")
    
    def handler(req):
        if "/passos" in req.url.path:
            return httpx.Response(201, json={"observacao": 1, "recompensa": -1.0, "terminated": False, "truncated": False})
        return httpx.Response(201, json={"id": "abc", "observacao": 0, "passos": 0, "encerrado": False})
    
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroContrato):
        cliente.passo("abc", acao=1)


def test_criar_episodio_corpo_nao_json():
    """Corpo não é JSON válido."""
    transport = httpx.MockTransport(mock_handler(201, b"not json"))
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroContrato):
        cliente.criar_episodio()


def test_criar_episodio_corpo_array_em_vez_de_objeto():
    """Corpo é array JSON em vez de objeto."""
    transport = httpx.MockTransport(mock_handler(201, b"[]"))
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroContrato):
        cliente.criar_episodio()


def test_criar_episodio_campo_id_ausente():
    """Campo 'id' ausente na resposta."""
    transport = httpx.MockTransport(
        mock_handler(201, b'{"observacao": 0, "passos": 0, "encerrado": false}')
    )
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroContrato):
        cliente.criar_episodio()


def test_criar_episodio_campo_observacao_ausente():
    """Campo 'observacao' ausente na resposta."""
    transport = httpx.MockTransport(
        mock_handler(201, b'{"id": "abc", "passos": 0, "encerrado": false}')
    )
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroContrato):
        cliente.criar_episodio()


def test_criar_episodio_observacao_tipo_invalido():
    """Campo 'observacao' é string em vez de int."""
    transport = httpx.MockTransport(
        mock_handler(201, b'{"id": "abc", "observacao": "zero", "passos": 0, "encerrado": false}')
    )
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroContrato):
        cliente.criar_episodio()


def test_passo_campo_recompensa_ausente():
    """Campo 'recompensa' ausente na resposta de passo."""
    transport = httpx.MockTransport(
        mock_handler(200, b'{"observacao": 1, "terminated": false, "truncated": false}')
    )
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroContrato):
        cliente.passo("abc", acao=1)


def test_passo_campo_terminated_ausente():
    """Campo 'terminated' ausente na resposta de passo."""
    transport = httpx.MockTransport(
        mock_handler(200, b'{"observacao": 1, "recompensa": -1.0, "truncated": false}')
    )
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroContrato):
        cliente.passo("abc", acao=1)


def test_passo_campo_terminated_int_recusado():
    """Campo 'terminated' como int (1) é recusado (deve ser bool)."""
    transport = httpx.MockTransport(
        mock_handler(200, b'{"observacao": 1, "recompensa": -1.0, "terminated": 1, "truncated": false}')
    )
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroContrato):
        cliente.passo("abc", acao=1)


def test_passo_recompensa_como_float():
    """Recompensa pode ser float (aceito pelas validações)."""
    cliente = ClienteAmbiente(criar_app(), comprimento=5, max_passos=10)
    episodio_id, obs = cliente.criar_episodio()
    # Serviço real sempre retorna float, mas o cliente deve aceitar
    transicao = cliente.passo(episodio_id, acao=1)
    assert isinstance(transicao.recompensa, float)
    assert transicao.recompensa == -1.0


# ===== Testes de regras de estado de episódio integradas =====

def test_rollout_episodio_termina_no_meio():
    """Episódio termina no passo 2 (de 5), próximo criado. next_values correto."""
    app = criar_app()
    cliente = ClienteAmbiente(TestClient(app), comprimento=3, max_passos=10)
    
    politica = lambda obs: 1  # sempre direita
    valor = lambda obs: float(obs)
    
    # Passos: 0→1 (0, -1, F, F), 1→2 (1, -1, F, F), 2→3 (2, +10, T, F) [remove]
    # novo: 0→1 (0, -1, F, F), 1→2 (1, -1, F, F) [último, truncado]
    
    rollout = coletar_rollout(cliente, politica, valor, n_passos=5, gamma=0.99, gae_lambda=0.95)
    
    # Oráculos calculados à mão
    np.testing.assert_array_equal(rollout.observacoes, [0, 1, 2, 0, 1])
    np.testing.assert_array_equal(rollout.acoes, [1, 1, 1, 1, 1])
    np.testing.assert_array_equal(rollout.recompensas, [-1.0, -1.0, 10.0, -1.0, -1.0])
    np.testing.assert_array_equal(rollout.values, [0.0, 1.0, 2.0, 0.0, 1.0])
    np.testing.assert_array_equal(rollout.next_values, [1.0, 2.0, 3.0, 1.0, 2.0])  # obs final do passo, não do próximo episódio
    np.testing.assert_array_equal(rollout.terminated, [False, False, True, False, False])
    np.testing.assert_array_equal(rollout.truncated, [False, False, False, False, True])


def test_rollout_episodio_trunca_no_meio():
    """Episódio trunca (max_passos=2) nos passos 1 e 3. next_values usa obs final."""
    app = criar_app()
    cliente = ClienteAmbiente(TestClient(app), comprimento=10, max_passos=2)
    
    politica = lambda obs: 1  # sempre direita
    valor = lambda obs: float(obs)
    
    # Passo 0: 0→1 (obs=0, rew=-1, F, F)
    # Passo 1: 1→2 (obs=1, rew=-1, F, T) [passos=2, max_passos=2] remove
    # Passo 2: 0→1 (obs=0, rew=-1, F, F)
    # Passo 3: 1→2 (obs=1, rew=-1, F, T) remove
    # Passo 4: 0→1 (obs=0, rew=-1, F, F) último, truncado=True
    
    rollout = coletar_rollout(cliente, politica, valor, n_passos=5, gamma=0.99, gae_lambda=0.95)
    
    np.testing.assert_array_equal(rollout.observacoes, [0, 1, 0, 1, 0])
    np.testing.assert_array_equal(rollout.recompensas, [-1.0, -1.0, -1.0, -1.0, -1.0])
    np.testing.assert_array_equal(rollout.values, [0.0, 1.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(rollout.next_values, [1.0, 2.0, 1.0, 2.0, 1.0])  # obs final
    np.testing.assert_array_equal(rollout.terminated, [False, False, False, False, False])
    np.testing.assert_array_equal(rollout.truncated, [False, True, False, True, True])


def test_rollout_episodio_termina_no_final():
    """Episódio termina no último passo (passo 2 de 3)."""
    app = criar_app()
    cliente = ClienteAmbiente(TestClient(app), comprimento=3, max_passos=10)
    
    politica = lambda obs: 1
    valor = lambda obs: float(obs)
    
    # Passo 0: 0→1 (0, -1, F, F)
    # Passo 1: 1→2 (1, -1, F, F)
    # Passo 2: 2→3 (2, +10, T, F) último, não truncado
    
    rollout = coletar_rollout(cliente, politica, valor, n_passos=3, gamma=0.99, gae_lambda=0.95)
    
    np.testing.assert_array_equal(rollout.observacoes, [0, 1, 2])
    np.testing.assert_array_equal(rollout.recompensas, [-1.0, -1.0, 10.0])
    np.testing.assert_array_equal(rollout.next_values, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(rollout.terminated, [False, False, True])
    np.testing.assert_array_equal(rollout.truncated, [False, False, False])


def test_rollout_ultimo_passo_marcado_truncado():
    """Último passo é marcado truncated=True mesmo que episódio não tenha terminado."""
    app = criar_app()
    cliente = ClienteAmbiente(TestClient(app), comprimento=10, max_passos=100)
    
    politica = lambda obs: 1
    valor = lambda obs: float(obs)
    
    rollout = coletar_rollout(cliente, politica, valor, n_passos=3, gamma=0.99, gae_lambda=0.95)
    
    assert rollout.truncated[-1] == True
    assert rollout.terminated[-1] == False


def test_rollout_episodios_criados_sao_removidos():
    """Todos os episódios criados pelo coletor são removidos do serviço ao final."""
    app_instance = criar_app()
    
    episodios_criados = []
    original_post = TestClient.post
    
    def post_com_registro(self, *args, **kwargs):
        resposta = original_post(self, *args, **kwargs)
        if "/episodios" in args[0] and resposta.status_code == 201:
            episodios_criados.append(resposta.json()["id"])
        return resposta
    
    TestClient.post = post_com_registro
    try:
        cliente = ClienteAmbiente(TestClient(app_instance), comprimento=3, max_passos=2)
        politica = lambda obs: 1
        valor = lambda obs: float(obs)
        coletar_rollout(cliente, politica, valor, n_passos=5, gamma=0.99, gae_lambda=0.95)
        
        # Verificar que todos os episódios foram removidos (GET retorna 404)
        http = TestClient(app_instance)
        for ep_id in episodios_criados:
            resposta = http.get(f"/episodios/{ep_id}")
            assert resposta.status_code == 404
    finally:
        TestClient.post = original_post


def test_rollout_recompensa_valor_correto_terminado():
    """Passo que termina o episódio retorna recompensa +10."""
    app = criar_app()
    cliente = ClienteAmbiente(TestClient(app), comprimento=2, max_passos=100)
    
    politica = lambda obs: 1
    valor = lambda obs: float(obs)
    
    # Dois passos: 0→1→2 (termina). Recompensas: -1, +10
    rollout = coletar_rollout(cliente, politica, valor, n_passos=2, gamma=0.99, gae_lambda=0.95)
    
    np.testing.assert_array_equal(rollout.recompensas, [-1.0, 10.0])
    np.testing.assert_array_equal(rollout.terminated, [False, True])


def test_rollout_recompensa_valor_correto_normal():
    """Passos que não terminam retornam recompensa -1."""
    app = criar_app()
    cliente = ClienteAmbiente(TestClient(app), comprimento=20, max_passos=100)
    
    politica = lambda obs: 1
    valor = lambda obs: 0.0  # valor constante
    
    # 3 passos sem terminar
    rollout = coletar_rollout(cliente, politica, valor, n_passos=3, gamma=0.99, gae_lambda=0.95)
    
    np.testing.assert_array_equal(rollout.recompensas, [-1.0, -1.0, -1.0])
    np.testing.assert_array_equal(rollout.terminated, [False, False, False])


def test_rollout_valores_array_com_funcao():
    """Array values é calculado com a função valor(observacao_atual)."""
    app = criar_app()
    cliente = ClienteAmbiente(TestClient(app), comprimento=10, max_passos=100)
    
    valor = lambda obs: obs * 2.0  # função diferente para cada observação
    politica = lambda obs: 1
    
    rollout = coletar_rollout(cliente, politica, valor, n_passos=3, gamma=0.99, gae_lambda=0.95)
    
    # Observações: 0, 1, 2 → values: 0, 2, 4
    np.testing.assert_array_equal(rollout.values, [0.0, 2.0, 4.0])


def test_rollout_next_values_com_observacao_final_apos_truncado():
    """next_values[t] usa observação final do episódio, não do episódio seguinte, após truncado."""
    app = criar_app()
    cliente = ClienteAmbiente(TestClient(app), comprimento=10, max_passos=2)
    
    valor = lambda obs: float(obs) + 0.1  # função distinta para cada obs
    politica = lambda obs: 1
    
    # max_passos=2, então episódio trunca no passo 1 com observação=2
    # next_values[1] deve ser valor(2)=2.1, não valor(0)=0.1 do próximo episódio
    rollout = coletar_rollout(cliente, politica, valor, n_passos=3, gamma=0.99, gae_lambda=0.95)
    
    # Passo 1 termina com obs=2, truncated=True
    assert rollout.truncated[1] == True
    assert rollout.next_values[1] == pytest.approx(2.1)


def test_coletar_rollout_n_passos_invalido():
    """coletar_rollout valida n_passos (deve ser int >= 1)."""
    app = criar_app()
    cliente = ClienteAmbiente(TestClient(app), comprimento=5, max_passos=10)
    politica = lambda obs: 0
    valor = lambda obs: 0.0
    
    with pytest.raises(ValueError):
        coletar_rollout(cliente, politica, valor, n_passos=0, gamma=0.99, gae_lambda=0.95)
    
    with pytest.raises(ValueError):
        coletar_rollout(cliente, politica, valor, n_passos=-1, gamma=0.99, gae_lambda=0.95)
    
    with pytest.raises(ValueError):
        coletar_rollout(cliente, politica, valor, n_passos=1.5, gamma=0.99, gae_lambda=0.95)
    
    with pytest.raises(ValueError):
        coletar_rollout(cliente, politica, valor, n_passos=True, gamma=0.99, gae_lambda=0.95)


def test_rollout_arrays_sao_numpy():
    """Arrays do Rollout são do tipo numpy, com dtypes corretos."""
    app = criar_app()
    cliente = ClienteAmbiente(TestClient(app), comprimento=5, max_passos=10)
    politica = lambda obs: 0
    valor = lambda obs: 0.0
    
    rollout = coletar_rollout(cliente, politica, valor, n_passos=2, gamma=0.99, gae_lambda=0.95)
    
    assert isinstance(rollout.observacoes, np.ndarray)
    assert isinstance(rollout.acoes, np.ndarray)
    assert isinstance(rollout.recompensas, np.ndarray)
    assert isinstance(rollout.values, np.ndarray)
    assert isinstance(rollout.next_values, np.ndarray)
    assert isinstance(rollout.terminated, np.ndarray)
    assert isinstance(rollout.truncated, np.ndarray)
    assert isinstance(rollout.advantages, np.ndarray)
    assert isinstance(rollout.returns, np.ndarray)
    
    assert rollout.observacoes.dtype == np.int64
    assert rollout.acoes.dtype == np.int64
    assert rollout.recompensas.dtype == np.float64
    assert rollout.values.dtype == np.float64
    assert rollout.next_values.dtype == np.float64
    assert rollout.terminated.dtype == bool
    assert rollout.truncated.dtype == bool
```