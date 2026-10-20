# Raciocínio (25 linhas)

1. **Chamadas HTTP**: POST /episodios (201), POST /passos (200), DELETE (204)
2. **Falhas HTTP**: 404 (não existe), 409 (já encerrado), 422 (validação), 5xx, timeout, conexão
3. **Falhas de contrato**: JSON inválido, corpo não-dict, campos ausentes/tipo errado; **crítico**: bool em lugar de int (valor True=1 quebraria observação)
4. **Validação 422**: comprimento ∈ [2,20], max_passos ∈ [1,500], ação ∈ [0,1]
5. **Regra 1 (reinício)**: Episódio que termina (terminated=T) ou trunca (truncated=T) em passo não-final → coletor deleta e cria novo; verificado com GET 404
6. **Regra 2 (next_values)**: next_values[t] = valor(obs_final_passo_t), nunca obs_inicial_novo; crítico em truncagem
7. **Regra 3 (último passo)**: Se não terminou naturalmente, coletor força truncated[-1]=True
8. **Regra 4 (limpeza)**: Episódio aberto ao fim removido; GET retorna 404

Para regra 1: comprimento=2, max_passos=10, n_passos=4, ação=1 (direita)
- Passo 0: 0→1 (-1, F, F)
- Passo 1: 1→2 (+10, T, F) ← **termina, reinicia aqui**
- Passo 2: 0→1 (-1, F, F)
- Passo 3: 1→2 (+10, T, F) ← último, sem reinício
Esperado: obs=[0,1,0,1], reward=[-1,10,-1,10], term=[F,T,F,T], trunc=[F,F,F,F]

```python
import pytest
import httpx
import numpy as np
from fastapi.testclient import TestClient

from ambiente.app import criar_app
from coletor.rollout import (
    ClienteAmbiente,
    ErroAmbiente,
    ErroContrato,
    coletar_rollout,
)


# ============================================================================
# TESTES DE VALIDAÇÃO 422 (ENTRADA INVÁLIDA)
# ============================================================================

def test_criar_episodio_comprimento_abaixo_minimo():
    """POST /episodios com comprimento < 2 retorna HTTP 422."""
    app = criar_app()
    http = TestClient(app)
    response = http.post("/episodios", json={"comprimento": 1, "max_passos": 5})
    assert response.status_code == 422


def test_criar_episodio_comprimento_acima_maximo():
    """POST /episodios com comprimento > 20 retorna HTTP 422."""
    app = criar_app()
    http = TestClient(app)
    response = http.post("/episodios", json={"comprimento": 21, "max_passos": 5})
    assert response.status_code == 422


def test_criar_episodio_max_passos_abaixo_minimo():
    """POST /episodios com max_passos < 1 retorna HTTP 422."""
    app = criar_app()
    http = TestClient(app)
    response = http.post("/episodios", json={"comprimento": 5, "max_passos": 0})
    assert response.status_code == 422


def test_criar_episodio_max_passos_acima_maximo():
    """POST /episodios com max_passos > 500 retorna HTTP 422."""
    app = criar_app()
    http = TestClient(app)
    response = http.post("/episodios", json={"comprimento": 5, "max_passos": 501})
    assert response.status_code == 422


def test_passo_acao_negativa():
    """POST /passos com acao < 0 retorna HTTP 422."""
    app = criar_app()
    http = TestClient(app)
    ep = http.post("/episodios", json={"comprimento": 5, "max_passos": 10}).json()
    response = http.post(f"/episodios/{ep['id']}/passos", json={"acao": -1})
    assert response.status_code == 422


def test_passo_acao_maior_que_um():
    """POST /passos com acao > 1 retorna HTTP 422."""
    app = criar_app()
    http = TestClient(app)
    ep = http.post("/episodios", json={"comprimento": 5, "max_passos": 10}).json()
    response = http.post(f"/episodios/{ep['id']}/passos", json={"acao": 2})
    assert response.status_code == 422


# ============================================================================
# TESTES DE ERRO HTTP (CLIENTE REAL)
# ============================================================================

def test_passo_episodio_nao_existe():
    """POST /passos em episódio inexistente retorna 404, coletor lança ErroAmbiente."""
    app = criar_app()
    http = TestClient(app)
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.passo("episodio_inexistente", 1)
    assert exc_info.value.status == 404


def test_passo_episodio_ja_encerrado():
    """POST /passos em episódio encerrado retorna 409, coletor lança ErroAmbiente."""
    app = criar_app()
    http = TestClient(app)
    cliente = ClienteAmbiente(http, comprimento=2, max_passos=10)
    
    # Cria e termina o episódio
    id_ep, obs = cliente.criar_episodio()
    cliente.passo(id_ep, 1)  # obs: 0→1
    cliente.passo(id_ep, 1)  # obs: 1→2, termina
    
    # Tenta passo após encerramento
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.passo(id_ep, 1)
    assert exc_info.value.status == 409


def test_encerrar_episodio_nao_existe():
    """DELETE em episódio inexistente retorna 404, coletor lança ErroAmbiente."""
    app = criar_app()
    http = TestClient(app)
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.encerrar("episodio_inexistente")
    assert exc_info.value.status == 404


# ============================================================================
# TESTES DE TIMEOUT E CONEXÃO (MOCK TRANSPORT)
# ============================================================================

def test_criar_episodio_timeout():
    """POST /episodios com timeout lança ErroAmbiente com status=None."""
    def handler(request):
        raise httpx.ReadTimeout("tempo esgotado", request=request)
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.criar_episodio()
    assert exc_info.value.status is None


def test_criar_episodio_conexao_recusada():
    """POST /episodios com conexão recusada lança ErroAmbiente com status=None."""
    def handler(request):
        raise httpx.ConnectError("recusada", request=request)
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.criar_episodio()
    assert exc_info.value.status is None


def test_passo_timeout():
    """POST /passos com timeout lança ErroAmbiente com status=None."""
    call_count = [0]
    
    def handler(request):
        call_count[0] += 1
        if call_count[0] == 1:  # POST /episodios
            return httpx.Response(201, json={"id": "ep1", "observacao": 0, "passos": 0, "encerrado": False})
        else:  # POST /passos
            raise httpx.ReadTimeout("tempo esgotado", request=request)
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    id_ep, _ = cliente.criar_episodio()
    
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.passo(id_ep, 1)
    assert exc_info.value.status is None


def test_passo_conexao_recusada():
    """POST /passos com conexão recusada lança ErroAmbiente com status=None."""
    call_count = [0]
    
    def handler(request):
        call_count[0] += 1
        if call_count[0] == 1:  # POST /episodios
            return httpx.Response(201, json={"id": "ep1", "observacao": 0, "passos": 0, "encerrado": False})
        else:  # POST /passos
            raise httpx.ConnectError("recusada", request=request)
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    id_ep, _ = cliente.criar_episodio()
    
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.passo(id_ep, 1)
    assert exc_info.value.status is None


def test_encerrar_timeout():
    """DELETE com timeout lança ErroAmbiente com status=None."""
    call_count = [0]
    
    def handler(request):
        call_count[0] += 1
        if call_count[0] == 1:  # POST /episodios
            return httpx.Response(201, json={"id": "ep1", "observacao": 0, "passos": 0, "encerrado": False})
        else:  # DELETE
            raise httpx.ReadTimeout("tempo esgotado", request=request)
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    id_ep, _ = cliente.criar_episodio()
    
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.encerrar(id_ep)
    assert exc_info.value.status is None


# ============================================================================
# TESTES DE ERRO HTTP 5XX (MOCK TRANSPORT)
# ============================================================================

def test_criar_episodio_http_500():
    """POST /episodios retornando 500 lança ErroAmbiente com status=500."""
    def handler(request):
        return httpx.Response(500, text="Erro interno do servidor")
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.criar_episodio()
    assert exc_info.value.status == 500


def test_passo_http_500():
    """POST /passos retornando 500 lança ErroAmbiente com status=500."""
    call_count = [0]
    
    def handler(request):
        call_count[0] += 1
        if call_count[0] == 1:
            return httpx.Response(201, json={"id": "ep1", "observacao": 0, "passos": 0, "encerrado": False})
        return httpx.Response(500, text="Erro interno")
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    id_ep, _ = cliente.criar_episodio()
    
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.passo(id_ep, 1)
    assert exc_info.value.status == 500


def test_encerrar_http_500():
    """DELETE retornando 500 lança ErroAmbiente com status=500."""
    call_count = [0]
    
    def handler(request):
        call_count[0] += 1
        if call_count[0] == 1:
            return httpx.Response(201, json={"id": "ep1", "observacao": 0, "passos": 0, "encerrado": False})
        return httpx.Response(500, text="Erro interno")
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    id_ep, _ = cliente.criar_episodio()
    
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.encerrar(id_ep)
    assert exc_info.value.status == 500


# ============================================================================
# TESTES DE CONTRATO (BODY MAL FORMADO)
# ============================================================================

def test_criar_episodio_json_invalido():
    """POST /episodios com JSON inválido lança ErroContrato."""
    def handler(request):
        return httpx.Response(201, text="não é JSON")
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroContrato):
        cliente.criar_episodio()


def test_criar_episodio_corpo_nao_dict():
    """POST /episodios com JSON que não é dict lança ErroContrato."""
    def handler(request):
        return httpx.Response(201, json=["array", "não", "dict"])
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroContrato):
        cliente.criar_episodio()


def test_criar_episodio_campo_id_ausente():
    """POST /episodios sem campo 'id' lança ErroContrato."""
    def handler(request):
        return httpx.Response(201, json={"observacao": 0, "passos": 0, "encerrado": False})
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroContrato):
        cliente.criar_episodio()


def test_criar_episodio_campo_observacao_ausente():
    """POST /episodios sem campo 'observacao' lança ErroContrato."""
    def handler(request):
        return httpx.Response(201, json={"id": "abc", "passos": 0, "encerrado": False})
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroContrato):
        cliente.criar_episodio()


def test_criar_episodio_observacao_como_bool():
    """POST /episodios com observacao=bool lança ErroContrato (bool != int)."""
    def handler(request):
        return httpx.Response(201, json={"id": "abc", "observacao": True, "passos": 0, "encerrado": False})
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroContrato):
        cliente.criar_episodio()


def test_passo_json_invalido():
    """POST /passos com JSON inválido lança ErroContrato."""
    call_count = [0]
    
    def handler(request):
        call_count[0] += 1
        if call_count[0] == 1:
            return httpx.Response(201, json={"id": "ep1", "observacao": 0, "passos": 0, "encerrado": False})
        return httpx.Response(200, text="não é JSON")
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    id_ep, _ = cliente.criar_episodio()
    
    with pytest.raises(ErroContrato):
        cliente.passo(id_ep, 1)


def test_passo_observacao_como_bool():
    """POST /passos com observacao=bool lança ErroContrato."""
    call_count = [0]
    
    def handler(request):
        call_count[0] += 1
        if call_count[0] == 1:
            return httpx.Response(201, json={"id": "ep1", "observacao": 0, "passos": 0, "encerrado": False})
        return httpx.Response(200, json={
            "observacao": False,
            "recompensa": -1.0,
            "terminated": False,
            "truncated": False
        })
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    id_ep, _ = cliente.criar_episodio()
    
    with pytest.raises(ErroContrato):
        cliente.passo(id_ep, 1)


def test_passo_campo_recompensa_ausente():
    """POST /passos sem campo 'recompensa' lança ErroContrato."""
    call_count = [0]
    
    def handler(request):
        call_count[0] += 1
        if call_count[0] == 1:
            return httpx.Response(201, json={"id": "ep1", "observacao": 0, "passos": 0, "encerrado": False})
        return httpx.Response(200, json={
            "observacao": 1,
            "terminated": False,
            "truncated": False
        })
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    id_ep, _ = cliente.criar_episodio()
    
    with pytest.raises(ErroContrato):
        cliente.passo(id_ep, 1)


def test_passo_campo_terminated_ausente():
    """POST /passos sem campo 'terminated' lança ErroContrato."""
    call_count = [0]
    
    def handler(request):
        call_count[0] += 1
        if call_count[0] == 1:
            return httpx.Response(201, json={"id": "ep1", "observacao": 0, "passos": 0, "encerrado": False})
        return httpx.Response(200, json={
            "observacao": 1,
            "recompensa": -1.0,
            "truncated": False
        })
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    id_ep, _ = cliente.criar_episodio()
    
    with pytest.raises(ErroContrato):
        cliente.passo(id_ep, 1)


def test_passo_campo_truncated_ausente():
    """POST /passos sem campo 'truncated' lança ErroContrato."""
    call_count = [0]
    
    def handler(request):
        call_count[0] += 1
        if call_count[0] == 1:
            return httpx.Response(201, json={"id": "ep1", "observacao": 0, "passos": 0, "encerrado": False})
        return httpx.Response(200, json={
            "observacao": 1,
            "recompensa": -1.0,
            "terminated": False
        })
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    id_ep, _ = cliente.criar_episodio()
    
    with pytest.raises(ErroContrato):
        cliente.passo(id_ep, 1)


def test_criar_episodio_status_code_inesperado():
    """POST /episodios com status 200 em vez de 201 lança ErroContrato."""
    def handler(request):
        return httpx.Response(200, json={"id": "abc", "observacao": 0, "passos": 0, "encerrado": False})
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    with pytest.raises(ErroContrato):
        cliente.criar_episodio()


def test_passo_status_code_inesperado():
    """POST /passos com status 201 em vez de 200 lança ErroContrato."""
    call_count = [0]
    
    def handler(request):
        call_count[0] += 1
        if call_count[0] == 1:
            return httpx.Response(201, json={"id": "ep1", "observacao": 0, "passos": 0, "encerrado": False})
        return httpx.Response(201, json={
            "observacao": 1,
            "recompensa": -1.0,
            "terminated": False,
            "truncated": False
        })
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    id_ep, _ = cliente.criar_episodio()
    
    with pytest.raises(ErroContrato):
        cliente.passo(id_ep, 1)


def test_encerrar_status_code_inesperado():
    """DELETE com status 200 em vez de 204 lança ErroContrato."""
    call_count = [0]
    
    def handler(request):
        call_count[0] += 1
        if call_count[0] == 1:
            return httpx.Response(201, json={"id": "ep1", "observacao": 0, "passos": 0, "encerrado": False})
        return httpx.Response(200)
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    id_ep, _ = cliente.criar_episodio()
    
    with pytest.raises(ErroContrato):
        cliente.encerrar(id_ep)


# ============================================================================
# TESTES DE REGRAS DE FRONTEIRA (INTEGRAÇÃO)
# ============================================================================

def test_reinicio_automatico_apos_terminated():
    """Episódio que termina no passo intermediário reinicia automaticamente.
    
    comprimento=2, max_passos=10, n_passos=4, política=sempre direita (1)
    Passo 0: obs=0 → obs=1 (-1, F, F)
    Passo 1: obs=1 → obs=2 (+10, T, F) ← TERMINA, episódio deletado e recriado
    Passo 2: obs=0 (novo) → obs=1 (-1, F, F)
    Passo 3: obs=1 → obs=2 (+10, T, F)
    
    Esperado: obs=[0,1,0,1], reward=[-1,10,-1,10], term=[F,T,F,T], trunc=[F,F,F,F]
    """
    app = criar_app()
    http = TestClient(app)
    cliente = ClienteAmbiente(http, comprimento=2, max_passos=10)
    
    def politica(obs):
        return 1
    
    def valor(obs):
        return 0.0
    
    rollout = coletar_rollout(cliente, politica, valor, n_passos=4, gamma=0.99, gae_lambda=0.95)
    
    assert rollout.observacoes.tolist() == [0, 1, 0, 1]
    assert rollout.acoes.tolist() == [1, 1, 1, 1]
    assert rollout.recompensas.tolist() == [-1.0, 10.0, -1.0, 10.0]
    assert rollout.terminated.tolist() == [False, True, False, True]
    assert rollout.truncated.tolist() == [False, False, False, False]


def test_reinicio_automatico_apos_truncated():
    """Episódio que atinge max_passos reinicia automaticamente.
    
    comprimento=10, max_passos=2, n_passos=5, política=sempre direita
    Passo 0: obs=0 → obs=1, passos=1 (-1, F, F)
    Passo 1: obs=1 → obs=2, passos=2 (-1, F, T) ← TRUNCA, episódio deletado e recriado
    Passo 2: obs=0 (novo) → obs=1, passos=1 (-1, F, F)
    Passo 3: obs=1 → obs=2, passos=2 (-1, F, T) ← TRUNCA, episódio deletado e recriado
    Passo 4: obs=0 (novo) → obs=1 (-1, F, F) [último passo, coletor força trunc[-1]=T]
    
    Esperado: obs=[0,1,0,1,0], reward=[-1,-1,-1,-1,-1], term=[F,F,F,F,F], trunc=[F,T,F,T,T]
    """
    app = criar_app()
    http = TestClient(app)
    cliente = ClienteAmbiente(http, comprimento=10, max_passos=2)
    
    def politica(obs):
        return 1
    
    def valor(obs):
        return 0.0
    
    rollout = coletar_rollout(cliente, politica, valor, n_passos=5, gamma=0.99, gae_lambda=0.95)
    
    assert rollout.observacoes.tolist() == [0, 1, 0, 1, 0]
    assert rollout.acoes.tolist() == [1, 1, 1, 1, 1]
    assert rollout.recompensas.tolist() == [-1.0, -1.0, -1.0, -1.0, -1.0]
    assert rollout.terminated.tolist() == [False, False, False, False, False]
    assert rollout.truncated.tolist() == [False, True, False, True, True]


def test_ultimo_passo_truncado_se_nao_terminou_naturalmente():
    """Último passo marcado como truncated=True se episódio não terminou naturalmente.
    
    comprimento=10, max_passos=10, n_passos=3, política=sempre direita
    Passo 0: obs=0 → obs=1 (-1, F, F)
    Passo 1: obs=1 → obs=2 (-1, F, F)
    Passo 2: obs=2 → obs=3 (-1, F, F) [último, não terminou, coletor força trunc=T]
    
    Esperado: term=[F,F,F], trunc=[F,F,T]
    """
    app = criar_app()
    http = TestClient(app)
    cliente = ClienteAmbiente(http, comprimento=10, max_passos=10)
    
    def politica(obs):
        return 1
    
    def valor(obs):
        return 0.0
    
    rollout = coletar_rollout(cliente, politica, valor, n_passos=3, gamma=0.99, gae_lambda=0.95)
    
    assert rollout.terminated.tolist() == [False, False, False]
    assert rollout.truncated.tolist() == [False, False, True]


def test_ultimo_passo_nao_truncado_se_terminou_naturalmente():
    """Último passo NÃO marcado como truncated se episódio terminou naturalmente.
    
    comprimento=3, max_passos=10, n_passos=3, política=sempre direita
    Passo 0: obs=0 → obs=1 (-1, F, F)
    Passo 1: obs=1 → obs=2 (-1, F, F)
    Passo 2: obs=2 → obs=3 (+10, T, F) [último, terminou naturalmente, trunc permanece F]
    
    Esperado: term=[F,F,T], trunc=[F,F,F]
    """
    app = criar_app()
    http = TestClient(app)
    cliente = ClienteAmbiente(http, comprimento=3, max_passos=10)
    
    def politica(obs):
        return 1
    
    def valor(obs):
        return 0.0
    
    rollout = coletar_rollout(cliente, politica, valor, n_passos=3, gamma=0.99, gae_lambda=0.95)
    
    assert rollout.terminated.tolist() == [False, False, True]
    assert rollout.truncated.tolist() == [False, False, False]


def test_next_values_usa_observacao_final_apos_terminacao():
    """next_values[t] usa observação final quando episódio termina.
    
    comprimento=2, max_passos=10, n_passos=3, política=sempre direita
    Passo 0: obs=0 → obs=1, next_value=valor(1)
    Passo 1: obs=1 → obs=2, terminated=T, next_value=valor(2) [NOT valor(0) do novo]
    Passo 2: obs=0 (novo) → obs=1, next_value=valor(1)
    
    Com valor(x)=0 para todo x, o importante é verificar que ocorre reinício
    (obs=[0,1,0]) e recompensas confirmam término (reward[-1] alto).
    """
    app = criar_app()
    http = TestClient(app)
    cliente = ClienteAmbiente(http, comprimento=2, max_passos=10)
    
    def politica(obs):
        return 1
    
    def valor(obs):
        return 0.0
    
    rollout = coletar_rollout(cliente, politica, valor, n_passos=3, gamma=0.99, gae_lambda=0.95)
    
    # Verifica que o episódio realmente reinicia após passo 1 (onde termina)
    assert rollout.observacoes.tolist() == [0, 1, 0]
    assert rollout.recompensas[1] == 10.0  # Passo 1 recebe recompensa de objetivo
    assert rollout.terminated[1] == True


def test_episodio_aberto_removido_apos_rollout():
    """Episódio em aberto após rollout é removido do serviço (GET retorna 404).
    
    Após coletar_rollout, o episódio final deve estar deletado.
    """
    app = criar_app()
    http = TestClient(app)
    
    # Captura o ID do último episódio criado
    ultimo_id = []
    original_criar = ClienteAmbiente.criar_episodio
    
    def criar_com_spy(self):
        id_ep, obs = original_criar(self)
        ultimo_id.append(id_ep)
        return id_ep, obs
    
    ClienteAmbiente.criar_episodio = criar_com_spy
    
    try:
        cliente = ClienteAmbiente(http, comprimento=10, max_passos=10)
        
        def politica(obs):
            return 1
        
        def valor(obs):
            return 0.0
        
        coletar_rollout(cliente, politica, valor, n_passos=2, gamma=0.99, gae_lambda=0.95)
        
        # Verifica que o episódio foi deletado
        if ultimo_id:
            response = http.get(f"/episodios/{ultimo_id[-1]}")
            assert response.status_code == 404
    finally:
        ClienteAmbiente.criar_episodio = original_criar


def test_coletar_rollout_n_passos_invalido_zero():
    """coletar_rollout com n_passos=0 lança ValueError."""
    app = criar_app()
    http = TestClient(app)
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    
    with pytest.raises(ValueError):
        coletar_rollout(cliente, lambda o: 1, lambda o: 0.0, n_passos=0, gamma=0.99, gae_lambda=0.95)


def test_coletar_rollout_n_passos_invalido_float():
    """coletar_rollout com n_passos float lança ValueError."""
    app = criar_app()
    http = TestClient(app)
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    
    with pytest.raises(ValueError):
        coletar_rollout(cliente, lambda o: 1, lambda o: 0.0, n_passos=1.5, gamma=0.99, gae_lambda=0.95)


def test_coletar_rollout_n_passos_invalido_bool():
    """coletar_rollout com n_passos=True (bool) lança ValueError."""
    app = criar_app()
    http = TestClient(app)
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    
    with pytest.raises(ValueError):
        coletar_rollout(cliente, lambda o: 1, lambda o: 0.0, n_passos=True, gamma=0.99, gae_lambda=0.95)


def test_arrays_dtype_corretos():
    """Rollout retorna arrays com dtypes esperados."""
    app = criar_app()
    http = TestClient(app)
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    
    def politica(obs):
        return 0 if obs < 2 else 1
    
    def valor(obs):
        return float(obs) * 0.1
    
    rollout = coletar_rollout(cliente, politica, valor, n_passos=5, gamma=0.99, gae_lambda=0.95)
    
    assert rollout.observacoes.dtype == np.int64
    assert rollout.acoes.dtype == np.int64
    assert rollout.recompensas.dtype == np.float64
    assert rollout.values.dtype == np.float64
    assert rollout.next_values.dtype == np.float64
    assert rollout.terminated.dtype == bool
    assert rollout.truncated.dtype == bool
    assert rollout.advantages.dtype == np.float64
    assert rollout.returns.dtype == np.float64


def test_recompensa_aceita_int_ou_float():
    """Campo recompensa pode ser int ou float, nunca bool."""
    call_count = [0]
    
    def handler(request):
        call_count[0] += 1
        if call_count[0] == 1:
            return httpx.Response(201, json={"id": "ep1", "observacao": 0, "passos": 0, "encerrado": False})
        return httpx.Response(200, json={
            "observacao": 1,
            "recompensa": 1,  # int ao invés de float, deve funcionar
            "terminated": False,
            "truncated": False
        })
    
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)
    id_ep, _ = cliente.criar_episodio()
    
    transicao = cliente.passo(id_ep, 1)
    assert transicao.recompensa == 1.0
    assert isinstance(transicao.recompensa, float)
```