import pytest
import httpx
import numpy as np
from fastapi.testclient import TestClient

from ambiente.app import criar_app
from coletor.rollout import (
    ClienteAmbiente, ErroAmbiente, ErroContrato, Transicao, Rollout, coletar_rollout
)


@pytest.fixture
def cliente_real():
    """Serviço real em memória, novo a cada teste."""
    app = criar_app()
    http = TestClient(app)
    return ClienteAmbiente(http, comprimento=5, max_passos=10)


@pytest.fixture
def cliente_real_curto():
    """Serviço real com episódios curtos."""
    app = criar_app()
    http = TestClient(app)
    return ClienteAmbiente(http, comprimento=3, max_passos=5)


# ============================================================================
# Sucessos básicos com serviço real
# ============================================================================

def test_criar_episodio_sucesso_basico(cliente_real):
    """Criar episódio retorna id válido e observação inicial 0."""
    episodio_id, observacao = cliente_real.criar_episodio()
    assert isinstance(episodio_id, str) and len(episodio_id) == 32
    assert observacao == 0


def test_passo_sucesso_recompensa_passo(cliente_real):
    """Passo normal retorna recompensa -1."""
    episodio_id, _ = cliente_real.criar_episodio()
    transicao = cliente_real.passo(episodio_id, 1)
    assert transicao.observacao == 1
    assert transicao.recompensa == -1.0
    assert not transicao.terminated and not transicao.truncated


def test_remover_episodio_sucesso(cliente_real):
    """Remover episódio impede passo posterior."""
    episodio_id, _ = cliente_real.criar_episodio()
    cliente_real.encerrar(episodio_id)
    with pytest.raises(ErroAmbiente) as exc:
        cliente_real.passo(episodio_id, 1)
    assert exc.value.status == 404


# ============================================================================
# Erros HTTP 404 com serviço real
# ============================================================================

def test_passo_episodio_inexistente_retorna_404(cliente_real):
    """Passo em episódio inexistente lança ErroAmbiente com status 404."""
    with pytest.raises(ErroAmbiente) as exc:
        cliente_real.passo("nao_existe", 1)
    assert exc.value.status == 404


def test_remover_episodio_inexistente_retorna_404(cliente_real):
    """Remover episódio inexistente lança ErroAmbiente com status 404."""
    with pytest.raises(ErroAmbiente) as exc:
        cliente_real.encerrar("nao_existe")
    assert exc.value.status == 404


# ============================================================================
# Erro HTTP 409 com serviço real
# ============================================================================

def test_passo_episodio_terminated_retorna_409(cliente_real):
    """Passo em episódio terminado lança ErroAmbiente com status 409."""
    episodio_id, _ = cliente_real.criar_episodio()
    for _ in range(5):
        cliente_real.passo(episodio_id, 1)
    with pytest.raises(ErroAmbiente) as exc:
        cliente_real.passo(episodio_id, 1)
    assert exc.value.status == 409


def test_passo_episodio_truncado_retorna_409(cliente_real_curto):
    """Passo em episódio truncado lança ErroAmbiente com status 409."""
    episodio_id, _ = cliente_real_curto.criar_episodio()
    for _ in range(5):
        cliente_real_curto.passo(episodio_id, 0)
    with pytest.raises(ErroAmbiente) as exc:
        cliente_real_curto.passo(episodio_id, 1)
    assert exc.value.status == 409


# ============================================================================
# Timeout com MockTransport
# ============================================================================

def test_criar_episodio_timeout_retorna_status_none():
    """Timeout ao criar episódio lança ErroAmbiente com status None."""
    def handler(request):
        raise httpx.TimeoutException("timeout")
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 5, 10)
    
    with pytest.raises(ErroAmbiente) as exc:
        cliente.criar_episodio()
    assert exc.value.status is None


def test_passo_timeout_retorna_status_none():
    """Timeout ao dar passo lança ErroAmbiente com status None."""
    def handler(request):
        raise httpx.TimeoutException("timeout")
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 5, 10)
    
    with pytest.raises(ErroAmbiente) as exc:
        cliente.passo("id", 1)
    assert exc.value.status is None


# ============================================================================
# Erro de conexão com MockTransport
# ============================================================================

def test_remover_episodio_erro_conexao_retorna_status_none():
    """Erro de conexão ao remover lança ErroAmbiente com status None."""
    def handler(request):
        raise httpx.ConnectError("conexão falhou")
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 5, 10)
    
    with pytest.raises(ErroAmbiente) as exc:
        cliente.encerrar("id")
    assert exc.value.status is None


# ============================================================================
# HTTP 5xx com MockTransport
# ============================================================================

def test_criar_episodio_http_500_retorna_status_500():
    """HTTP 500 ao criar lança ErroAmbiente com status 500."""
    def handler(request):
        return httpx.Response(500, text="erro")
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 5, 10)
    
    with pytest.raises(ErroAmbiente) as exc:
        cliente.criar_episodio()
    assert exc.value.status == 500


def test_passo_http_503_retorna_status_503():
    """HTTP 503 ao dar passo lança ErroAmbiente com status 503."""
    def handler(request):
        return httpx.Response(503, text="indisponível")
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 5, 10)
    
    with pytest.raises(ErroAmbiente) as exc:
        cliente.passo("id", 1)
    assert exc.value.status == 503


# ============================================================================
# Corpo não-JSON com MockTransport
# ============================================================================

def test_criar_episodio_corpo_nao_json_lanca_erro_contrato():
    """Corpo não-JSON ao criar lança ErroContrato."""
    def handler(request):
        return httpx.Response(201, content=b"not json")
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 5, 10)
    
    with pytest.raises(ErroContrato):
        cliente.criar_episodio()


def test_passo_corpo_nao_json_lanca_erro_contrato():
    """Corpo não-JSON ao dar passo lança ErroContrato."""
    def handler(request):
        return httpx.Response(200, content=b"not json")
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 5, 10)
    
    with pytest.raises(ErroContrato):
        cliente.passo("id", 1)


# ============================================================================
# Corpo não-objeto JSON com MockTransport
# ============================================================================

def test_criar_episodio_corpo_array_json_lanca_erro_contrato():
    """Corpo é array em vez de objeto ao criar lança ErroContrato."""
    def handler(request):
        return httpx.Response(201, json=[1, 2, 3])
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 5, 10)
    
    with pytest.raises(ErroContrato) as exc:
        cliente.criar_episodio()
    assert "objeto JSON" in str(exc.value)


def test_passo_corpo_string_json_lanca_erro_contrato():
    """Corpo é string em vez de objeto ao dar passo lança ErroContrato."""
    def handler(request):
        return httpx.Response(200, json="string")
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 5, 10)
    
    with pytest.raises(ErroContrato) as exc:
        cliente.passo("id", 1)
    assert "objeto JSON" in str(exc.value)


# ============================================================================
# Campo ausente com MockTransport
# ============================================================================

def test_criar_episodio_campo_id_ausente_lanca_erro_contrato():
    """Campo 'id' ausente ao criar lança ErroContrato."""
    def handler(request):
        return httpx.Response(201, json={"observacao": 0, "passos": 0, "encerrado": False})
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 5, 10)
    
    with pytest.raises(ErroContrato) as exc:
        cliente.criar_episodio()
    assert "id" in str(exc.value) and "ausente" in str(exc.value)


def test_passo_campo_observacao_ausente_lanca_erro_contrato():
    """Campo 'observacao' ausente ao dar passo lança ErroContrato."""
    def handler(request):
        return httpx.Response(200, json={"recompensa": -1.0, "terminated": False, "truncated": False})
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 5, 10)
    
    with pytest.raises(ErroContrato) as exc:
        cliente.passo("id", 1)
    assert "observacao" in str(exc.value)


def test_passo_campo_recompensa_ausente_lanca_erro_contrato():
    """Campo 'recompensa' ausente ao dar passo lança ErroContrato."""
    def handler(request):
        return httpx.Response(200, json={"observacao": 1, "terminated": False, "truncated": False})
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 5, 10)
    
    with pytest.raises(ErroContrato) as exc:
        cliente.passo("id", 1)
    assert "recompensa" in str(exc.value)


def test_passo_campo_terminated_ausente_lanca_erro_contrato():
    """Campo 'terminated' ausente ao dar passo lança ErroContrato."""
    def handler(request):
        return httpx.Response(200, json={"observacao": 1, "recompensa": -1.0, "truncated": False})
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 5, 10)
    
    with pytest.raises(ErroContrato) as exc:
        cliente.passo("id", 1)
    assert "terminated" in str(exc.value)


def test_passo_campo_truncated_ausente_lanca_erro_contrato():
    """Campo 'truncated' ausente ao dar passo lança ErroContrato."""
    def handler(request):
        return httpx.Response(200, json={"observacao": 1, "recompensa": -1.0, "terminated": False})
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 5, 10)
    
    with pytest.raises(ErroContrato) as exc:
        cliente.passo("id", 1)
    assert "truncated" in str(exc.value)


# ============================================================================
# Tipo de campo errado com MockTransport
# ============================================================================

def test_criar_episodio_campo_id_tipo_int_lanca_erro_contrato():
    """Campo 'id' com tipo int em vez de str ao criar lança ErroContrato."""
    def handler(request):
        return httpx.Response(201, json={"id": 12345, "observacao": 0, "passos": 0, "encerrado": False})
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 5, 10)
    
    with pytest.raises(ErroContrato) as exc:
        cliente.criar_episodio()
    assert "id" in str(exc.value) and "tipo inválido" in str(exc.value)


def test_criar_episodio_campo_observacao_tipo_bool_lanca_erro_contrato():
    """Campo 'observacao' com tipo bool ao criar lança ErroContrato."""
    def handler(request):
        return httpx.Response(201, json={"id": "abc", "observacao": True, "passos": 0, "encerrado": False})
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 5, 10)
    
    with pytest.raises(ErroContrato) as exc:
        cliente.criar_episodio()
    assert "observacao" in str(exc.value) and "tipo inválido" in str(exc.value)


def test_passo_campo_observacao_tipo_bool_lanca_erro_contrato():
    """Campo 'observacao' com tipo bool ao dar passo lança ErroContrato."""
    def handler(request):
        return httpx.Response(200, json={"observacao": False, "recompensa": -1.0, "terminated": False, "truncated": False})
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 5, 10)
    
    with pytest.raises(ErroContrato) as exc:
        cliente.passo("id", 1)
    assert "observacao" in str(exc.value)


def test_passo_campo_recompensa_tipo_bool_lanca_erro_contrato():
    """Campo 'recompensa' com tipo bool ao dar passo lança ErroContrato."""
    def handler(request):
        return httpx.Response(200, json={"observacao": 1, "recompensa": True, "terminated": False, "truncated": False})
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 5, 10)
    
    with pytest.raises(ErroContrato) as exc:
        cliente.passo("id", 1)
    assert "recompensa" in str(exc.value)


def test_passo_campo_terminated_tipo_string_lanca_erro_contrato():
    """Campo 'terminated' com tipo string ao dar passo lança ErroContrato."""
    def handler(request):
        return httpx.Response(200, json={"observacao": 1, "recompensa": -1.0, "terminated": "true", "truncated": False})
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 5, 10)
    
    with pytest.raises(ErroContrato) as exc:
        cliente.passo("id", 1)
    assert "terminated" in str(exc.value)


# ============================================================================
# Status HTTP inesperado com MockTransport
# ============================================================================

def test_criar_episodio_status_200_em_vez_201_lanca_erro_contrato():
    """HTTP 200 em vez de 201 ao criar lança ErroContrato."""
    def handler(request):
        return httpx.Response(200, json={"id": "abc", "observacao": 0, "passos": 0, "encerrado": False})
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 5, 10)
    
    with pytest.raises(ErroContrato) as exc:
        cliente.criar_episodio()
    assert "esperado HTTP 201" in str(exc.value) and "200" in str(exc.value)


def test_passo_status_201_em_vez_200_lanca_erro_contrato():
    """HTTP 201 em vez de 200 ao dar passo lança ErroContrato."""
    def handler(request):
        return httpx.Response(201, json={"observacao": 1, "recompensa": -1.0, "terminated": False, "truncated": False})
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 5, 10)
    
    with pytest.raises(ErroContrato) as exc:
        cliente.passo("id", 1)
    assert "esperado HTTP 200" in str(exc.value) and "201" in str(exc.value)


def test_remover_episodio_status_200_em_vez_204_lanca_erro_contrato():
    """HTTP 200 em vez de 204 ao remover lança ErroContrato."""
    def handler(request):
        return httpx.Response(200, json={"ok": True})
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://a")
    cliente = ClienteAmbiente(http, 5, 10)
    
    with pytest.raises(ErroContrato) as exc:
        cliente.encerrar("id")
    assert "esperado HTTP 204" in str(exc.value) and "200" in str(exc.value)


# ============================================================================
# Regras de integração: recompensas e estados
# ============================================================================

def test_passo_recompensa_objetivo_ao_atingir_comprimento(cliente_real):
    """Passo que atinge o objetivo (comprimento) retorna recompensa +10."""
    episodio_id, _ = cliente_real.criar_episodio()
    for i in range(5):
        transicao = cliente_real.passo(episodio_id, 1)
    assert transicao.recompensa == 10.0
    assert transicao.terminated


def test_passo_posicao_minima_zero(cliente_real):
    """Posição nunca fica abaixo de 0 ao mover esquerda."""
    episodio_id, _ = cliente_real.criar_episodio()
    cliente_real.passo(episodio_id, 0)
    cliente_real.passo(episodio_id, 0)
    transicao = cliente_real.passo(episodio_id, 0)
    assert transicao.observacao == 0


def test_passo_terminated_marca_encerrado():
    """Passo com terminated=True marca episódio como encerrado."""
    app = criar_app()
    http = TestClient(app)
    cliente = ClienteAmbiente(http, comprimento=2, max_passos=10)
    episodio_id, _ = cliente.criar_episodio()
    for _ in range(2):
        cliente.passo(episodio_id, 1)
    # Próximo passo deve falhar
    with pytest.raises(ErroAmbiente) as exc:
        cliente.passo(episodio_id, 1)
    assert exc.value.status == 409


def test_passo_truncated_marca_encerrado():
    """Passo com truncated=True marca episódio como encerrado."""
    app = criar_app()
    http = TestClient(app)
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=3)
    episodio_id, _ = cliente.criar_episodio()
    for _ in range(3):
        cliente.passo(episodio_id, 0)
    # Próximo passo deve falhar
    with pytest.raises(ErroAmbiente) as exc:
        cliente.passo(episodio_id, 1)
    assert exc.value.status == 409


# ============================================================================
# Regras de integração: rollout
# ============================================================================

def test_rollout_coleta_n_passos(cliente_real):
    """Rollout coleta exatamente n_passos transições."""
    def politica(obs):
        return 1
    def valor(obs):
        return 0.0
    
    rollout = coletar_rollout(cliente_real, politica, valor, 3, gamma=0.99, gae_lambda=0.95)
    assert rollout.observacoes.shape[0] == 3


def test_rollout_ultimo_passo_truncado_se_nao_terminated(cliente_real):
    """Rollout marca último passo como truncated=True se não foi terminated."""
    def politica(obs):
        return 0
    def valor(obs):
        return 0.0
    
    rollout = coletar_rollout(cliente_real, politica, valor, 3, gamma=0.99, gae_lambda=0.95)
    assert not rollout.terminated[-1]
    assert rollout.truncated[-1]


def test_rollout_ultimo_passo_nao_truncado_se_terminated(cliente_real):
    """Rollout não marca último passo como truncated se foi terminated."""
    def politica(obs):
        return 1
    def valor(obs):
        return 0.0
    
    # comprimento=5, começamos em 0, 5 passos à direita chegam ao fim
    rollout = coletar_rollout(cliente_real, politica, valor, 5, gamma=0.99, gae_lambda=0.95)
    assert rollout.terminated[-1]
    assert not rollout.truncated[-1]


def test_rollout_next_values_usa_observacao_transicao(cliente_real):
    """next_values usa observação final da transição, não do episódio seguinte."""
    def politica(obs):
        return 1
    def valor(obs):
        return float(obs)
    
    # Coletar 3 passos: obs [0, 1, 2], next_obs [1, 2, 3]
    rollout = coletar_rollout(cliente_real, politica, valor, 3, gamma=0.99, gae_lambda=0.95)
    
    assert rollout.observacoes[0] == 0 and rollout.next_values[0] == 1.0
    assert rollout.observacoes[1] == 1 and rollout.next_values[1] == 2.0
    assert rollout.observacoes[2] == 2 and rollout.next_values[2] == 3.0


def test_rollout_cria_novo_episodio_apos_terminated():
    """Rollout cria novo episódio após terminated, exceto no último passo."""
    app = criar_app()
    http = TestClient(app)
    cliente = ClienteAmbiente(http, comprimento=2, max_passos=10)
    
    def politica(obs):
        return 1
    def valor(obs):
        return 0.0
    
    # Coletar 4 passos: primeiros 2 terminam episódio, depois cria novo
    rollout = coletar_rollout(cliente, politica, valor, 4, gamma=0.99, gae_lambda=0.95)
    
    assert rollout.observacoes.shape[0] == 4
    assert rollout.terminated[1]  # segundo passo termina


def test_rollout_n_passos_zero_lanca_valor_error():
    """Rollout com n_passos=0 lança ValueError."""
    app = criar_app()
    http = TestClient(app)
    cliente = ClienteAmbiente(http, 5, 10)
    
    def politica(obs):
        return 1
    def valor(obs):
        return 0.0
    
    with pytest.raises(ValueError):
        coletar_rollout(cliente, politica, valor, 0, gamma=0.99, gae_lambda=0.95)


def test_rollout_n_passos_negativo_lanca_valor_error():
    """Rollout com n_passos negativo lança ValueError."""
    app = criar_app()
    http = TestClient(app)
    cliente = ClienteAmbiente(http, 5, 10)
    
    def politica(obs):
        return 1
    def valor(obs):
        return 0.0
    
    with pytest.raises(ValueError):
        coletar_rollout(cliente, politica, valor, -1, gamma=0.99, gae_lambda=0.95)


def test_rollout_n_passos_bool_lanca_valor_error():
    """Rollout com n_passos=True (bool) lança ValueError."""
    app = criar_app()
    http = TestClient(app)
    cliente = ClienteAmbiente(http, 5, 10)
    
    def politica(obs):
        return 1
    def valor(obs):
        return 0.0
    
    with pytest.raises(ValueError):
        coletar_rollout(cliente, politica, valor, True, gamma=0.99, gae_lambda=0.95)


# ============================================================================
# Integração: fluxo completo
# ============================================================================

def test_integracao_criar_passar_remover(cliente_real):
    """Fluxo completo: criar episódio, dar passos, remover."""
    episodio_id, obs = cliente_real.criar_episodio()
    assert obs == 0
    
    t1 = cliente_real.passo(episodio_id, 1)
    assert t1.observacao == 1
    assert t1.recompensa == -1.0
    
    t2 = cliente_real.passo(episodio_id, 1)
    assert t2.observacao == 2
    
    cliente_real.encerrar(episodio_id)
    
    with pytest.raises(ErroAmbiente) as exc:
        cliente_real.passo(episodio_id, 1)
    assert exc.value.status == 404
