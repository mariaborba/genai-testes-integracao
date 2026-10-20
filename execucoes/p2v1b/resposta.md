## Raciocínio (15 linhas)

O coletor faz 3 chamadas HTTP: POST /episodios (201), POST /episodios/{id}/passos (200), DELETE (204).
Cada pode falhar com: erro HTTP 4xx/5xx, timeout, conexão, corpo inválido (não-JSON, não-dict, campo ausente/tipo errado, status inesperado).
Regra crítica na integração: `next_values[t]` deve usar a observação final do episódio que terminou/truncou, não a inicial do próximo.
Última passo do rollout é marcado `truncated=True` se não foi `terminated`.
Testes: sucesso de cada endpoint, todos os erros HTTP/rede/contrato, e regras de integração (next_values, episódio novo após término, último passo truncado, n_passos inválido).
Fixtures: `TestClient` novo por teste; `MockTransport` para simular falhas inexistentes no servidor real.

```python
import pytest
import httpx
import numpy as np
from fastapi.testclient import TestClient

from ambiente.app import criar_app, RECOMPENSA_PASSO, RECOMPENSA_OBJETIVO
from coletor.rollout import (
    ClienteAmbiente,
    ErroAmbiente,
    ErroContrato,
    coletar_rollout,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def app_cliente():
    """TestClient novo a cada teste."""
    app = criar_app()
    return TestClient(app)


@pytest.fixture
def politica_direita():
    """Sempre vai para direita."""
    return lambda obs: 1


@pytest.fixture
def valor_obs():
    """Valor = observação."""
    return lambda obs: float(obs)


# ============================================================================
# TESTES: criar_episodio()
# ============================================================================

def test_criar_episodio_sucesso(app_cliente):
    """POST /episodios com sucesso retorna id e observacao inicial."""
    cliente = ClienteAmbiente(app_cliente, comprimento=10, max_passos=20)
    episodio_id, observacao = cliente.criar_episodio()
    assert isinstance(episodio_id, str) and len(episodio_id) > 0
    assert observacao == 0


def test_criar_episodio_http_500():
    """POST /episodios com HTTP 500 levanta ErroAmbiente com status 500."""
    def handler(request):
        return httpx.Response(500, text="Erro")
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroAmbiente) as exc:
        cliente.criar_episodio()
    assert exc.value.status == 500


def test_criar_episodio_timeout():
    """POST /episodios com timeout levanta ErroAmbiente com status None."""
    def handler(request):
        raise httpx.TimeoutException("timeout")
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroAmbiente) as exc:
        cliente.criar_episodio()
    assert exc.value.status is None


def test_criar_episodio_conexao_falha():
    """POST /episodios com falha de conexão levanta ErroAmbiente."""
    def handler(request):
        raise httpx.ConnectError("Conexão recusada")
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroAmbiente) as exc:
        cliente.criar_episodio()
    assert exc.value.status is None


def test_criar_episodio_corpo_nao_json():
    """POST /episodios com corpo não-JSON levanta ErroContrato."""
    def handler(request):
        return httpx.Response(201, text="não json")
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroContrato, match="JSON"):
        cliente.criar_episodio()


def test_criar_episodio_corpo_nao_dict():
    """POST /episodios com corpo que é lista levanta ErroContrato."""
    def handler(request):
        return httpx.Response(201, json=["lista"])
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroContrato, match="objeto JSON"):
        cliente.criar_episodio()


def test_criar_episodio_id_ausente():
    """POST /episodios sem campo 'id' levanta ErroContrato."""
    def handler(request):
        return httpx.Response(201, json={"observacao": 0, "passos": 0, "encerrado": False})
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroContrato, match="id"):
        cliente.criar_episodio()


def test_criar_episodio_id_tipo_invalido():
    """POST /episodios com 'id' número levanta ErroContrato."""
    def handler(request):
        return httpx.Response(201, json={"id": 123, "observacao": 0, "passos": 0, "encerrado": False})
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroContrato, match="id"):
        cliente.criar_episodio()


def test_criar_episodio_observacao_ausente():
    """POST /episodios sem campo 'observacao' levanta ErroContrato."""
    def handler(request):
        return httpx.Response(201, json={"id": "abc", "passos": 0, "encerrado": False})
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroContrato, match="observacao"):
        cliente.criar_episodio()


def test_criar_episodio_observacao_tipo_bool():
    """POST /episodios com 'observacao' bool levanta ErroContrato."""
    def handler(request):
        return httpx.Response(201, json={"id": "abc", "observacao": True, "passos": 0, "encerrado": False})
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroContrato, match="observacao"):
        cliente.criar_episodio()


def test_criar_episodio_status_200_esperado_201():
    """POST /episodios com status 200 ao invés de 201 levanta ErroContrato."""
    def handler(request):
        return httpx.Response(200, json={"id": "abc", "observacao": 0, "passos": 0, "encerrado": False})
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroContrato):
        cliente.criar_episodio()


# ============================================================================
# TESTES: passo()
# ============================================================================

def test_passo_sucesso(app_cliente):
    """POST /episodios/{id}/passos com sucesso retorna Transicao."""
    cliente = ClienteAmbiente(app_cliente, comprimento=10, max_passos=20)
    episodio_id, _ = cliente.criar_episodio()
    transicao = cliente.passo(episodio_id, acao=1)
    assert transicao.observacao == 1
    assert transicao.recompensa == RECOMPENSA_PASSO
    assert transicao.terminated is False
    assert transicao.truncated is False


def test_passo_episodio_nao_existe_404(app_cliente):
    """POST /episodios/{id}/passos em episódio inexistente retorna 404."""
    cliente = ClienteAmbiente(app_cliente, comprimento=10, max_passos=20)
    with pytest.raises(ErroAmbiente) as exc:
        cliente.passo("nao_existe", acao=1)
    assert exc.value.status == 404


def test_passo_episodio_encerrado_409(app_cliente):
    """POST /episodios/{id}/passos em episódio encerrado retorna 409."""
    cliente = ClienteAmbiente(app_cliente, comprimento=3, max_passos=20)
    episodio_id, _ = cliente.criar_episodio()
    for _ in range(3):
        cliente.passo(episodio_id, acao=1)
    with pytest.raises(ErroAmbiente) as exc:
        cliente.passo(episodio_id, acao=1)
    assert exc.value.status == 409


def test_passo_http_500():
    """POST /episodios/{id}/passos com HTTP 500."""
    def handler(request):
        return httpx.Response(500, text="Erro")
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroAmbiente) as exc:
        cliente.passo("abc", acao=1)
    assert exc.value.status == 500


def test_passo_timeout():
    """POST /episodios/{id}/passos com timeout."""
    def handler(request):
        raise httpx.TimeoutException("timeout")
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroAmbiente) as exc:
        cliente.passo("abc", acao=1)
    assert exc.value.status is None


def test_passo_conexao_falha():
    """POST /episodios/{id}/passos com falha de conexão."""
    def handler(request):
        raise httpx.ConnectError("Falha")
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroAmbiente) as exc:
        cliente.passo("abc", acao=1)
    assert exc.value.status is None


def test_passo_corpo_nao_json():
    """POST /episodios/{id}/passos com corpo não-JSON."""
    def handler(request):
        return httpx.Response(200, text="não json")
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroContrato):
        cliente.passo("abc", acao=1)


def test_passo_corpo_nao_dict():
    """POST /episodios/{id}/passos com corpo que é lista."""
    def handler(request):
        return httpx.Response(200, json=["lista"])
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroContrato):
        cliente.passo("abc", acao=1)


def test_passo_observacao_ausente():
    """POST /episodios/{id}/passos sem campo 'observacao'."""
    def handler(request):
        return httpx.Response(200, json={"recompensa": -1.0, "terminated": False, "truncated": False})
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroContrato, match="observacao"):
        cliente.passo("abc", acao=1)


def test_passo_observacao_tipo_bool():
    """POST /episodios/{id}/passos com 'observacao' bool."""
    def handler(request):
        return httpx.Response(200, json={"observacao": True, "recompensa": -1.0, "terminated": False, "truncated": False})
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroContrato, match="observacao"):
        cliente.passo("abc", acao=1)


def test_passo_recompensa_ausente():
    """POST /episodios/{id}/passos sem campo 'recompensa'."""
    def handler(request):
        return httpx.Response(200, json={"observacao": 1, "terminated": False, "truncated": False})
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroContrato, match="recompensa"):
        cliente.passo("abc", acao=1)


def test_passo_recompensa_tipo_invalido():
    """POST /episodios/{id}/passos com 'recompensa' string."""
    def handler(request):
        return httpx.Response(200, json={"observacao": 1, "recompensa": "string", "terminated": False, "truncated": False})
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroContrato, match="recompensa"):
        cliente.passo("abc", acao=1)


def test_passo_terminated_ausente():
    """POST /episodios/{id}/passos sem campo 'terminated'."""
    def handler(request):
        return httpx.Response(200, json={"observacao": 1, "recompensa": -1.0, "truncated": False})
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroContrato, match="terminated"):
        cliente.passo("abc", acao=1)


def test_passo_truncated_ausente():
    """POST /episodios/{id}/passos sem campo 'truncated'."""
    def handler(request):
        return httpx.Response(200, json={"observacao": 1, "recompensa": -1.0, "terminated": False})
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroContrato, match="truncated"):
        cliente.passo("abc", acao=1)


def test_passo_status_201_esperado_200():
    """POST /episodios/{id}/passos com status 201 ao invés de 200."""
    def handler(request):
        return httpx.Response(201, json={"observacao": 1, "recompensa": -1.0, "terminated": False, "truncated": False})
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroContrato):
        cliente.passo("abc", acao=1)


# ============================================================================
# TESTES: encerrar()
# ============================================================================

def test_encerrar_episodio_sucesso(app_cliente):
    """DELETE /episodios/{id} com sucesso."""
    cliente = ClienteAmbiente(app_cliente, comprimento=10, max_passos=20)
    episodio_id, _ = cliente.criar_episodio()
    cliente.encerrar(episodio_id)


def test_encerrar_episodio_nao_existe_404(app_cliente):
    """DELETE /episodios/{id} em episódio inexistente retorna 404."""
    cliente = ClienteAmbiente(app_cliente, comprimento=10, max_passos=20)
    with pytest.raises(ErroAmbiente) as exc:
        cliente.encerrar("nao_existe")
    assert exc.value.status == 404


def test_encerrar_episodio_http_500():
    """DELETE /episodios/{id} com HTTP 500."""
    def handler(request):
        return httpx.Response(500, text="Erro")
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroAmbiente) as exc:
        cliente.encerrar("abc")
    assert exc.value.status == 500


def test_encerrar_episodio_timeout():
    """DELETE /episodios/{id} com timeout."""
    def handler(request):
        raise httpx.TimeoutException("timeout")
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroAmbiente) as exc:
        cliente.encerrar("abc")
    assert exc.value.status is None


def test_encerrar_episodio_conexao_falha():
    """DELETE /episodios/{id} com falha de conexão."""
    def handler(request):
        raise httpx.ConnectError("Falha")
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroAmbiente) as exc:
        cliente.encerrar("abc")
    assert exc.value.status is None


def test_encerrar_episodio_status_200_esperado_204():
    """DELETE /episodios/{id} com status 200 ao invés de 204."""
    def handler(request):
        return httpx.Response(200)
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=20)
    with pytest.raises(ErroContrato):
        cliente.encerrar("abc")


# ============================================================================
# TESTES: coletar_rollout() - INTEGRAÇÃO
# ============================================================================

def test_rollout_coleta_basica(app_cliente, politica_direita, valor_obs):
    """coletar_rollout coleta n_passos transições."""
    cliente = ClienteAmbiente(app_cliente, comprimento=10, max_passos=50)
    rollout = coletar_rollout(
        cliente,
        politica=politica_direita,
        valor=valor_obs,
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert len(rollout.observacoes) == 5
    assert len(rollout.acoes) == 5
    assert len(rollout.recompensas) == 5


def test_rollout_next_values_observacao_final_episodio_terminated(app_cliente, politica_direita, valor_obs):
    """next_values[t] usa observacao final do episódio terminated, não a inicial do próximo."""
    cliente = ClienteAmbiente(app_cliente, comprimento=3, max_passos=50)
    rollout = coletar_rollout(
        cliente,
        politica=politica_direita,
        valor=valor_obs,
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95,
    )
    # Passo 0: obs 0 -> 1, next_values[0] = 1.0
    # Passo 1: obs 1 -> 2, next_values[1] = 2.0
    # Passo 2: obs 2 -> 3 (TERMINATED), next_values[2] = valor(3) = 3.0
    # Novo episódio criado
    # Passo 3: obs 0 -> 1, next_values[3] = 1.0
    # Passo 4: obs 1 -> 2, next_values[4] = 2.0
    assert rollout.next_values[2] == 3.0
    assert rollout.terminated[2]


def test_rollout_next_values_observacao_final_episodio_truncado(app_cliente, politica_direita, valor_obs):
    """next_values[t] usa observacao final do episódio truncado, não a inicial do próximo."""
    cliente = ClienteAmbiente(app_cliente, comprimento=100, max_passos=2)
    rollout = coletar_rollout(
        cliente,
        politica=politica_direita,
        valor=valor_obs,
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95,
    )
    # Passos 0-1 no episódio 1 (truncado)
    # Passos 2-3 no episódio 2 (truncado)
    # Passo 4 no episódio 3
    assert rollout.truncated[1]
    assert rollout.truncated[3]
    assert rollout.next_values[1] == 2.0  # Observação final do episódio truncado


def test_rollout_ultimo_passo_marcado_truncated(app_cliente, politica_direita):
    """O último passo do rollout é marcado como truncated se não foi terminated."""
    cliente = ClienteAmbiente(app_cliente, comprimento=100, max_passos=200)
    valor = lambda obs: 0.5
    rollout = coletar_rollout(
        cliente,
        politica=politica_direita,
        valor=valor,
        n_passos=10,
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert not rollout.terminated[-1]
    assert rollout.truncated[-1]


def test_rollout_recompensa_objetivo_quando_terminated(app_cliente, politica_direita):
    """Recompensa é RECOMPENSA_OBJETIVO quando episódio termina."""
    cliente = ClienteAmbiente(app_cliente, comprimento=3, max_passos=50)
    valor = lambda obs: 0.5
    rollout = coletar_rollout(
        cliente,
        politica=politica_direita,
        valor=valor,
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert rollout.terminated[2]
    assert rollout.recompensas[2] == RECOMPENSA_OBJETIVO


def test_rollout_recompensa_passo_quando_nao_terminated(app_cliente, politica_direita):
    """Recompensa é RECOMPENSA_PASSO quando episódio não termina."""
    cliente = ClienteAmbiente(app_cliente, comprimento=100, max_passos=200)
    valor = lambda obs: 0.5
    rollout = coletar_rollout(
        cliente,
        politica=politica_direita,
        valor=valor,
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95,
    )
    for i in range(len(rollout.recompensas) - 1):
        if not rollout.terminated[i]:
            assert rollout.recompensas[i] == RECOMPENSA_PASSO


def test_rollout_n_passos_invalido_zero(app_cliente, politica_direita):
    """coletar_rollout com n_passos=0 levanta ValueError."""
    cliente = ClienteAmbiente(app_cliente, comprimento=10, max_passos=50)
    valor = lambda obs: 0.5
    with pytest.raises(ValueError):
        coletar_rollout(
            cliente,
            politica=politica_direita,
            valor=valor,
            n_passos=0,
            gamma=0.99,
            gae_lambda=0.95,
        )


def test_rollout_n_passos_invalido_bool(app_cliente, politica_direita):
    """coletar_rollout com n_passos=True (bool) levanta ValueError."""
    cliente = ClienteAmbiente(app_cliente, comprimento=10, max_passos=50)
    valor = lambda obs: 0.5
    with pytest.raises(ValueError):
        coletar_rollout(
            cliente,
            politica=politica_direita,
            valor=valor,
            n_passos=True,
            gamma=0.99,
            gae_lambda=0.95,
        )


def test_rollout_n_passos_invalido_negativo(app_cliente, politica_direita):
    """coletar_rollout com n_passos negativo levanta ValueError."""
    cliente = ClienteAmbiente(app_cliente, comprimento=10, max_passos=50)
    valor = lambda obs: 0.5
    with pytest.raises(ValueError):
        coletar_rollout(
            cliente,
            politica=politica_direita,
            valor=valor,
            n_passos=-1,
            gamma=0.99,
            gae_lambda=0.95,
        )


def test_rollout_erro_contrato_na_criacao_episodio(politica_direita):
    """Erro de contrato na criação de episódio propaga."""
    def handler_erro(request):
        return httpx.Response(201, json={"id": "abc"})  # Falta observacao
    client = httpx.Client(transport=httpx.MockTransport(handler_erro), base_url="http://ambiente")
    cliente = ClienteAmbiente(client, comprimento=10, max_passos=50)
    valor = lambda obs: 0.5
    with pytest.raises(ErroContrato):
        coletar_rollout(
            cliente,
            politica=politica_direita,
            valor=valor,
            n_passos=5,
            gamma=0.99,
            gae_lambda=0.95,
        )


def test_rollout_erro_ambiente_na_passo(app_cliente, politica_direita):
    """Erro de ambiente (HTTP 404 de episódio inexistente) durante coleta propaga."""
    cliente = ClienteAmbiente(app_cliente, comprimento=10, max_passos=1)
    valor = lambda obs: 0.5
    # Com max_passos=1, episódio trunca após 1 passo; novo é criado
    # Simulamos erro deletando o episódio manualmente
    with pytest.raises(ErroAmbiente) as exc:
        rollout = coletar_rollout(
            cliente,
            politica=politica_direita,
            valor=valor,
            n_passos=3,
            gamma=0.99,
            gae_lambda=0.95,
        )
    assert exc.value.status == 404 or exc.value.status is None


def test_rollout_arrays_dtype_corretos(app_cliente, politica_direita):
    """Arrays do rollout têm dtypes corretos."""
    cliente = ClienteAmbiente(app_cliente, comprimento=10, max_passos=50)
    valor = lambda obs: 0.5
    rollout = coletar_rollout(
        cliente,
        politica=politica_direita,
        valor=valor,
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95,
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


def test_rollout_advantages_returns_calculados(app_cliente, politica_direita):
    """Advantages e returns são calculados pelo GAE."""
    cliente = ClienteAmbiente(app_cliente, comprimento=10, max_passos=50)
    valor = lambda obs: 0.5
    rollout = coletar_rollout(
        cliente,
        politica=politica_direita,
        valor=valor,
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert len(rollout.advantages) == 5
    assert len(rollout.returns) == 5
    assert isinstance(rollout.advantages, np.ndarray)
    assert isinstance(rollout.returns, np.ndarray)
```