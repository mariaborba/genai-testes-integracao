import pytest
import httpx
import numpy as np
from unittest.mock import patch
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
def cliente_teste():
    """Fixture que retorna um TestClient novo para cada teste."""
    return TestClient(criar_app())


# ============================================================================
# TESTES DA API REST: POST /episodios
# ============================================================================

def test_criar_episodio_sucesso(cliente_teste):
    """Criar episódio com parâmetros válidos retorna 201 e EstadoEpisodio."""
    response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 5, "max_passos": 10}
    )
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["observacao"] == 0
    assert data["passos"] == 0
    assert data["encerrado"] is False


def test_criar_episodio_comprimento_minimo(cliente_teste):
    """Criar episódio com comprimento mínimo (2)."""
    response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 2, "max_passos": 1}
    )
    assert response.status_code == 201
    assert response.json()["observacao"] == 0


def test_criar_episodio_comprimento_maximo(cliente_teste):
    """Criar episódio com comprimento máximo (20)."""
    response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 20, "max_passos": 500}
    )
    assert response.status_code == 201


def test_criar_episodio_comprimento_abaixo_minimo(cliente_teste):
    """Criar episódio com comprimento < 2 retorna 422."""
    response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 1, "max_passos": 10}
    )
    assert response.status_code == 422


def test_criar_episodio_comprimento_acima_maximo(cliente_teste):
    """Criar episódio com comprimento > 20 retorna 422."""
    response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 21, "max_passos": 10}
    )
    assert response.status_code == 422


def test_criar_episodio_max_passos_minimo(cliente_teste):
    """Criar episódio com max_passos mínimo (1)."""
    response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 5, "max_passos": 1}
    )
    assert response.status_code == 201


def test_criar_episodio_max_passos_maximo(cliente_teste):
    """Criar episódio com max_passos máximo (500)."""
    response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 5, "max_passos": 500}
    )
    assert response.status_code == 201


def test_criar_episodio_max_passos_abaixo_minimo(cliente_teste):
    """Criar episódio com max_passos < 1 retorna 422."""
    response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 5, "max_passos": 0}
    )
    assert response.status_code == 422


def test_criar_episodio_max_passos_acima_maximo(cliente_teste):
    """Criar episódio com max_passos > 500 retorna 422."""
    response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 5, "max_passos": 501}
    )
    assert response.status_code == 422


def test_criar_episodio_falta_comprimento(cliente_teste):
    """Criar episódio sem comprimento retorna 422."""
    response = cliente_teste.post(
        "/episodios",
        json={"max_passos": 10}
    )
    assert response.status_code == 422


def test_criar_episodio_falta_max_passos(cliente_teste):
    """Criar episódio sem max_passos retorna 422."""
    response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 5}
    )
    assert response.status_code == 422


def test_criar_episodio_corpo_vazio(cliente_teste):
    """Criar episódio com corpo vazio retorna 422."""
    response = cliente_teste.post(
        "/episodios",
        json={}
    )
    assert response.status_code == 422


# ============================================================================
# TESTES DA API REST: GET /episodios/{id}
# ============================================================================

def test_consultar_episodio_sucesso(cliente_teste):
    """Consultar episódio existente retorna 200 e EstadoEpisodio."""
    create_response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 5, "max_passos": 10}
    )
    episodio_id = create_response.json()["id"]

    response = cliente_teste.get(f"/episodios/{episodio_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == episodio_id
    assert data["observacao"] == 0
    assert data["passos"] == 0
    assert data["encerrado"] is False


def test_consultar_episodio_nao_existe(cliente_teste):
    """Consultar episódio inexistente retorna 404."""
    response = cliente_teste.get("/episodios/nao-existe")
    assert response.status_code == 404


def test_consultar_episodio_apos_passo(cliente_teste):
    """Consultar episódio após passo mostra estado atualizado."""
    create_response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 5, "max_passos": 10}
    )
    episodio_id = create_response.json()["id"]

    cliente_teste.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})

    response = cliente_teste.get(f"/episodios/{episodio_id}")
    assert response.json()["observacao"] == 1
    assert response.json()["passos"] == 1


# ============================================================================
# TESTES DA API REST: DELETE /episodios/{id}
# ============================================================================

def test_remover_episodio_sucesso(cliente_teste):
    """Remover episódio existente retorna 204."""
    create_response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 5, "max_passos": 10}
    )
    episodio_id = create_response.json()["id"]

    response = cliente_teste.delete(f"/episodios/{episodio_id}")
    assert response.status_code == 204

    get_response = cliente_teste.get(f"/episodios/{episodio_id}")
    assert get_response.status_code == 404


def test_remover_episodio_nao_existe(cliente_teste):
    """Remover episódio inexistente retorna 404."""
    response = cliente_teste.delete("/episodios/nao-existe")
    assert response.status_code == 404


def test_remover_episodio_duplicado(cliente_teste):
    """Remover o mesmo episódio duas vezes: segunda vez retorna 404."""
    create_response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 5, "max_passos": 10}
    )
    episodio_id = create_response.json()["id"]

    response1 = cliente_teste.delete(f"/episodios/{episodio_id}")
    assert response1.status_code == 204

    response2 = cliente_teste.delete(f"/episodios/{episodio_id}")
    assert response2.status_code == 404


# ============================================================================
# TESTES DA API REST: POST /episodios/{id}/passos
# ============================================================================

def test_dar_passo_sucesso_esquerda(cliente_teste):
    """Dar passo à esquerda (ação 0) retorna 200 e Transicao."""
    create_response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 5, "max_passos": 10}
    )
    episodio_id = create_response.json()["id"]

    response = cliente_teste.post(
        f"/episodios/{episodio_id}/passos",
        json={"acao": 0}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["observacao"] == 0
    assert data["recompensa"] == -1.0
    assert data["terminated"] is False
    assert data["truncated"] is False


def test_dar_passo_sucesso_direita(cliente_teste):
    """Dar passo à direita (ação 1) retorna 200 e Transicao."""
    create_response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 5, "max_passos": 10}
    )
    episodio_id = create_response.json()["id"]

    response = cliente_teste.post(
        f"/episodios/{episodio_id}/passos",
        json={"acao": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["observacao"] == 1
    assert data["recompensa"] == -1.0
    assert data["terminated"] is False
    assert data["truncated"] is False


def test_dar_passo_episodio_nao_existe(cliente_teste):
    """Dar passo em episódio inexistente retorna 404."""
    response = cliente_teste.post(
        "/episodios/nao-existe/passos",
        json={"acao": 1}
    )
    assert response.status_code == 404


def test_dar_passo_acao_invalida_negativa(cliente_teste):
    """Dar passo com ação < 0 retorna 422."""
    create_response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 5, "max_passos": 10}
    )
    episodio_id = create_response.json()["id"]

    response = cliente_teste.post(
        f"/episodios/{episodio_id}/passos",
        json={"acao": -1}
    )
    assert response.status_code == 422


def test_dar_passo_acao_invalida_grande(cliente_teste):
    """Dar passo com ação > 1 retorna 422."""
    create_response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 5, "max_passos": 10}
    )
    episodio_id = create_response.json()["id"]

    response = cliente_teste.post(
        f"/episodios/{episodio_id}/passos",
        json={"acao": 2}
    )
    assert response.status_code == 422


def test_dar_passo_acao_faltando(cliente_teste):
    """Dar passo sem ação retorna 422."""
    create_response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 5, "max_passos": 10}
    )
    episodio_id = create_response.json()["id"]

    response = cliente_teste.post(
        f"/episodios/{episodio_id}/passos",
        json={}
    )
    assert response.status_code == 422


# ============================================================================
# TESTES DO COMPORTAMENTO DO AMBIENTE
# ============================================================================

def test_ambiente_comeca_em_zero(cliente_teste):
    """Agente começa na posição 0."""
    response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 5, "max_passos": 10}
    )
    assert response.json()["observacao"] == 0


def test_ambiente_acao_direita_incrementa(cliente_teste):
    """Ação 1 move agente para direita."""
    create_response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 5, "max_passos": 10}
    )
    episodio_id = create_response.json()["id"]

    response = cliente_teste.post(
        f"/episodios/{episodio_id}/passos",
        json={"acao": 1}
    )
    assert response.json()["observacao"] == 1

    response = cliente_teste.post(
        f"/episodios/{episodio_id}/passos",
        json={"acao": 1}
    )
    assert response.json()["observacao"] == 2


def test_ambiente_acao_esquerda_decrementa(cliente_teste):
    """Ação 0 move agente para esquerda."""
    create_response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 5, "max_passos": 10}
    )
    episodio_id = create_response.json()["id"]

    cliente_teste.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})
    cliente_teste.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})

    response = cliente_teste.post(
        f"/episodios/{episodio_id}/passos",
        json={"acao": 0}
    )
    assert response.json()["observacao"] == 1


def test_ambiente_acao_esquerda_em_zero_fica_zero(cliente_teste):
    """Ação 0 na posição 0 mantém agente em 0."""
    create_response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 5, "max_passos": 10}
    )
    episodio_id = create_response.json()["id"]

    response = cliente_teste.post(
        f"/episodios/{episodio_id}/passos",
        json={"acao": 0}
    )
    assert response.json()["observacao"] == 0


def test_ambiente_recompensa_passo_normal(cliente_teste):
    """Cada passo normal vale -1.0 de recompensa."""
    create_response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 5, "max_passos": 10}
    )
    episodio_id = create_response.json()["id"]

    response = cliente_teste.post(
        f"/episodios/{episodio_id}/passos",
        json={"acao": 1}
    )
    assert response.json()["recompensa"] == -1.0


def test_ambiente_chegada_objetivo_recompensa_bonus(cliente_teste):
    """Chegar ao objetivo vale +10.0 de recompensa."""
    create_response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 2, "max_passos": 10}
    )
    episodio_id = create_response.json()["id"]

    response = cliente_teste.post(
        f"/episodios/{episodio_id}/passos",
        json={"acao": 1}
    )
    assert response.json()["recompensa"] == -1.0

    response = cliente_teste.post(
        f"/episodios/{episodio_id}/passos",
        json={"acao": 1}
    )
    assert response.json()["observacao"] == 2
    assert response.json()["recompensa"] == 10.0


def test_ambiente_chegada_objetivo_terminated(cliente_teste):
    """Chegar ao objetivo faz terminated=true."""
    create_response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 2, "max_passos": 10}
    )
    episodio_id = create_response.json()["id"]

    cliente_teste.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})
    response = cliente_teste.post(
        f"/episodios/{episodio_id}/passos",
        json={"acao": 1}
    )
    assert response.json()["terminated"] is True
    assert response.json()["truncated"] is False


def test_ambiente_max_passos_truncated(cliente_teste):
    """Atingir max_passos faz truncated=true."""
    create_response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 10, "max_passos": 2}
    )
    episodio_id = create_response.json()["id"]

    cliente_teste.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})
    response = cliente_teste.post(
        f"/episodios/{episodio_id}/passos",
        json={"acao": 1}
    )
    assert response.json()["truncated"] is True
    assert response.json()["terminated"] is False


def test_ambiente_terminated_e_truncated_ambos_true(cliente_teste):
    """Tanto terminated quanto truncated podem ser true no mesmo passo."""
    create_response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 2, "max_passos": 2}
    )
    episodio_id = create_response.json()["id"]

    cliente_teste.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})
    response = cliente_teste.post(
        f"/episodios/{episodio_id}/passos",
        json={"acao": 1}
    )
    assert response.json()["terminated"] is True
    assert response.json()["truncated"] is True


def test_ambiente_episodio_encerrado_409(cliente_teste):
    """Depois de terminated, novo passo retorna 409."""
    create_response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 2, "max_passos": 10}
    )
    episodio_id = create_response.json()["id"]

    cliente_teste.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})
    cliente_teste.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})

    response = cliente_teste.post(
        f"/episodios/{episodio_id}/passos",
        json={"acao": 1}
    )
    assert response.status_code == 409


def test_ambiente_episodio_truncado_depois_novo_409(cliente_teste):
    """Depois de truncated, novo passo também retorna 409."""
    create_response = cliente_teste.post(
        "/episodios",
        json={"comprimento": 10, "max_passos": 2}
    )
    episodio_id = create_response.json()["id"]

    cliente_teste.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})
    cliente_teste.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})

    response = cliente_teste.post(
        f"/episodios/{episodio_id}/passos",
        json={"acao": 1}
    )
    assert response.status_code == 409


# ============================================================================
# TESTES DO ClienteAmbiente
# ============================================================================

def test_cliente_criar_episodio(cliente_teste):
    """ClienteAmbiente.criar_episodio retorna (id, observacao_inicial=0)."""
    cliente = ClienteAmbiente(cliente_teste, comprimento=5, max_passos=10)
    episodio_id, obs = cliente.criar_episodio()
    assert isinstance(episodio_id, str)
    assert obs == 0


def test_cliente_passo(cliente_teste):
    """ClienteAmbiente.passo retorna Transicao."""
    cliente = ClienteAmbiente(cliente_teste, comprimento=5, max_passos=10)
    episodio_id, _ = cliente.criar_episodio()

    transicao = cliente.passo(episodio_id, acao=1)
    assert isinstance(transicao, Transicao)
    assert transicao.observacao == 1
    assert transicao.recompensa == -1.0
    assert transicao.terminated is False
    assert transicao.truncated is False


def test_cliente_encerrar(cliente_teste):
    """ClienteAmbiente.encerrar remove episódio."""
    cliente = ClienteAmbiente(cliente_teste, comprimento=5, max_passos=10)
    episodio_id, _ = cliente.criar_episodio()

    cliente.encerrar(episodio_id)

    response = cliente_teste.get(f"/episodios/{episodio_id}")
    assert response.status_code == 404


def test_cliente_passo_apos_terminar_erro(cliente_teste):
    """ClienteAmbiente.passo em episódio terminado levanta ErroAmbiente."""
    cliente = ClienteAmbiente(cliente_teste, comprimento=2, max_passos=10)
    episodio_id, _ = cliente.criar_episodio()

    cliente.passo(episodio_id, acao=1)
    cliente.passo(episodio_id, acao=1)

    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.passo(episodio_id, acao=1)
    assert exc_info.value.status == 409


# ============================================================================
# TESTES DE coletar_rollout: COLETA BÁSICA
# ============================================================================

def test_coletor_rollout_basico(cliente_teste):
    """coletar_rollout coleta n_passos transições."""
    cliente = ClienteAmbiente(cliente_teste, comprimento=10, max_passos=100)
    politica = lambda obs: 1
    valor = lambda obs: float(obs)

    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=valor,
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert len(rollout.observacoes) == 5
    assert len(rollout.acoes) == 5
    assert len(rollout.recompensas) == 5
    assert len(rollout.values) == 5
    assert len(rollout.next_values) == 5
    assert len(rollout.terminated) == 5
    assert len(rollout.truncated) == 5
    assert len(rollout.advantages) == 5
    assert len(rollout.returns) == 5


def test_coletor_rollout_observacoes_corretas(cliente_teste):
    """Observações coletadas são: posição antes de cada passo."""
    cliente = ClienteAmbiente(cliente_teste, comprimento=10, max_passos=100)
    politica = lambda obs: 1

    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=lambda obs: float(obs),
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95,
    )

    np.testing.assert_array_equal(rollout.observacoes, [0, 1, 2, 3, 4])


def test_coletor_rollout_acoes_corretas(cliente_teste):
    """Ações coletadas seguem a política."""
    cliente = ClienteAmbiente(cliente_teste, comprimento=10, max_passos=100)
    politica = lambda obs: 1

    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=lambda obs: float(obs),
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95,
    )

    np.testing.assert_array_equal(rollout.acoes, [1, 1, 1, 1, 1])


def test_coletor_rollout_recompensas_corretas(cliente_teste):
    """Recompensas coletadas são -1.0 para cada passo (sem fim)."""
    cliente = ClienteAmbiente(cliente_teste, comprimento=10, max_passos=100)
    politica = lambda obs: 1

    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=lambda obs: float(obs),
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95,
    )

    np.testing.assert_array_equal(rollout.recompensas, [-1.0, -1.0, -1.0, -1.0, -1.0])


def test_coletor_rollout_values_corretos(cliente_teste):
    """Values são calculados ANTES de cada passo."""
    cliente = ClienteAmbiente(cliente_teste, comprimento=10, max_passos=100)
    valor = lambda obs: float(obs)
    politica = lambda obs: 1

    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=valor,
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95,
    )

    np.testing.assert_array_equal(rollout.values, [0.0, 1.0, 2.0, 3.0, 4.0])


def test_coletor_rollout_next_values_diferentes_values(cliente_teste):
    """next_values são observações finais, diferentes de values."""
    cliente = ClienteAmbiente(cliente_teste, comprimento=10, max_passos=100)
    valor = lambda obs: float(obs)
    politica = lambda obs: 1

    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=valor,
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95,
    )

    np.testing.assert_array_equal(rollout.next_values, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert not np.array_equal(rollout.next_values, rollout.values)


def test_coletor_rollout_terminated_false_semacabou(cliente_teste):
    """terminated é False quando episódio não acabou."""
    cliente = ClienteAmbiente(cliente_teste, comprimento=10, max_passos=100)
    politica = lambda obs: 1
    valor = lambda obs: float(obs)

    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=valor,
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95,
    )

    np.testing.assert_array_equal(rollout.terminated, [False, False, False, False, False])


def test_coletor_rollout_ultimo_truncated(cliente_teste):
    """Último passo é truncated se rollout interrompe."""
    cliente = ClienteAmbiente(cliente_teste, comprimento=10, max_passos=100)
    politica = lambda obs: 1
    valor = lambda obs: float(obs)

    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=valor,
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert rollout.truncated[-1] is True


# ============================================================================
# TESTES DE coletar_rollout: EPISÓDIO ACABA DENTRO DO ROLLOUT
# ============================================================================

def test_coletor_episodio_acaba_meio_do_rollout_passo_1(cliente_teste):
    """Episódio acaba DENTRO do rollout; é recriado (exceto no fim).
    
    Comentário: o episódio termina no passo 1 (posição 2 = objetivo).
    """
    cliente = ClienteAmbiente(cliente_teste, comprimento=2, max_passos=100)
    politica = lambda obs: 1
    valor = lambda obs: float(obs)

    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=valor,
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # Passo 0: obs=0, ação=1 -> 1, r=-1, terminated=False
    # Passo 1: obs=1, ação=1 -> 2, r=+10, terminated=True (episódio recria)
    # Passo 2: obs=0, ação=1 -> 1, r=-1, terminated=False
    # Passo 3: obs=1, ação=1 -> 2, r=+10, terminated=True (episódio recria)
    # Passo 4: obs=0, ação=1 -> 1, r=-1, truncated=True (rollout interrompe)
    
    np.testing.assert_array_equal(rollout.observacoes, [0, 1, 0, 1, 0])
    np.testing.assert_array_equal(rollout.recompensas, [-1.0, 10.0, -1.0, 10.0, -1.0])
    np.testing.assert_array_equal(rollout.terminated, [False, True, False, True, False])
    np.testing.assert_array_equal(rollout.next_values, [1.0, 2.0, 1.0, 2.0, 1.0])
    assert rollout.truncated[-1] is True


def test_coletor_episodio_acaba_no_ultimo_passo(cliente_teste):
    """Episódio acaba no último passo: não é recriado.
    
    Comentário: o episódio termina no passo 2 (último passo do rollout).
    """
    cliente = ClienteAmbiente(cliente_teste, comprimento=3, max_passos=100)
    politica = lambda obs: 1
    valor = lambda obs: float(obs)

    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=valor,
        n_passos=3,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # Passo 0: obs=0 -> 1, r=-1, terminated=False
    # Passo 1: obs=1 -> 2, r=-1, terminated=False
    # Passo 2: obs=2 -> 3, r=+10, terminated=True (último, não recria)
    
    np.testing.assert_array_equal(rollout.observacoes, [0, 1, 2])
    np.testing.assert_array_equal(rollout.recompensas, [-1.0, -1.0, 10.0])
    np.testing.assert_array_equal(rollout.terminated, [False, False, True])
    np.testing.assert_array_equal(rollout.truncated, [False, False, False])


def test_coletor_episodio_trunca_meio_do_rollout_passo_2(cliente_teste):
    """Episódio trunca por max_passos DENTRO do rollout; é recriado.
    
    Comentário: o episódio trunca no passo 2 (atingiu max_passos=3).
    """
    cliente = ClienteAmbiente(cliente_teste, comprimento=20, max_passos=3)
    politica = lambda obs: 1
    valor = lambda obs: float(obs)

    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=valor,
        n_passos=6,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # Passo 0: obs=0, count=1 -> 1
    # Passo 1: obs=1, count=2 -> 2
    # Passo 2: obs=2, count=3 -> 3, truncated=True (episódio recria)
    # Passo 3: obs=0, count=1 -> 1
    # Passo 4: obs=1, count=2 -> 2
    # Passo 5: obs=2, count=3 -> 3, truncated=True (rollout interrompe)
    
    np.testing.assert_array_equal(rollout.observacoes, [0, 1, 2, 0, 1, 2])
    np.testing.assert_array_equal(rollout.acoes, [1, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(rollout.recompensas, [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    np.testing.assert_array_equal(rollout.terminated, [False, False, False, False, False, False])
    assert rollout.truncated[2] is True
    assert rollout.truncated[5] is True


def test_coletor_episodio_terminated_e_truncado_meio(cliente_teste):
    """Episódio pode terminar E truncar no mesmo passo, dentro do rollout.
    
    Comentário: o episódio termina E trunca no passo 1 (pos=2, max_passos=2).
    """
    cliente = ClienteAmbiente(cliente_teste, comprimento=2, max_passos=2)
    politica = lambda obs: 1
    valor = lambda obs: float(obs)

    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=valor,
        n_passos=4,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # Passo 0: obs=0, count=1 -> 1, r=-1
    # Passo 1: obs=1, count=2 -> 2, r=+10, terminated=True, truncated=True (recria)
    # Passo 2: obs=0, count=1 -> 1, r=-1
    # Passo 3: obs=1, count=2 -> 2, r=+10, terminated=True, truncated=True (rollout fim)
    
    np.testing.assert_array_equal(rollout.observacoes, [0, 1, 0, 1])
    np.testing.assert_array_equal(rollout.terminated, [False, True, False, True])
    np.testing.assert_array_equal(rollout.truncated, [False, True, False, True])


# ============================================================================
# TESTES DE coletar_rollout: TIPOS E ARRAYS
# ============================================================================

def test_coletor_rollout_arrays_tipos(cliente_teste):
    """Arrays do rollout têm tipos numpy corretos."""
    cliente = ClienteAmbiente(cliente_teste, comprimento=5, max_passos=10)

    rollout = coletar_rollout(
        cliente,
        politica=lambda obs: 1,
        valor=lambda obs: float(obs),
        n_passos=3,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert rollout.observacoes.dtype == np.int64
    assert rollout.acoes.dtype == np.int64
    assert rollout.recompensas.dtype == np.float64
    assert rollout.values.dtype == np.float64
    assert rollout.next_values.dtype == np.float64
    assert rollout.terminated.dtype == np.bool_
    assert rollout.truncated.dtype == np.bool_
    assert rollout.advantages.dtype == np.float64
    assert rollout.returns.dtype == np.float64


def test_coletor_rollout_returns_calculo(cliente_teste):
    """returns = advantages + values."""
    cliente = ClienteAmbiente(cliente_teste, comprimento=10, max_passos=100)

    rollout = coletar_rollout(
        cliente,
        politica=lambda obs: 1,
        valor=lambda obs: float(obs),
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95,
    )

    np.testing.assert_array_almost_equal(
        rollout.returns,
        rollout.advantages + rollout.values
    )


def test_coletor_rollout_politica_dinamica(cliente_teste):
    """Política pode variar com observação."""
    cliente = ClienteAmbiente(cliente_teste, comprimento=10, max_passos=100)

    # Política: vai direita se obs < 3, senão esquerda
    politica = lambda obs: 1 if obs < 3 else 0
    valor = lambda obs: float(obs)

    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=valor,
        n_passos=6,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # 0->1 (a=1), 1->2 (a=1), 2->3 (a=1), 3->2 (a=0), 2->3 (a=1), 3->2 (a=0)
    np.testing.assert_array_equal(rollout.observacoes, [0, 1, 2, 3, 2, 3])
    np.testing.assert_array_equal(rollout.acoes, [1, 1, 1, 0, 1, 0])


def test_coletor_rollout_valor_nao_constante(cliente_teste):
    """Valor deve variar com obs para distinguir final e recriação."""
    cliente = ClienteAmbiente(cliente_teste, comprimento=2, max_passos=100)

    valor = lambda obs: float(obs) * 10

    rollout = coletar_rollout(
        cliente,
        politica=lambda obs: 1,
        valor=valor,
        n_passos=4,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # Passo 0: obs=0, valor=0 -> obs_final=1, valor_next=10
    # Passo 1: obs=1, valor=10 -> obs_final=2, valor_next=20 (terminated)
    # Passo 2: obs=0, valor=0 -> obs_final=1, valor_next=10
    # Passo 3: obs=1, valor=10 -> obs_final=2, valor_next=20
    
    np.testing.assert_array_equal(rollout.values, [0.0, 10.0, 0.0, 10.0])
    np.testing.assert_array_equal(rollout.next_values, [10.0, 20.0, 10.0, 20.0])


# ============================================================================
# TESTES DE coletar_rollout: VALIDAÇÃO
# ============================================================================

def test_coletor_rollout_n_passos_zero(cliente_teste):
    """n_passos=0 levanta ValueError."""
    cliente = ClienteAmbiente(cliente_teste, comprimento=5, max_passos=10)

    with pytest.raises(ValueError):
        coletar_rollout(
            cliente,
            politica=lambda obs: 1,
            valor=lambda obs: float(obs),
            n_passos=0,
            gamma=0.99,
            gae_lambda=0.95,
        )


def test_coletor_rollout_n_passos_negativo(cliente_teste):
    """n_passos<0 levanta ValueError."""
    cliente = ClienteAmbiente(cliente_teste, comprimento=5, max_passos=10)

    with pytest.raises(ValueError):
        coletar_rollout(
            cliente,
            politica=lambda obs: 1,
            valor=lambda obs: float(obs),
            n_passos=-1,
            gamma=0.99,
            gae_lambda=0.95,
        )


# ============================================================================
# TESTES DE coletar_rollout: LIMPEZA E EFEITOS COLATERAIS
# ============================================================================

def test_coletor_rollout_limpeza_episodio_final(cliente_teste):
    """Episódio aberto ao final do rollout é removido."""
    created_ids = []
    original_post = cliente_teste.post

    def tracked_post(url, *args, **kwargs):
        response = original_post(url, *args, **kwargs)
        if url == "/episodios" and response.status_code == 201:
            created_ids.append(response.json()["id"])
        return response

    cliente_teste.post = tracked_post

    cliente = ClienteAmbiente(cliente_teste, comprimento=10, max_passos=100)

    coletar_rollout(
        cliente,
        politica=lambda obs: 1,
        valor=lambda obs: float(obs),
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # Verificar que todos foram removidos
    for episodio_id in created_ids:
        response = cliente_teste.get(f"/episodios/{episodio_id}")
        assert response.status_code == 404


def test_coletor_rollout_limpeza_episodios_reciclados(cliente_teste):
    """Episódios reciclados durante o rollout são também removidos."""
    created_ids = []
    original_post = cliente_teste.post

    def tracked_post(url, *args, **kwargs):
        response = original_post(url, *args, **kwargs)
        if url == "/episodios" and response.status_code == 201:
            created_ids.append(response.json()["id"])
        return response

    cliente_teste.post = tracked_post

    cliente = ClienteAmbiente(cliente_teste, comprimento=2, max_passos=100)

    coletar_rollout(
        cliente,
        politica=lambda obs: 1,
        valor=lambda obs: float(obs),
        n_passos=6,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # Episódio final é removido
    # Episódios intermediários foram removidos após fim
    for episodio_id in created_ids:
        response = cliente_teste.get(f"/episodios/{episodio_id}")
        assert response.status_code == 404


# ============================================================================
# TESTES DE ERRO: ErroAmbiente
# ============================================================================

def test_coletor_erro_http_500():
    """coletar_rollout com HTTP 500 levanta ErroAmbiente com status=500."""
    def handler(request):
        return httpx.Response(500, content=b"Internal Server Error")

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)

    with pytest.raises(ErroAmbiente) as exc_info:
        coletar_rollout(
            cliente,
            politica=lambda obs: 1,
            valor=lambda obs: float(obs),
            n_passos=5,
            gamma=0.99,
            gae_lambda=0.95,
        )
    assert exc_info.value.status == 500


def test_coletor_erro_http_502():
    """coletar_rollout com HTTP 502 levanta ErroAmbiente com status=502."""
    def handler(request):
        return httpx.Response(502, content=b"Bad Gateway")

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)

    with pytest.raises(ErroAmbiente) as exc_info:
        coletar_rollout(
            cliente,
            politica=lambda obs: 1,
            valor=lambda obs: float(obs),
            n_passos=5,
            gamma=0.99,
            gae_lambda=0.95,
        )
    assert exc_info.value.status == 502


def test_coletor_erro_timeout():
    """coletar_rollout com timeout levanta ErroAmbiente com status=None."""
    request_obj = httpx.Request("POST", "http://ambiente/episodios", json={})

    def handler(request):
        raise httpx.ReadTimeout("tempo esgotado", request=request_obj)

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)

    with pytest.raises(ErroAmbiente) as exc_info:
        coletar_rollout(
            cliente,
            politica=lambda obs: 1,
            valor=lambda obs: float(obs),
            n_passos=5,
            gamma=0.99,
            gae_lambda=0.95,
        )
    assert exc_info.value.status is None


def test_coletor_erro_conexao():
    """coletar_rollout com erro de conexão levanta ErroAmbiente com status=None."""
    request_obj = httpx.Request("POST", "http://ambiente/episodios", json={})

    def handler(request):
        raise httpx.ConnectError("recusada", request=request_obj)

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)

    with pytest.raises(ErroAmbiente) as exc_info:
        coletar_rollout(
            cliente,
            politica=lambda obs: 1,
            valor=lambda obs: float(obs),
            n_passos=5,
            gamma=0.99,
            gae_lambda=0.95,
        )
    assert exc_info.value.status is None


# ============================================================================
# TESTES DE ERRO: ErroContrato
# ============================================================================

def test_cliente_erro_contrato_criar_falta_id():
    """ClienteAmbiente levanta ErroContrato se 'id' falta na criação."""
    def handler(request):
        if request.method == "POST":
            return httpx.Response(201, json={"observacao": 0, "passos": 0, "encerrado": False})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)

    with pytest.raises(ErroContrato):
        cliente.criar_episodio()


def test_cliente_erro_contrato_passo_falta_observacao():
    """ClienteAmbiente levanta ErroContrato se 'observacao' falta em passo."""
    def handler(request):
        if request.method == "POST" and "passos" not in request.url.path:
            return httpx.Response(201, json={"id": "123", "observacao": 0, "passos": 0, "encerrado": False})
        elif request.method == "POST":
            return httpx.Response(200, json={"recompensa": -1.0, "terminated": False, "truncated": False})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)

    episodio_id, _ = cliente.criar_episodio()
    with pytest.raises(ErroContrato):
        cliente.passo(episodio_id, 1)


def test_cliente_erro_contrato_passo_falta_recompensa():
    """ClienteAmbiente levanta ErroContrato se 'recompensa' falta em passo."""
    def handler(request):
        if request.method == "POST" and "passos" not in request.url.path:
            return httpx.Response(201, json={"id": "123", "observacao": 0, "passos": 0, "encerrado": False})
        elif request.method == "POST":
            return httpx.Response(200, json={"observacao": 1, "terminated": False, "truncated": False})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)

    episodio_id, _ = cliente.criar_episodio()
    with pytest.raises(ErroContrato):
        cliente.passo(episodio_id, 1)


def test_cliente_erro_contrato_passo_falta_terminated():
    """ClienteAmbiente levanta ErroContrato se 'terminated' falta em passo."""
    def handler(request):
        if request.method == "POST" and "passos" not in request.url.path:
            return httpx.Response(201, json={"id": "123", "observacao": 0, "passos": 0, "encerrado": False})
        elif request.method == "POST":
            return httpx.Response(200, json={"observacao": 1, "recompensa": -1.0, "truncated": False})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)

    episodio_id, _ = cliente.criar_episodio()
    with pytest.raises(ErroContrato):
        cliente.passo(episodio_id, 1)


def test_cliente_erro_contrato_passo_falta_truncated():
    """ClienteAmbiente levanta ErroContrato se 'truncated' falta em passo."""
    def handler(request):
        if request.method == "POST" and "passos" not in request.url.path:
            return httpx.Response(201, json={"id": "123", "observacao": 0, "passos": 0, "encerrado": False})
        elif request.method == "POST":
            return httpx.Response(200, json={"observacao": 1, "recompensa": -1.0, "terminated": False})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)

    episodio_id, _ = cliente.criar_episodio()
    with pytest.raises(ErroContrato):
        cliente.passo(episodio_id, 1)


def test_cliente_erro_contrato_passo_tipo_invalido():
    """ClienteAmbiente levanta ErroContrato se tipos estão errados."""
    def handler(request):
        if request.method == "POST" and "passos" not in request.url.path:
            return httpx.Response(201, json={"id": "123", "observacao": 0, "passos": 0, "encerrado": False})
        elif request.method == "POST":
            # observacao como string em vez de int
            return httpx.Response(200, json={"observacao": "um", "recompensa": -1.0, "terminated": False, "truncated": False})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="http://ambiente")
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=10)

    episodio_id, _ = cliente.criar_episodio()
    with pytest.raises(ErroContrato):
        cliente.passo(episodio_id, 1)
