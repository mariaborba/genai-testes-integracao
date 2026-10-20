import pytest
import httpx
import numpy as np
from fastapi.testclient import TestClient
from ambiente.app import criar_app
from coletor.rollout import (
    ClienteAmbiente, ErroAmbiente, ErroContrato,
    Transicao, Rollout, coletar_rollout
)


# ===== FIXTURES =====

@pytest.fixture
def cliente_http():
    """Novo TestClient para cada teste."""
    app = criar_app()
    return TestClient(app)


@pytest.fixture
def cliente_ambiente(cliente_http):
    """ClienteAmbiente com parâmetros padrão."""
    return ClienteAmbiente(cliente_http, comprimento=5, max_passos=20)


@pytest.fixture
def politica_direita():
    """Política que sempre vai para a direita."""
    return lambda obs: 1


@pytest.fixture
def valor_zero():
    """Função valor que sempre retorna 0."""
    return lambda obs: 0.0


# ===== CRIAR EPISÓDIO =====

def test_criar_episodio_sucesso_valores_validos(cliente_http):
    """POST /episodios com parâmetros válidos retorna 201."""
    resp = cliente_http.post("/episodios", json={
        "comprimento": 5,
        "max_passos": 20
    })
    assert resp.status_code == 201
    data = resp.json()
    assert isinstance(data["id"], str)
    assert data["observacao"] == 0
    assert data["passos"] == 0
    assert data["encerrado"] is False


def test_criar_episodio_comprimento_limite_inferior(cliente_http):
    """POST /episodios com comprimento=2 (mínimo) retorna 201."""
    resp = cliente_http.post("/episodios", json={
        "comprimento": 2,
        "max_passos": 10
    })
    assert resp.status_code == 201


def test_criar_episodio_comprimento_limite_superior(cliente_http):
    """POST /episodios com comprimento=20 (máximo) retorna 201."""
    resp = cliente_http.post("/episodios", json={
        "comprimento": 20,
        "max_passos": 10
    })
    assert resp.status_code == 201


def test_criar_episodio_max_passos_limite_inferior(cliente_http):
    """POST /episodios com max_passos=1 (mínimo) retorna 201."""
    resp = cliente_http.post("/episodios", json={
        "comprimento": 5,
        "max_passos": 1
    })
    assert resp.status_code == 201


def test_criar_episodio_max_passos_limite_superior(cliente_http):
    """POST /episodios com max_passos=500 (máximo) retorna 201."""
    resp = cliente_http.post("/episodios", json={
        "comprimento": 5,
        "max_passos": 500
    })
    assert resp.status_code == 201


def test_criar_episodio_comprimento_abaixo_minimo(cliente_http):
    """POST /episodios com comprimento=1 (< 2) retorna 422."""
    resp = cliente_http.post("/episodios", json={
        "comprimento": 1,
        "max_passos": 10
    })
    assert resp.status_code == 422


def test_criar_episodio_comprimento_acima_maximo(cliente_http):
    """POST /episodios com comprimento=21 (> 20) retorna 422."""
    resp = cliente_http.post("/episodios", json={
        "comprimento": 21,
        "max_passos": 10
    })
    assert resp.status_code == 422


def test_criar_episodio_max_passos_abaixo_minimo(cliente_http):
    """POST /episodios com max_passos=0 (< 1) retorna 422."""
    resp = cliente_http.post("/episodios", json={
        "comprimento": 5,
        "max_passos": 0
    })
    assert resp.status_code == 422


def test_criar_episodio_max_passos_acima_maximo(cliente_http):
    """POST /episodios com max_passos=501 (> 500) retorna 422."""
    resp = cliente_http.post("/episodios", json={
        "comprimento": 5,
        "max_passos": 501
    })
    assert resp.status_code == 422


# ===== CONSULTAR EPISÓDIO =====

def test_consultar_episodio_existente_retorna_200(cliente_http):
    """GET /episodios/{id} para episódio criado retorna 200."""
    resp_criar = cliente_http.post("/episodios", json={
        "comprimento": 5,
        "max_passos": 20
    })
    ep_id = resp_criar.json()["id"]
    
    resp = cliente_http.get(f"/episodios/{ep_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == ep_id
    assert resp.json()["observacao"] == 0


def test_consultar_episodio_inexistente_retorna_404(cliente_http):
    """GET /episodios/{id} para id inexistente retorna 404."""
    resp = cliente_http.get("/episodios/id_inexistente_xyz")
    assert resp.status_code == 404


def test_consultar_episodio_apos_passo_reflete_estado(cliente_http):
    """GET /episodios/{id} após passo mostra estado atualizado."""
    resp_criar = cliente_http.post("/episodios", json={
        "comprimento": 5,
        "max_passos": 20
    })
    ep_id = resp_criar.json()["id"]
    
    cliente_http.post(f"/episodios/{ep_id}/passos", json={"acao": 1})
    
    resp = cliente_http.get(f"/episodios/{ep_id}")
    assert resp.json()["observacao"] == 1
    assert resp.json()["passos"] == 1
    assert resp.json()["encerrado"] is False


# ===== REMOVER EPISÓDIO =====

def test_remover_episodio_existente_retorna_204(cliente_http):
    """DELETE /episodios/{id} para episódio criado retorna 204."""
    resp_criar = cliente_http.post("/episodios", json={
        "comprimento": 5,
        "max_passos": 20
    })
    ep_id = resp_criar.json()["id"]
    
    resp = cliente_http.delete(f"/episodios/{ep_id}")
    assert resp.status_code == 204


def test_remover_episodio_inexistente_retorna_404(cliente_http):
    """DELETE /episodios/{id} para id inexistente retorna 404."""
    resp = cliente_http.delete("/episodios/id_inexistente_xyz")
    assert resp.status_code == 404


def test_remover_episodio_remove_do_servico(cliente_http):
    """Episódio removido não é mais consultável."""
    resp_criar = cliente_http.post("/episodios", json={
        "comprimento": 5,
        "max_passos": 20
    })
    ep_id = resp_criar.json()["id"]
    
    cliente_http.delete(f"/episodios/{ep_id}")
    
    resp_consulta = cliente_http.get(f"/episodios/{ep_id}")
    assert resp_consulta.status_code == 404


def test_remover_episodio_duas_vezes_segunda_retorna_404(cliente_http):
    """DELETE do mesmo episódio duas vezes: primeira 204, segunda 404."""
    resp_criar = cliente_http.post("/episodios", json={
        "comprimento": 5,
        "max_passos": 20
    })
    ep_id = resp_criar.json()["id"]
    
    resp1 = cliente_http.delete(f"/episodios/{ep_id}")
    assert resp1.status_code == 204
    
    resp2 = cliente_http.delete(f"/episodios/{ep_id}")
    assert resp2.status_code == 404


# ===== DAR PASSO =====

def test_dar_passo_acao_direita_avanca_posicao(cliente_http):
    """POST /episodios/{id}/passos com acao=1 move para direita."""
    resp_criar = cliente_http.post("/episodios", json={
        "comprimento": 5,
        "max_passos": 20
    })
    ep_id = resp_criar.json()["id"]
    
    resp = cliente_http.post(f"/episodios/{ep_id}/passos", json={"acao": 1})
    assert resp.status_code == 200
    assert resp.json()["observacao"] == 1
    assert resp.json()["recompensa"] == -1.0
    assert resp.json()["terminated"] is False
    assert resp.json()["truncated"] is False


def test_dar_passo_acao_esquerda_da_zero_mantem_em_zero(cliente_http):
    """POST /episodios/{id}/passos com acao=0 de posição 0 fica em 0."""
    resp_criar = cliente_http.post("/episodios", json={
        "comprimento": 5,
        "max_passos": 20
    })
    ep_id = resp_criar.json()["id"]
    
    resp = cliente_http.post(f"/episodios/{ep_id}/passos", json={"acao": 0})
    assert resp.status_code == 200
    assert resp.json()["observacao"] == 0
    assert resp.json()["recompensa"] == -1.0


def test_dar_passo_acao_esquerda_de_posicao_nao_zero_recua(cliente_http):
    """POST /episodios/{id}/passos com acao=0 de posição > 0 recua."""
    resp_criar = cliente_http.post("/episodios", json={
        "comprimento": 5,
        "max_passos": 20
    })
    ep_id = resp_criar.json()["id"]
    
    # Mover para direita
    cliente_http.post(f"/episodios/{ep_id}/passos", json={"acao": 1})
    
    # Mover para esquerda
    resp = cliente_http.post(f"/episodios/{ep_id}/passos", json={"acao": 0})
    assert resp.json()["observacao"] == 0


def test_dar_passo_acao_negativa_retorna_422(cliente_http):
    """POST /episodios/{id}/passos com acao=-1 retorna 422."""
    resp_criar = cliente_http.post("/episodios", json={
        "comprimento": 5,
        "max_passos": 20
    })
    ep_id = resp_criar.json()["id"]
    
    resp = cliente_http.post(f"/episodios/{ep_id}/passos", json={"acao": -1})
    assert resp.status_code == 422


def test_dar_passo_acao_acima_um_retorna_422(cliente_http):
    """POST /episodios/{id}/passos com acao=2 retorna 422."""
    resp_criar = cliente_http.post("/episodios", json={
        "comprimento": 5,
        "max_passos": 20
    })
    ep_id = resp_criar.json()["id"]
    
    resp = cliente_http.post(f"/episodios/{ep_id}/passos", json={"acao": 2})
    assert resp.status_code == 422


def test_dar_passo_episodio_inexistente_retorna_404(cliente_http):
    """POST /episodios/{id}/passos para id inexistente retorna 404."""
    resp = cliente_http.post("/episodios/id_inexistente_xyz/passos", json={"acao": 1})
    assert resp.status_code == 404


# ===== FLUXO DE EPISÓDIO: TERMINATED =====

def test_fluxo_termina_ao_atingir_comprimento_terminated_true(cliente_http):
    """Episódio termina ao atingir comprimento: terminated=true, reward=+10."""
    resp_criar = cliente_http.post("/episodios", json={
        "comprimento": 2,
        "max_passos": 20
    })
    ep_id = resp_criar.json()["id"]
    
    # Passo 1: 0 -> 1
    resp1 = cliente_http.post(f"/episodios/{ep_id}/passos", json={"acao": 1})
    assert resp1.json()["observacao"] == 1
    assert resp1.json()["terminated"] is False
    
    # Passo 2: 1 -> 2 (atinge comprimento)
    resp2 = cliente_http.post(f"/episodios/{ep_id}/passos", json={"acao": 1})
    assert resp2.status_code == 200
    assert resp2.json()["observacao"] == 2
    assert resp2.json()["recompensa"] == 10.0
    assert resp2.json()["terminated"] is True
    assert resp2.json()["truncated"] is False


def test_fluxo_episodio_encerrado_novo_passo_retorna_409(cliente_http):
    """Passo após terminated: retorna 409."""
    resp_criar = cliente_http.post("/episodios", json={
        "comprimento": 2,
        "max_passos": 20
    })
    ep_id = resp_criar.json()["id"]
    
    # Terminar o episódio
    cliente_http.post(f"/episodios/{ep_id}/passos", json={"acao": 1})
    cliente_http.post(f"/episodios/{ep_id}/passos", json={"acao": 1})
    
    # Tentar passo após terminar
    resp = cliente_http.post(f"/episodios/{ep_id}/passos", json={"acao": 1})
    assert resp.status_code == 409


# ===== FLUXO DE EPISÓDIO: TRUNCATED =====

def test_fluxo_trunca_ao_atingir_max_passos_truncated_true(cliente_http):
    """Episódio trunca ao atingir max_passos: truncated=true."""
    resp_criar = cliente_http.post("/episodios", json={
        "comprimento": 20,
        "max_passos": 2
    })
    ep_id = resp_criar.json()["id"]
    
    # Passo 1
    resp1 = cliente_http.post(f"/episodios/{ep_id}/passos", json={"acao": 1})
    assert resp1.json()["truncated"] is False
    
    # Passo 2 (atinge max_passos)
    resp2 = cliente_http.post(f"/episodios/{ep_id}/passos", json={"acao": 1})
    assert resp2.status_code == 200
    assert resp2.json()["truncated"] is True
    assert resp2.json()["terminated"] is False


def test_fluxo_episodio_truncado_novo_passo_retorna_409(cliente_http):
    """Passo após truncated: retorna 409."""
    resp_criar = cliente_http.post("/episodios", json={
        "comprimento": 20,
        "max_passos": 2
    })
    ep_id = resp_criar.json()["id"]
    
    # Truncar o episódio
    cliente_http.post(f"/episodios/{ep_id}/passos", json={"acao": 1})
    cliente_http.post(f"/episodios/{ep_id}/passos", json={"acao": 1})
    
    # Tentar passo após truncar
    resp = cliente_http.post(f"/episodios/{ep_id}/passos", json={"acao": 1})
    assert resp.status_code == 409


# ===== FLUXO DE EPISÓDIO: TERMINATED E TRUNCATED =====

def test_fluxo_simultaneamente_terminated_e_truncated(cliente_http):
    """Passo com terminated=true e truncated=true simultâneos."""
    resp_criar = cliente_http.post("/episodios", json={
        "comprimento": 2,
        "max_passos": 2
    })
    ep_id = resp_criar.json()["id"]
    
    # Passo 1: 0 -> 1, passos_count=1
    resp1 = cliente_http.post(f"/episodios/{ep_id}/passos", json={"acao": 1})
    assert resp1.json()["terminated"] is False
    assert resp1.json()["truncated"] is False
    
    # Passo 2: 1 -> 2, passos_count=2 (atingiu max_passos E objetivo)
    resp2 = cliente_http.post(f"/episodios/{ep_id}/passos", json={"acao": 1})
    assert resp2.status_code == 200
    assert resp2.json()["observacao"] == 2
    assert resp2.json()["recompensa"] == 10.0
    assert resp2.json()["terminated"] is True
    assert resp2.json()["truncated"] is True


# ===== CLIENTE AMBIENTE =====

def test_cliente_criar_episodio_retorna_id_e_obs(cliente_http):
    """ClienteAmbiente.criar_episodio() retorna (id, observacao_inicial)."""
    cliente = ClienteAmbiente(cliente_http, comprimento=5, max_passos=20)
    ep_id, obs = cliente.criar_episodio()
    
    assert isinstance(ep_id, str)
    assert len(ep_id) > 0
    assert obs == 0


def test_cliente_passo_retorna_transicao(cliente_http):
    """ClienteAmbiente.passo() retorna Transicao."""
    cliente = ClienteAmbiente(cliente_http, comprimento=5, max_passos=20)
    ep_id, _ = cliente.criar_episodio()
    
    transicao = cliente.passo(ep_id, 1)
    
    assert isinstance(transicao, Transicao)
    assert transicao.observacao == 1
    assert transicao.recompensa == -1.0
    assert transicao.terminated is False
    assert transicao.truncated is False


def test_cliente_passo_episodio_inexistente_lanca_erro_ambiente_404(cliente_http):
    """ClienteAmbiente.passo() com id inexistente lança ErroAmbiente status=404."""
    cliente = ClienteAmbiente(cliente_http, comprimento=5, max_passos=20)
    
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.passo("id_inexistente_xyz", 1)
    
    assert exc_info.value.status == 404


def test_cliente_passo_episodio_encerrado_lanca_erro_ambiente_409(cliente_http):
    """ClienteAmbiente.passo() após terminated lança ErroAmbiente status=409."""
    cliente = ClienteAmbiente(cliente_http, comprimento=2, max_passos=20)
    ep_id, _ = cliente.criar_episodio()
    
    cliente.passo(ep_id, 1)  # 0 -> 1
    cliente.passo(ep_id, 1)  # 1 -> 2, terminated
    
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.passo(ep_id, 1)
    
    assert exc_info.value.status == 409


def test_cliente_passo_episodio_truncado_lanca_erro_ambiente_409(cliente_http):
    """ClienteAmbiente.passo() após truncated lança ErroAmbiente status=409."""
    cliente = ClienteAmbiente(cliente_http, comprimento=20, max_passos=2)
    ep_id, _ = cliente.criar_episodio()
    
    cliente.passo(ep_id, 1)  # 0 -> 1
    cliente.passo(ep_id, 1)  # 1 -> 2, truncated
    
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.passo(ep_id, 1)
    
    assert exc_info.value.status == 409


def test_cliente_encerrar_episodio(cliente_http):
    """ClienteAmbiente.encerrar() remove episódio."""
    cliente = ClienteAmbiente(cliente_http, comprimento=5, max_passos=20)
    ep_id, _ = cliente.criar_episodio()
    
    cliente.encerrar(ep_id)
    
    # Verificar remoção
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.passo(ep_id, 1)
    
    assert exc_info.value.status == 404


def test_cliente_encerrar_inexistente_nao_lanca_erro(cliente_http):
    """ClienteAmbiente.encerrar() de id inexistente não lança erro."""
    cliente = ClienteAmbiente(cliente_http, comprimento=5, max_passos=20)
    
    # Não deve lançar
    cliente.encerrar("id_inexistente_xyz")


# ===== COLETOR: ESTRUTURA BÁSICA =====

def test_coletor_retorna_rollout_com_n_passos(cliente_http, politica_direita, valor_zero):
    """coletar_rollout() retorna Rollout com comprimento n_passos."""
    cliente = ClienteAmbiente(cliente_http, comprimento=20, max_passos=100)
    
    rollout = coletar_rollout(
        cliente,
        politica=politica_direita,
        valor=valor_zero,
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    assert isinstance(rollout, Rollout)
    assert len(rollout.observacoes) == 5
    assert len(rollout.acoes) == 5
    assert len(rollout.recompensas) == 5
    assert len(rollout.values) == 5
    assert len(rollout.next_values) == 5
    assert len(rollout.terminated) == 5
    assert len(rollout.truncated) == 5
    assert len(rollout.advantages) == 5
    assert len(rollout.returns) == 5


def test_coletor_dtypes_corretos(cliente_http, politica_direita, valor_zero):
    """Rollout tem dtypes corretos."""
    cliente = ClienteAmbiente(cliente_http, comprimento=20, max_passos=100)
    
    rollout = coletar_rollout(
        cliente,
        politica=politica_direita,
        valor=valor_zero,
        n_passos=3,
        gamma=0.99,
        gae_lambda=0.95
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


# ===== COLETOR: VALORES LITERAIS =====

def test_coletor_valores_literais_episodio_completo_sem_reinicio(cliente_http):
    """Coleta episódio inteiro (termina antes do fim): verifica valores literais."""
    # comprimento=2, max_passos=100
    # Passo 0: obs=0, acao=1, obs_nova=1, reward=-1, terminated=False, truncated=False
    # Passo 1: obs=1, acao=1, obs_nova=2, reward=+10, terminated=True, truncated=False
    politica = lambda obs: 1
    valor = lambda obs: 0.0
    
    cliente = ClienteAmbiente(cliente_http, comprimento=2, max_passos=100)
    
    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=valor,
        n_passos=2,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    assert np.array_equal(rollout.observacoes, np.array([0, 1], dtype=np.int64))
    assert np.array_equal(rollout.acoes, np.array([1, 1], dtype=np.int64))
    assert np.array_equal(rollout.recompensas, np.array([-1.0, 10.0], dtype=np.float64))
    assert np.array_equal(rollout.terminated, np.array([False, True], dtype=bool))
    assert np.array_equal(rollout.truncated, np.array([False, False], dtype=bool))


def test_coletor_next_values_vem_da_funcao_valor(cliente_http):
    """next_values[t] = valor(observacao_nova_do_passo_t)."""
    # Função valor: obs -> obs/10
    valor = lambda obs: obs / 10.0
    politica = lambda obs: 1
    
    cliente = ClienteAmbiente(cliente_http, comprimento=20, max_passos=100)
    
    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=valor,
        n_passos=3,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    # Passo 0: obs_nova=1, next_values[0]=0.1
    # Passo 1: obs_nova=2, next_values[1]=0.2
    # Passo 2: obs_nova=3, next_values[2]=0.3
    assert np.isclose(rollout.next_values[0], 0.1)
    assert np.isclose(rollout.next_values[1], 0.2)
    assert np.isclose(rollout.next_values[2], 0.3)


def test_coletor_values_vem_da_funcao_valor(cliente_http):
    """values[t] = valor(observacao_inicial_do_passo_t)."""
    valor = lambda obs: obs / 10.0
    politica = lambda obs: 1
    
    cliente = ClienteAmbiente(cliente_http, comprimento=20, max_passos=100)
    
    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=valor,
        n_passos=3,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    # Passo 0: obs=0, values[0]=0.0
    # Passo 1: obs=1, values[1]=0.1
    # Passo 2: obs=2, values[2]=0.2
    assert np.isclose(rollout.values[0], 0.0)
    assert np.isclose(rollout.values[1], 0.1)
    assert np.isclose(rollout.values[2], 0.2)


# ===== COLETOR: REINICIALIZAÇÃO DE EPISÓDIO =====

def test_coletor_reinicia_apos_terminated_dentro_do_rollout(cliente_http):
    """Termination no meio do rollout: episódio é reiniciado. Passo final termina em passo 2."""
    politica = lambda obs: 1
    valor = lambda obs: 0.0
    
    cliente = ClienteAmbiente(cliente_http, comprimento=3, max_passos=100)
    
    # Passo 0: 0->1
    # Passo 1: 1->2
    # Passo 2: 2->3, terminated=True (fim episódio 1)
    # Passo 3: novo episódio, 0->1
    # Passo 4: 1->2, truncado pelo rollout
    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=valor,
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    assert rollout.terminated[2] is True
    assert rollout.observacoes[2] == 2  # observação final do episódio 1
    assert rollout.observacoes[3] == 0  # observação inicial do episódio 2
    assert rollout.truncated[4] is True  # último passo truncado pelo rollout


def test_coletor_reinicia_apos_truncated_dentro_do_rollout(cliente_http):
    """Truncation no meio do rollout: episódio é reiniciado. Passo final trunca em passo 2."""
    politica = lambda obs: 1
    valor = lambda obs: 0.0
    
    cliente = ClienteAmbiente(cliente_http, comprimento=20, max_passos=3)
    
    # Passo 0: 0->1
    # Passo 1: 1->2
    # Passo 2: 2->3, truncated=True (fim episódio 1)
    # Passo 3: novo episódio, 0->1
    # Passo 4: 1->2, truncado pelo rollout
    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=valor,
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    assert rollout.truncated[2] is True
    assert rollout.terminated[2] is False
    assert rollout.observacoes[3] == 0  # novo episódio reinicia em 0


def test_coletor_ultimo_passo_truncado_se_nao_terminou(cliente_http):
    """Último passo do rollout é marcado truncated=true se não terminou."""
    politica = lambda obs: 1
    valor = lambda obs: 0.0
    
    cliente = ClienteAmbiente(cliente_http, comprimento=100, max_passos=500)
    
    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=valor,
        n_passos=3,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    # Último passo truncado por fim do rollout
    assert rollout.truncated[2] is True
    assert rollout.terminated[2] is False


def test_coletor_ultimo_passo_terminated_se_terminou_naturalmente(cliente_http):
    """Se episódio termina exatamente no último passo, terminated=true."""
    politica = lambda obs: 1
    valor = lambda obs: 0.0
    
    cliente = ClienteAmbiente(cliente_http, comprimento=2, max_passos=100)
    
    # Passo 0: 0->1
    # Passo 1: 1->2, terminated=True (exatamente no fim)
    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=valor,
        n_passos=2,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    assert rollout.terminated[1] is True
    assert rollout.truncated[1] is False


# ===== COLETOR: CASOS DE ERRO =====

def test_coletor_n_passos_zero_lanca_valor_error(cliente_http, politica_direita, valor_zero):
    """coletar_rollout() com n_passos=0 lança ValueError."""
    cliente = ClienteAmbiente(cliente_http, comprimento=5, max_passos=20)
    
    with pytest.raises(ValueError):
        coletar_rollout(
            cliente,
            politica=politica_direita,
            valor=valor_zero,
            n_passos=0,
            gamma=0.99,
            gae_lambda=0.95
        )


def test_coletor_n_passos_negativo_lanca_valor_error(cliente_http, politica_direita, valor_zero):
    """coletar_rollout() com n_passos<0 lança ValueError."""
    cliente = ClienteAmbiente(cliente_http, comprimento=5, max_passos=20)
    
    with pytest.raises(ValueError):
        coletar_rollout(
            cliente,
            politica=politica_direita,
            valor=valor_zero,
            n_passos=-1,
            gamma=0.99,
            gae_lambda=0.95
        )


def test_coletor_n_passos_nao_inteiro_lanca_valor_error(cliente_http, politica_direita, valor_zero):
    """coletar_rollout() com n_passos não inteiro lança ValueError."""
    cliente = ClienteAmbiente(cliente_http, comprimento=5, max_passos=20)
    
    with pytest.raises(ValueError):
        coletar_rollout(
            cliente,
            politica=politica_direita,
            valor=valor_zero,
            n_passos=2.5,
            gamma=0.99,
            gae_lambda=0.95
        )


# ===== ERRO DE REDE E CONTRATO =====

def test_erro_http_500_capturado_como_erro_ambiente(politica_direita, valor_zero):
    """HTTP 500 capturado como ErroAmbiente com status=500."""
    def handler(request):
        return httpx.Response(500, text="Internal Server Error")
    
    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://ambiente"
    )
    
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=20)
    
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.criar_episodio()
    
    assert exc_info.value.status == 500


def test_erro_timeout_capturado_como_erro_ambiente():
    """Timeout capturado como ErroAmbiente com status=None."""
    def handler(request):
        raise httpx.ReadTimeout("tempo esgotado", request=request)
    
    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://ambiente"
    )
    
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=20)
    
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.criar_episodio()
    
    assert exc_info.value.status is None


def test_erro_conexao_capturado_como_erro_ambiente():
    """Falha de conexão capturada como ErroAmbiente com status=None."""
    def handler(request):
        raise httpx.ConnectError("recusada", request=request)
    
    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://ambiente"
    )
    
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=20)
    
    with pytest.raises(ErroAmbiente) as exc_info:
        cliente.criar_episodio()
    
    assert exc_info.value.status is None


def test_erro_contrato_campo_tipo_invalido_criar_episodio():
    """Resposta 201 com campo de tipo inválido lança ErroContrato."""
    def handler(request):
        # id deve ser string, não int
        return httpx.Response(201, json={
            "id": 12345,
            "observacao": 0,
            "passos": 0,
            "encerrado": False
        })
    
    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://ambiente"
    )
    
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=20)
    
    with pytest.raises(ErroContrato):
        cliente.criar_episodio()


def test_erro_contrato_campo_faltando_em_transicao():
    """Resposta 200 de passo sem campo obrigatório lança ErroContrato."""
    count = 0
    
    def handler(request):
        nonlocal count
        if "passos" in request.url.path:
            # Faltam campos: terminated, truncated
            return httpx.Response(200, json={
                "observacao": 1,
                "recompensa": -1.0
            })
        # POST criar episódio
        count += 1
        return httpx.Response(201, json={
            "id": f"ep{count}",
            "observacao": 0,
            "passos": 0,
            "encerrado": False
        })
    
    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://ambiente"
    )
    
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=20)
    ep_id, _ = cliente.criar_episodio()
    
    with pytest.raises(ErroContrato):
        cliente.passo(ep_id, 1)


def test_erro_contrato_tipo_invalido_em_campo_numerico():
    """Campo numérico com tipo errado lança ErroContrato."""
    count = 0
    
    def handler(request):
        nonlocal count
        if "passos" in request.url.path:
            return httpx.Response(200, json={
                "observacao": "não é int",
                "recompensa": -1.0,
                "terminated": False,
                "truncated": False
            })
        count += 1
        return httpx.Response(201, json={
            "id": f"ep{count}",
            "observacao": 0,
            "passos": 0,
            "encerrado": False
        })
    
    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://ambiente"
    )
    
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=20)
    ep_id, _ = cliente.criar_episodio()
    
    with pytest.raises(ErroContrato):
        cliente.passo(ep_id, 1)


def test_erro_contrato_boolean_esperado_int_recebido():
    """Campo booleano com int no lugar lança ErroContrato."""
    count = 0
    
    def handler(request):
        nonlocal count
        if "passos" in request.url.path:
            return httpx.Response(200, json={
                "observacao": 1,
                "recompensa": -1.0,
                "terminated": 0,  # deve ser bool
                "truncated": False
            })
        count += 1
        return httpx.Response(201, json={
            "id": f"ep{count}",
            "observacao": 0,
            "passos": 0,
            "encerrado": False
        })
    
    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://ambiente"
    )
    
    cliente = ClienteAmbiente(http, comprimento=5, max_passos=20)
    ep_id, _ = cliente.criar_episodio()
    
    with pytest.raises(ErroContrato):
        cliente.passo(ep_id, 1)


# ===== INTEGRAÇÕES COMPLEXAS =====

def test_fluxo_completo_multiplos_episodios_e_reiniciacao(cliente_http):
    """Fluxo com múltiplos episódios dentro de um rollout."""
    # Faz 6 passos, episódio termina em 2
    # Esperado: 3 episódios iniciados, 2 deles terminam dentro do rollout
    politica = lambda obs: 1
    valor = lambda obs: 0.0
    
    cliente = ClienteAmbiente(cliente_http, comprimento=2, max_passos=100)
    
    # Episódio 1: passos 0-1 (termina em passo 1)
    # Episódio 2: passos 2-3 (termina em passo 3)
    # Episódio 3: passos 4-5 (truncado em passo 5)
    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=valor,
        n_passos=6,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    assert rollout.terminated[1] is True
    assert rollout.observacoes[2] == 0  # novo episódio
    assert rollout.terminated[3] is True
    assert rollout.observacoes[4] == 0  # novo episódio
    assert rollout.truncated[5] is True


def test_fluxo_politica_adaptativa_muda_direcao(cliente_http):
    """Política que muda de direção baseado na observação."""
    # Se obs < 2, vai direita; se obs >= 2, vai esquerda
    politica = lambda obs: 1 if obs < 2 else 0
    valor = lambda obs: 0.0
    
    cliente = ClienteAmbiente(cliente_http, comprimento=5, max_passos=50)
    
    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=valor,
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    # Passo 0: obs=0, acao=1, obs_nova=1
    # Passo 1: obs=1, acao=1, obs_nova=2
    # Passo 2: obs=2, acao=0, obs_nova=1 (volta)
    # ...
    assert rollout.acoes[0] == 1
    assert rollout.acoes[1] == 1
    assert rollout.acoes[2] == 0


def test_fluxo_com_valores_nao_zero(cliente_http):
    """Coletor funciona com função valor que retorna valores não-zero."""
    valor = lambda obs: obs * 2.0  # obs * 2
    politica = lambda obs: 1
    
    cliente = ClienteAmbiente(cliente_http, comprimento=5, max_passos=50)
    
    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=valor,
        n_passos=3,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    # Verificar que values são não-zero
    assert rollout.values[0] == 0.0  # obs=0
    assert rollout.values[1] == 2.0  # obs=1
    assert rollout.values[2] == 4.0  # obs=2


def test_fluxo_gae_calcula_advantages_e_returns(cliente_http):
    """Rollout calcula advantages e returns via GAE."""
    valor = lambda obs: 0.0
    politica = lambda obs: 1
    
    cliente = ClienteAmbiente(cliente_http, comprimento=5, max_passos=50)
    
    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=valor,
        n_passos=2,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    # Deve ter calculado advantages e returns
    assert len(rollout.advantages) == 2
    assert len(rollout.returns) == 2
    assert rollout.advantages.dtype == np.float64
    assert rollout.returns.dtype == np.float64
    # returns = advantages + values
    assert np.allclose(
        rollout.returns,
        rollout.advantages + rollout.values
    )
