rollout_valor_funcao_diferentes_obs(http_client):
    """Valor diferente para cada observação."""
    cliente = ClienteAmbiente(http_client, comprimento=100, max_passos=100)
    
    valor = lambda obs: float(obs ** 2)
    
    rollout = coletar_rollout(
        cliente,
        politica=lambda obs: 1,
        valor=valor,
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    np.testing.assert_array_equal(rollout.values, [0.0, 1.0, 4.0, 9.0, 16.0])
    np.testing.assert_array_equal(rollout.next_values, [1.0, 4.0, 9.0, 16.0, 25.0])


# ============================================================================
# Testes do Coletor - Políticas Variadas
# ============================================================================

def test_coletar_rollout_politica_acao_0(http_client):
    """Coletor funciona com política que sempre escolhe ação 0."""
    cliente = ClienteAmbiente(http_client, comprimento=100, max_passos=100)
    
    rollout = coletar_rollout(
        cliente,
        politica=lambda obs: 0,
        valor=lambda obs: float(obs),
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    np.testing.assert_array_equal(rollout.observacoes, [0, 0, 0, 0, 0])
    np.testing.assert_array_equal(rollout.acoes, [0, 0, 0, 0, 0])


def test_coletar_rollout_politica_dependente_obs(http_client):
    """Coletor funciona com política que depende da observação."""
    cliente = ClienteAmbiente(http_client, comprimento=100, max_passos=100)
    
    def politica(obs):
        return 1 if obs % 2 == 0 else 0
    
    rollout = coletar_rollout(
        cliente,
        politica=politica,
        valor=lambda obs: float(obs),
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    np.testing.assert_array_equal(rollout.observacoes, [0, 1, 0, 1, 0])
    np.testing.assert_array_equal(rollout.acoes, [1, 0, 1, 0, 1])


def test_coletar_rollout_multiplos_episodios(http_client):
    """Coleta com múltiplos episódios sucessivos."""
    cliente = ClienteAmbiente(http_client, comprimento=2, max_passos=100)
    
    rollout = coletar_rollout(
        cliente,
        politica=lambda obs: 1,
        valor=lambda obs: float(obs),
        n_passos=10,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    assert len(rollout.observacoes) == 10
    assert any(rollout.terminated)


# ============================================================================
# Testes do Coletor - Parâmetros de GAE
# ============================================================================

def test_coletar_rollout_gamma_1_0(http_client):
    """Coletor funciona com gamma=1.0."""
    cliente = ClienteAmbiente(http_client, comprimento=100, max_passos=100)
    
    rollout = coletar_rollout(
        cliente,
        politica=lambda obs: 1,
        valor=lambda obs: 0.0,
        n_passos=5,
        gamma=1.0,
        gae_lambda=0.95
    )
    
    assert rollout.advantages.shape == (5,)


def test_coletar_rollout_gamma_0_0(http_client):
    """Coletor funciona com gamma=0.0."""
    cliente = ClienteAmbiente(http_client, comprimento=100, max_passos=100)
    
    rollout = coletar_rollout(
        cliente,
        politica=lambda obs: 1,
        valor=lambda obs: 0.0,
        n_passos=5,
        gamma=0.0,
        gae_lambda=0.95
    )
    
    assert rollout.advantages.shape == (5,)


def test_coletar_rollout_gae_lambda_0_0(http_client):
    """Coletor funciona com gae_lambda=0.0 (TD error)."""
    cliente = ClienteAmbiente(http_client, comprimento=100, max_passos=100)
    
    rollout = coletar_rollout(
        cliente,
        politica=lambda obs: 1,
        valor=lambda obs: 0.0,
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.0
    )
    
    assert rollout.advantages.shape == (5,)


def test_coletar_rollout_gae_lambda_1_0(http_client):
    """Coletor funciona com gae_lambda=1.0 (Monte Carlo)."""
    cliente = ClienteAmbiente(http_client, comprimento=100, max_passos=100)
    
    rollout = coletar_rollout(
        cliente,
        politica=lambda obs: 1,
        valor=lambda obs: 0.0,
        n_passos=5,
        gamma=0.99,
        gae_lambda=1.0
    )
    
    assert rollout.advantages.shape == (5,)


# ============================================================================
# Testes do Coletor - Returns e Advantages
# ============================================================================

def test_coletar_rollout_returns_iguais_advantages_plus_values(http_client):
    """returns = advantages + values."""
    cliente = ClienteAmbiente(http_client, comprimento=100, max_passos=100)
    
    rollout = coletar_rollout(
        cliente,
        politica=lambda obs: 1,
        valor=lambda obs: 0.0,
        n_passos=5,
        gamma=0.99,
        gae_lambda=1.0
    )
    
    expected = rollout.advantages + rollout.values
    np.testing.assert_array_almost_equal(rollout.returns, expected)


def test_coletar_rollout_recompensas_corretas(http_client):
    """Recompensas: -1.0 por passo, +10.0 ao atingir comprimento."""
    cliente = ClienteAmbiente(http_client, comprimento=3, max_passos=100)
    
    rollout = coletar_rollout(
        cliente,
        politica=lambda obs: 1,
        valor=lambda obs: float(obs),
        n_passos=4,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    # Passos 0-1: -1; Passo 2: termina em 3, recompensa +10; Passo 3: novo episódio -1
    assert rollout.recompensas.tolist()[0] == -1.0
    assert rollout.recompensas.tolist()[1] == -1.0
    assert rollout.recompensas.tolist()[2] == 10.0


# ============================================================================
# Testes do Coletor - Estados Encerrados
# ============================================================================

def test_coletar_rollout_terminated_true_truncated_false_normal(http_client):
    """Episódio que termina naturalmente: terminated=true, truncated=false."""
    cliente = ClienteAmbiente(http_client, comprimento=3, max_passos=100)
    
    rollout = coletar_rollout(
        cliente,
        politica=lambda obs: 1,
        valor=lambda obs: float(obs),
        n_passos=3,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    assert rollout.terminated.tolist()[2] is True
    assert rollout.truncated.tolist()[2] is False


def test_coletar_rollout_terminated_false_truncated_true_pelo_limite(http_client):
    """Episódio truncado por limite: terminated=false, truncated=true."""
    cliente = ClienteAmbiente(http_client, comprimento=100, max_passos=3)
    
    rollout = coletar_rollout(
        cliente,
        politica=lambda obs: 1,
        valor=lambda obs: float(obs),
        n_passos=4,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    assert rollout.terminated.tolist()[2] is False
    assert rollout.truncated.tolist()[2] is True


def test_coletar_rollout_terminated_true_truncated_true_ambos(http_client):
    """Episódio que termina E trunca simultaneamente."""
    cliente = ClienteAmbiente(http_client, comprimento=3, max_passos=3)
    
    rollout = coletar_rollout(
        cliente,
        politica=lambda obs: 1,
        valor=lambda obs: float(obs),
        n_passos=5,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    found_both = False
    for i in range(len(rollout.terminated)):
        if rollout.terminated[i] and rollout.truncated[i]:
            found_both = True
            break
    assert found_both


# ============================================================================
# Testes de Integração Completa
# ============================================================================

def test_integracao_criar_passo_consultar(http_client):
    """Fluxo: criar -> passo -> passo -> consultar."""
    resp_criar = http_client.post("/episodios", json={
        "comprimento": 5,
        "max_passos": 20
    })
    ep_id = resp_criar.json()["id"]
    
    # Dois passos
    http_client.post(f"/episodios/{ep_id}/passos", json={"acao": 1})
    http_client.post(f"/episodios/{ep_id}/passos", json={"acao": 1})
    
    # Consultar
    resp = http_client.get(f"/episodios/{ep_id}")
    assert resp.status_code == 200
    assert resp.json()["observacao"] == 2
    assert resp.json()["passos"] == 2
    assert resp.json()["encerrado"] is False


def test_integracao_criar_passo_ate_terminal_tentar_novo_passo(http_client):
    """Fluxo: criar -> passos até terminal -> tentar novo passo (409)."""
    resp_criar = http_client.post("/episodios", json={
        "comprimento": 2,
        "max_passos": 20
    })
    ep_id = resp_criar.json()["id"]
    
    # Dois passos terminam
    http_client.post(f"/episodios/{ep_id}/passos", json={"acao": 1})
    resp_term = http_client.post(f"/episodios/{ep_id}/passos", json={"acao": 1})
    assert resp_term.json()["terminated"] is True
    
    # Tentar novo
    resp_novo = http_client.post(f"/episodios/{ep_id}/passos", json={"acao": 1})
    assert resp_novo.status_code == 409


def test_integracao_criar_remover_consultar_retorna_404(http_client):
    """Fluxo: criar -> remover -> consultar (404)."""
    resp_criar = http_client.post("/episodios", json={
        "comprimento": 5,
        "max_passos": 20
    })
    ep_id = resp_criar.json()["id"]
    
    http_client.delete(f"/episodios/{ep_id}")
    
    resp = http_client.get(f"/episodios/{ep_id}")
    assert resp.status_code == 404


def test_integracao_cliente_ambiente_criar_passo_encerrar(http_client):
    """ClienteAmbiente: criar_episodio -> passo -> encerrar."""
    cliente = ClienteAmbiente(http_client, comprimento=5, max_passos=20)
    
    ep_id, obs_ini = cliente.criar_episodio()
    assert obs_ini == 0
    
    trans = cliente.passo(ep_id, 1)
    assert isinstance(trans, Transicao)
    assert trans.observacao == 1
    assert trans.recompensa == -1.0
    
    cliente.encerrar(ep_id)
    
    resp = http_client.get(f"/episodios/{ep_id}")
    assert resp.status_code == 404


def test_integracao_cliente_ambiente_episodio_termina(http_client):
    """ClienteAmbiente com episódio que termina."""
    cliente = ClienteAmbiente(http_client, comprimento=2, max_passos=20)
    
    ep_id, _ = cliente.criar_episodio()
    
    trans1 = cliente.passo(ep_id, 1)
    assert trans1.terminated is False
    
    trans2 = cliente.passo(ep_id, 1)
    assert trans2.terminated is True
    assert trans2.recompensa == 10.0


def test_integracao_cliente_ambiente_episodio_trunca(http_client):
    """ClienteAmbiente com episódio que trunca."""
    cliente = ClienteAmbiente(http_client, comprimento=100, max_passos=2)
    
    ep_id, _ = cliente.criar_episodio()
    
    trans1 = cliente.passo(ep_id, 1)
    assert trans1.truncated is False
    
    trans2 = cliente.passo(ep_id, 1)
    assert trans2.truncated is True
```