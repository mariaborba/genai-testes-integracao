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
    Transicao,
    Rollout,
    coletar_rollout,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def app_client():
    """Fixture: novo TestClient para cada teste."""
    app = criar_app()
    return TestClient(app)


# ==============================================================================
# TESTES DA API REST - POST /episodios
# ==============================================================================


class TestCriarEpisodio:
    """Testes para criação de episódios."""

    def test_criar_episodio_valido_retorna_201(self, app_client):
        """POST /episodios com parâmetros válidos retorna 201."""
        response = app_client.post(
            "/episodios", json={"comprimento": 5, "max_passos": 10}
        )
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["observacao"] == 0
        assert data["passos"] == 0
        assert data["encerrado"] is False

    def test_criar_episodio_comprimento_minimo_valido(self, app_client):
        """POST /episodios com comprimento=2 (mínimo) retorna 201."""
        response = app_client.post(
            "/episodios", json={"comprimento": 2, "max_passos": 1}
        )
        assert response.status_code == 201

    def test_criar_episodio_comprimento_maximo_valido(self, app_client):
        """POST /episodios com comprimento=20 (máximo) retorna 201."""
        response = app_client.post(
            "/episodios", json={"comprimento": 20, "max_passos": 500}
        )
        assert response.status_code == 201

    def test_criar_episodio_comprimento_abaixo_minimo_retorna_422(self, app_client):
        """POST /episodios com comprimento<2 retorna 422."""
        response = app_client.post(
            "/episodios", json={"comprimento": 1, "max_passos": 1}
        )
        assert response.status_code == 422

    def test_criar_episodio_comprimento_acima_maximo_retorna_422(self, app_client):
        """POST /episodios com comprimento>20 retorna 422."""
        response = app_client.post(
            "/episodios", json={"comprimento": 21, "max_passos": 1}
        )
        assert response.status_code == 422

    def test_criar_episodio_max_passos_abaixo_minimo_retorna_422(self, app_client):
        """POST /episodios com max_passos<1 retorna 422."""
        response = app_client.post(
            "/episodios", json={"comprimento": 5, "max_passos": 0}
        )
        assert response.status_code == 422

    def test_criar_episodio_max_passos_acima_maximo_retorna_422(self, app_client):
        """POST /episodios com max_passos>500 retorna 422."""
        response = app_client.post(
            "/episodios", json={"comprimento": 5, "max_passos": 501}
        )
        assert response.status_code == 422

    def test_criar_episodio_max_passos_minimo_valido(self, app_client):
        """POST /episodios com max_passos=1 (mínimo) retorna 201."""
        response = app_client.post(
            "/episodios", json={"comprimento": 2, "max_passos": 1}
        )
        assert response.status_code == 201

    def test_criar_episodio_max_passos_maximo_valido(self, app_client):
        """POST /episodios com max_passos=500 (máximo) retorna 201."""
        response = app_client.post(
            "/episodios", json={"comprimento": 2, "max_passos": 500}
        )
        assert response.status_code == 201


# ==============================================================================
# TESTES DA API REST - GET /episodios/{episodio_id}
# ==============================================================================


class TestConsultarEpisodio:
    """Testes para consulta de episódios."""

    def test_consultar_episodio_existente_retorna_200(self, app_client):
        """GET /episodios/{id} existente retorna 200 com estado."""
        create = app_client.post("/episodios", json={"comprimento": 5, "max_passos": 10})
        episodio_id = create.json()["id"]

        response = app_client.get(f"/episodios/{episodio_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == episodio_id
        assert data["observacao"] == 0
        assert data["passos"] == 0
        assert data["encerrado"] is False

    def test_consultar_episodio_inexistente_retorna_404(self, app_client):
        """GET /episodios/{id} inexistente retorna 404."""
        response = app_client.get("/episodios/nao-existe")
        assert response.status_code == 404


# ==============================================================================
# TESTES DA API REST - DELETE /episodios/{episodio_id}
# ==============================================================================


class TestRemoverEpisodio:
    """Testes para remoção de episódios."""

    def test_remover_episodio_existente_retorna_204(self, app_client):
        """DELETE /episodios/{id} existente retorna 204."""
        create = app_client.post("/episodios", json={"comprimento": 5, "max_passos": 10})
        episodio_id = create.json()["id"]

        response = app_client.delete(f"/episodios/{episodio_id}")
        assert response.status_code == 204

        # Confirma remoção
        assert app_client.get(f"/episodios/{episodio_id}").status_code == 404

    def test_remover_episodio_inexistente_retorna_404(self, app_client):
        """DELETE /episodios/{id} inexistente retorna 404."""
        response = app_client.delete("/episodios/nao-existe")
        assert response.status_code == 404


# ==============================================================================
# TESTES DA API REST - POST /episodios/{episodio_id}/passos
# ==============================================================================


class TestDarPasso:
    """Testes para dar passos em episódios."""

    def test_dar_passo_acao_valida_retorna_200(self, app_client):
        """POST /episodios/{id}/passos com ação válida retorna 200."""
        create = app_client.post("/episodios", json={"comprimento": 5, "max_passos": 10})
        episodio_id = create.json()["id"]

        response = app_client.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["observacao"], int)
        assert isinstance(data["recompensa"], (int, float))
        assert isinstance(data["terminated"], bool)
        assert isinstance(data["truncated"], bool)

    def test_dar_passo_acao_zero_valida(self, app_client):
        """POST /episodios/{id}/passos com ação=0 retorna 200."""
        create = app_client.post("/episodios", json={"comprimento": 5, "max_passos": 10})
        episodio_id = create.json()["id"]

        response = app_client.post(f"/episodios/{episodio_id}/passos", json={"acao": 0})
        assert response.status_code == 200

    def test_dar_passo_episodio_inexistente_retorna_404(self, app_client):
        """POST /episodios/{id}/passos em episódio inexistente retorna 404."""
        response = app_client.post("/episodios/nao-existe/passos", json={"acao": 1})
        assert response.status_code == 404

    def test_dar_passo_acao_invalida_maior_que_1_retorna_422(self, app_client):
        """POST /episodios/{id}/passos com ação>1 retorna 422."""
        create = app_client.post("/episodios", json={"comprimento": 5, "max_passos": 10})
        episodio_id = create.json()["id"]

        response = app_client.post(f"/episodios/{episodio_id}/passos", json={"acao": 2})
        assert response.status_code == 422

    def test_dar_passo_acao_invalida_negativa_retorna_422(self, app_client):
        """POST /episodios/{id}/passos com ação<0 retorna 422."""
        create = app_client.post("/episodios", json={"comprimento": 5, "max_passos": 10})
        episodio_id = create.json()["id"]

        response = app_client.post(f"/episodios/{episodio_id}/passos", json={"acao": -1})
        assert response.status_code == 422


# ==============================================================================
# TESTES DAS REGRAS DE MECÂNICA
# ==============================================================================


class TestMecanicaAgente:
    """Testes das regras de movimento e recompensa do agente."""

    def test_agente_comeca_em_posicao_zero(self, app_client):
        """Estado inicial de novo episódio tem observacao=0."""
        response = app_client.post("/episodios", json={"comprimento": 5, "max_passos": 10})
        assert response.json()["observacao"] == 0

    def test_acao_direita_incrementa_posicao(self, app_client):
        """Ação 1 incrementa a posição."""
        create = app_client.post("/episodios", json={"comprimento": 5, "max_passos": 10})
        episodio_id = create.json()["id"]

        step = app_client.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})
        assert step.json()["observacao"] == 1

    def test_acao_esquerda_decrementa_posicao(self, app_client):
        """Ação 0 decrementa a posição."""
        create = app_client.post("/episodios", json={"comprimento": 5, "max_passos": 10})
        episodio_id = create.json()["id"]

        # Vai para 1
        app_client.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})
        # Volta para 0
        step = app_client.post(f"/episodios/{episodio_id}/passos", json={"acao": 0})
        assert step.json()["observacao"] == 0

    def test_acao_esquerda_em_zero_mantem_zero(self, app_client):
        """Ação 0 em posição 0 não decrementa (limite esquerdo)."""
        create = app_client.post("/episodios", json={"comprimento": 5, "max_passos": 10})
        episodio_id = create.json()["id"]

        step = app_client.post(f"/episodios/{episodio_id}/passos", json={"acao": 0})
        assert step.json()["observacao"] == 0

    def test_passo_normal_recompensa_negativa(self, app_client):
        """Cada passo normal (não objetivo) dá recompensa=-1.0."""
        create = app_client.post("/episodios", json={"comprimento": 5, "max_passos": 10})
        episodio_id = create.json()["id"]

        step = app_client.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})
        assert step.json()["recompensa"] == -1.0

    def test_chegar_ao_objetivo_recompensa_positiva(self, app_client):
        """Passo que chega ao objetivo dá recompensa=+10.0."""
        create = app_client.post("/episodios", json={"comprimento": 2, "max_passos": 10})
        episodio_id = create.json()["id"]

        # Passo 0: 0 -> 1
        app_client.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})
        # Passo 1: 1 -> 2 (objetivo)
        step = app_client.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})

        assert step.json()["recompensa"] == 10.0

    def test_objetivo_termina_episodio(self, app_client):
        """Passo que chega ao objetivo tem terminated=true."""
        create = app_client.post("/episodios", json={"comprimento": 2, "max_passos": 10})
        episodio_id = create.json()["id"]

        app_client.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})
        step = app_client.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})

        assert step.json()["terminated"] is True
        assert step.json()["truncated"] is False

    def test_max_passos_trunca_episodio(self, app_client):
        """Passo que atinge max_passos tem truncated=true."""
        create = app_client.post(
            "/episodios", json={"comprimento": 10, "max_passos": 2}
        )
        episodio_id = create.json()["id"]

        # Passo 0
        app_client.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})
        # Passo 1 (atinge max_passos=2)
        step = app_client.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})

        assert step.json()["truncated"] is True
        assert step.json()["terminated"] is False

    def test_terminated_e_truncated_simultaneos(self, app_client):
        """Quando objetivo E max_passos no mesmo passo, ambos true."""
        create = app_client.post(
            "/episodios", json={"comprimento": 2, "max_passos": 2}
        )
        episodio_id = create.json()["id"]

        app_client.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})
        # Passo 1: 1->2, atinge objetivo E max_passos simultaneamente
        step = app_client.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})

        assert step.json()["terminated"] is True
        assert step.json()["truncated"] is True

    def test_passo_apos_terminated_retorna_409(self, app_client):
        """POST /episodios/{id}/passos após terminated retorna 409."""
        create = app_client.post("/episodios", json={"comprimento": 2, "max_passos": 10})
        episodio_id = create.json()["id"]

        # Termina
        app_client.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})
        app_client.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})

        # Próximo passo
        response = app_client.post(f"/episodios/{episodio_id}/passos", json={"acao": 0})
        assert response.status_code == 409

    def test_passo_apos_truncated_retorna_409(self, app_client):
        """POST /episodios/{id}/passos após truncated retorna 409."""
        create = app_client.post(
            "/episodios", json={"comprimento": 10, "max_passos": 2}
        )
        episodio_id = create.json()["id"]

        # Trunca
        app_client.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})
        app_client.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})

        # Próximo passo
        response = app_client.post(f"/episodios/{episodio_id}/passos", json={"acao": 0})
        assert response.status_code == 409


# ==============================================================================
# TESTES DO ClienteAmbiente
# ==============================================================================


class TestClienteAmbiente:
    """Testes do cliente do serviço."""

    def test_cliente_cria_episodio_retorna_id_e_observacao(self, app_client):
        """ClienteAmbiente.criar_episodio() retorna (id, observacao)."""
        cliente = ClienteAmbiente(app_client, comprimento=5, max_passos=10)
        episodio_id, observacao = cliente.criar_episodio()

        assert isinstance(episodio_id, str)
        assert episodio_id != ""
        assert observacao == 0

    def test_cliente_passo_retorna_transicao(self, app_client):
        """ClienteAmbiente.passo() retorna Transicao."""
        cliente = ClienteAmbiente(app_client, comprimento=5, max_passos=10)
        episodio_id, _ = cliente.criar_episodio()

        transicao = cliente.passo(episodio_id, 1)

        assert isinstance(transicao, Transicao)
        assert transicao.observacao == 1
        assert transicao.recompensa == -1.0
        assert transicao.terminated is False
        assert transicao.truncated is False

    def test_cliente_encerra_remove_episodio(self, app_client):
        """ClienteAmbiente.encerrar() remove o episódio do serviço."""
        cliente = ClienteAmbiente(app_client, comprimento=5, max_passos=10)
        episodio_id, _ = cliente.criar_episodio()

        cliente.encerrar(episodio_id)

        # Confirma remoção
        response = app_client.get(f"/episodios/{episodio_id}")
        assert response.status_code == 404

    def test_cliente_erro_http_levanta_erro_ambiente(self, app_client):
        """ClienteAmbiente levanta ErroAmbiente em erro HTTP."""

        def mock_handler(request):
            return httpx.Response(500)

        http_mock = httpx.Client(
            transport=httpx.MockTransport(mock_handler), base_url="http://teste"
        )
        cliente = ClienteAmbiente(http_mock, comprimento=5, max_passos=10)

        with pytest.raises(ErroAmbiente) as exc_info:
            cliente.criar_episodio()

        assert exc_info.value.status == 500

    def test_cliente_timeout_levanta_erro_ambiente_sem_status(self, app_client):
        """ClienteAmbiente levanta ErroAmbiente com status=None em timeout."""

        def mock_handler(request):
            raise httpx.ReadTimeout("tempo esgotado", request=request)

        http_mock = httpx.Client(
            transport=httpx.MockTransport(mock_handler), base_url="http://teste"
        )
        cliente = ClienteAmbiente(http_mock, comprimento=5, max_passos=10)

        with pytest.raises(ErroAmbiente) as exc_info:
            cliente.criar_episodio()

        assert exc_info.value.status is None

    def test_cliente_conexao_recusada_levanta_erro_ambiente(self, app_client):
        """ClienteAmbiente levanta ErroAmbiente em erro de conexão."""

        def mock_handler(request):
            raise httpx.ConnectError("recusada", request=request)

        http_mock = httpx.Client(
            transport=httpx.MockTransport(mock_handler), base_url="http://teste"
        )
        cliente = ClienteAmbiente(http_mock, comprimento=5, max_passos=10)

        with pytest.raises(ErroAmbiente) as exc_info:
            cliente.criar_episodio()

        assert exc_info.value.status is None

    def test_cliente_resposta_invalida_levanta_erro_contrato(self, app_client):
        """ClienteAmbiente levanta ErroContrato se resposta viola contrato."""

        def mock_handler(request):
            if "passos" in str(request.url):
                # Resposta sem campos obrigatórios
                return httpx.Response(200, json={"observacao": 1})
            return httpx.Response(200, json={"id": "123", "observacao": 0, "passos": 0, "encerrado": False})

        http_mock = httpx.Client(
            transport=httpx.MockTransport(mock_handler), base_url="http://teste"
        )
        cliente = ClienteAmbiente(http_mock, comprimento=5, max_passos=10)
        cliente.criar_episodio()

        with pytest.raises(ErroContrato):
            cliente.passo("123", 1)


# ==============================================================================
# TESTES DO COLETOR - coletar_rollout
# ==============================================================================


class TestColetorRollout:
    """Testes do coletor de rollouts."""

    def test_coletor_coleta_n_passos_exatos(self, app_client):
        """coletar_rollout coleta exatamente n_passos transições."""
        cliente = ClienteAmbiente(app_client, comprimento=20, max_passos=100)

        def politica(obs):
            return 1

        def valor(obs):
            return 0.0

        rollout = coletar_rollout(
            cliente=cliente,
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
        assert len(rollout.advantages) == 5
        assert len(rollout.returns) == 5

    def test_coletor_tipos_corretos(self, app_client):
        """Rollout contém tipos numpy corretos."""
        cliente = ClienteAmbiente(app_client, comprimento=20, max_passos=100)

        rollout = coletar_rollout(
            cliente=cliente,
            politica=lambda obs: 1,
            valor=lambda obs: 0.0,
            n_passos=5,
            gamma=0.99,
            gae_lambda=0.95,
        )

        assert rollout.observacoes.dtype == np.int64
        assert rollout.acoes.dtype == np.int64
        assert rollout.recompensas.dtype == np.float64
        assert rollout.values.dtype == np.float64
        assert rollout.next_values.dtype == np.float64
        assert rollout.advantages.dtype == np.float64
        assert rollout.returns.dtype == np.float64
        assert rollout.terminated.dtype == bool
        assert rollout.truncated.dtype == bool

    def test_coletor_returns_igual_advantages_mais_values(self, app_client):
        """returns[t] = advantages[t] + values[t]."""
        cliente = ClienteAmbiente(app_client, comprimento=20, max_passos=100)

        rollout = coletar_rollout(
            cliente=cliente,
            politica=lambda obs: 1,
            valor=lambda obs: float(obs),
            n_passos=5,
            gamma=0.99,
            gae_lambda=0.95,
        )

        np.testing.assert_array_almost_equal(
            rollout.returns, rollout.advantages + rollout.values
        )

    def test_coletor_ultimo_passo_truncated_se_nao_terminou(self, app_client):
        """Último passo marcado como truncated se episódio não terminou."""
        cliente = ClienteAmbiente(app_client, comprimento=20, max_passos=100)

        rollout = coletar_rollout(
            cliente=cliente,
            politica=lambda obs: 1,
            valor=lambda obs: 0.0,
            n_passos=5,
            gamma=0.99,
            gae_lambda=0.95,
        )

        # Último passo não terminou naturalmente, deve estar truncado pelo coletor
        assert rollout.truncated[-1] is True

    def test_coletor_reinicia_apos_terminate(self, app_client):
        """Episódio termina no meio do rollout, coletor reinicia.
        
        Episódio termina no passo 2 (passos 0,1 normais, passo 2 no objetivo).
        """
        cliente = ClienteAmbiente(app_client, comprimento=3, max_passos=50)

        rollout = coletar_rollout(
            cliente=cliente,
            politica=lambda obs: 1,
            valor=lambda obs: 0.0,
            n_passos=10,
            gamma=0.99,
            gae_lambda=0.95,
        )

        # Passo 0: 0->1, passo 1: 1->2, passo 2: 2->3 (termina)
        assert rollout.terminated[2] is True
        # Passo 3 deve iniciar novo episódio (observação=0)
        assert rollout.observacoes[3] == 0
        assert len(rollout.acoes) == 10

    def test_coletor_reinicia_apos_truncate(self, app_client):
        """Episódio trunca no meio do rollout, coletor reinicia.
        
        Episódio trunca no passo 2 (max_passos=3, passos=0,1,2).
        """
        cliente = ClienteAmbiente(app_client, comprimento=20, max_passos=3)

        rollout = coletar_rollout(
            cliente=cliente,
            politica=lambda obs: 1,
            valor=lambda obs: 0.0,
            n_passos=10,
            gamma=0.99,
            gae_lambda=0.95,
        )

        # Trunca no passo 2
        assert rollout.truncated[2] is True
        assert rollout.terminated[2] is False
        # Passo 3 reinicia
        assert rollout.observacoes[3] == 0
        assert len(rollout.acoes) == 10

    def test_coletor_policia_usado_corretamente(self, app_client):
        """Política é chamada para cada observação e ação usada."""
        cliente = ClienteAmbiente(app_client, comprimento=20, max_passos=100)

        contador_direita = [0]

        def politica(obs):
            contador_direita[0] += 1
            return 1

        rollout = coletar_rollout(
            cliente=cliente,
            politica=politica,
            valor=lambda obs: 0.0,
            n_passos=5,
            gamma=0.99,
            gae_lambda=0.95,
        )

        # Política chamada 5 vezes
        assert contador_direita[0] == 5
        # Todas as ações são 1
        np.testing.assert_array_equal(rollout.acoes, np.ones(5, dtype=np.int64))

    def test_coletor_valor_chamado_para_next_values(self, app_client):
        """Função valor é usada para calcular next_values."""
        cliente = ClienteAmbiente(app_client, comprimento=20, max_passos=100)

        def valor(obs):
            return float(obs) * 2.0

        rollout = coletar_rollout(
            cliente=cliente,
            politica=lambda obs: 1,
            valor=valor,
            n_passos=5,
            gamma=0.99,
            gae_lambda=0.95,
        )

        # next_values[t] = valor(observacao[t] depois do passo)
        # Observações: 0->1->2->3->4->5
        for t in range(5):
            obs_apos_passo = rollout.observacoes[t] + 1  # todos pegam ação 1
            expected_next_value = float(obs_apos_passo) * 2.0
            assert rollout.next_values[t] == expected_next_value


# ==============================================================================
# TESTES DE CENÁRIOS COMPLEXOS
# ==============================================================================


class TestCenariosComplexosRollout:
    """Testes de fluxos complexos do coletor com episódios em diferentes estados."""

    def test_terminal_e_truncado_simultaneos_no_rollout(self, app_client):
        """Episódio termina E trunca no mesmo passo durante rollout.
        
        Termina no passo 1 (max_passos=2, comprimento=2).
        """
        cliente = ClienteAmbiente(app_client, comprimento=2, max_passos=2)

        rollout = coletar_rollout(
            cliente=cliente,
            politica=lambda obs: 1,
            valor=lambda obs: 0.0,
            n_passos=5,
            gamma=0.99,
            gae_lambda=0.95,
        )

        # Passo 1: termina e trunca simultaneamente
        assert rollout.terminated[1] is True
        assert rollout.truncated[1] is True
        # Próximo episódio começa
        assert rollout.observacoes[2] == 0

    def test_multiplos_episodios_em_um_rollout(self, app_client):
        """Coletor coleta de múltiplos episódios diferentes."""
        cliente = ClienteAmbiente(app_client, comprimento=2, max_passos=50)

        rollout = coletar_rollout(
            cliente=cliente,
            politica=lambda obs: 1,
            valor=lambda obs: 0.0,
            n_passos=20,
            gamma=0.99,
            gae_lambda=0.95,
        )

        # Primeiro episódio termina no passo 2
        assert rollout.terminated[2] is True
        # Próximos episódios também terminam em passos 2 espaçados
        # (passo 2, 5, 8, 11, 14, 17, 20)
        assert len(rollout.acoes) == 20

    def test_coletor_com_valor_dependente_observacao(self, app_client):
        """Função valor varia com observação."""
        cliente = ClienteAmbiente(app_client, comprimento=20, max_passos=100)

        def valor(obs):
            return float(obs) * 0.5

        rollout = coletar_rollout(
            cliente=cliente,
            politica=lambda obs: 1,
            valor=valor,
            n_passos=5,
            gamma=0.99,
            gae_lambda=0.95,
        )

        # values[t] = valor(observacoes[t])
        for t in range(5):
            expected = float(rollout.observacoes[t]) * 0.5
            assert rollout.values[t] == expected


# ==============================================================================
# TESTES DE ERROS E VALIDAÇÕES
# ==============================================================================


class TestErrosEValidacoes:
    """Testes de tratamento de erros e validações."""

    def test_coletor_propaga_erro_ambiente(self, app_client):
        """coletar_rollout propaga ErroAmbiente do cliente."""

        def mock_handler(request):
            return httpx.Response(503)

        http_mock = httpx.Client(
            transport=httpx.MockTransport(mock_handler), base_url="http://teste"
        )
        cliente = ClienteAmbiente(http_mock, comprimento=5, max_passos=10)

        with pytest.raises(ErroAmbiente) as exc_info:
            coletar_rollout(
                cliente=cliente,
                politica=lambda obs: 1,
                valor=lambda obs: 0.0,
                n_passos=5,
                gamma=0.99,
                gae_lambda=0.95,
            )

        assert exc_info.value.status == 503

    def test_coletor_propaga_erro_contrato(self, app_client):
        """coletar_rollout propaga ErroContrato do cliente."""

        def mock_handler(request):
            # Criar episódio funciona, mas passo retorna resposta inválida
            if "passos" in str(request.url):
                return httpx.Response(200, json={"observacao": 1})
            return httpx.Response(
                201,
                json={
                    "id": "123",
                    "observacao": 0,
                    "passos": 0,
                    "encerrado": False,
                },
            )

        http_mock = httpx.Client(
            transport=httpx.MockTransport(mock_handler), base_url="http://teste"
        )
        cliente = ClienteAmbiente(http_mock, comprimento=5, max_passos=10)

        with pytest.raises(ErroContrato):
            coletar_rollout(
                cliente=cliente,
                politica=lambda obs: 1,
                valor=lambda obs: 0.0,
                n_passos=5,
                gamma=0.99,
                gae_lambda=0.95,
            )


# ==============================================================================
# TESTES DE FRONTEIRAS DE EPISÓDIO
# ==============================================================================


class TestFronteirasEpisodio:
    """Testes de situações limites em episódios."""

    def test_episodio_termina_primeiro_passo(self, app_client):
        """Episódio termina no primeiro passo do rollout.
        
        comprimento=1 não é válido (mínimo 2), então usamos comprimento=2.
        Primeiro passo: 0->1, segundo passo: 1->2 (objetivo).
        Portanto, termina no passo 1 (segundo passo).
        """
        cliente = ClienteAmbiente(app_client, comprimento=2, max_passos=100)

        rollout = coletar_rollout(
            cliente=cliente,
            politica=lambda obs: 1,
            valor=lambda obs: 0.0,
            n_passos=10,
            gamma=0.99,
            gae_lambda=0.95,
        )

        assert rollout.terminated[1] is True
        assert rollout.observacoes[2] == 0  # Novo episódio

    def test_episodio_max_passos_primeiro_passo(self, app_client):
        """Episódio trunca no primeiro passo do rollout.
        
        max_passos=1 significa que o primeiro passo já é o máximo.
        """
        cliente = ClienteAmbiente(app_client, comprimento=20, max_passos=1)

        rollout = coletar_rollout(
            cliente=cliente,
            politica=lambda obs: 1,
            valor=lambda obs: 0.0,
            n_passos=10,
            gamma=0.99,
            gae_lambda=0.95,
        )

        assert rollout.truncated[0] is True
        assert rollout.observacoes[1] == 0  # Novo episódio

    def test_observacoes_comecam_em_zero(self, app_client):
        """Cada novo episódio começa com observacao=0."""
        cliente = ClienteAmbiente(app_client, comprimento=2, max_passos=100)

        rollout = coletar_rollout(
            cliente=cliente,
            politica=lambda obs: 1,
            valor=lambda obs: 0.0,
            n_passos=20,
            gamma=0.99,
            gae_lambda=0.95,
        )

        # Episódios começam em passos 0, 3, 6, 9, 12, 15, 18
        # (cada episódio com comprimento=2 dura 3 passos: 0->1->2, termina)
        for i in [0, 3, 6, 9, 12, 15, 18]:
            if i < len(rollout.observacoes):
                assert rollout.observacoes[i] == 0

    def test_recompensa_no_objetivo(self, app_client):
        """Recompensa é 10.0 no passo que atinge objetivo."""
        cliente = ClienteAmbiente(app_client, comprimento=3, max_passos=100)

        rollout = coletar_rollout(
            cliente=cliente,
            politica=lambda obs: 1,
            valor=lambda obs: 0.0,
            n_passos=10,
            gamma=0.99,
            gae_lambda=0.95,
        )

        # Objetivo em 3: passo 2 (0->1->2->3)
        assert rollout.recompensas[2] == 10.0
        # Próximos passos normais: -1.0
        assert rollout.recompensas[3] == -1.0

    def test_recompensa_normal_antes_objetivo(self, app_client):
        """Recompensas normais são -1.0 antes do objetivo."""
        cliente = ClienteAmbiente(app_client, comprimento=10, max_passos=100)

        rollout = coletar_rollout(
            cliente=cliente,
            politica=lambda obs: 1,
            valor=lambda obs: 0.0,
            n_passos=5,
            gamma=0.99,
            gae_lambda=0.95,
        )

        # Primeiros 5 passos não chegam ao objetivo (comprimento=10)
        np.testing.assert_array_equal(
            rollout.recompensas, np.full(5, -1.0, dtype=np.float64)
        )


# ==============================================================================
# TESTES DE AÇÕES
# ==============================================================================


class TestAcoes:
    """Testes de comportamento das ações."""

    def test_acoes_coletadas_conforme_politica(self, app_client):
        """Ações coletadas correspondem à política."""
        cliente = ClienteAmbiente(app_client, comprimento=20, max_passos=100)

        contador = [0]

        def politica(obs):
            acao = contador[0] % 2
            contador[0] += 1
            return acao

        rollout = coletar_rollout(
            cliente=cliente,
            politica=politica,
            valor=lambda obs: 0.0,
            n_passos=6,
            gamma=0.99,
            gae_lambda=0.95,
        )

        np.testing.assert_array_equal(rollout.acoes, np.array([0, 1, 0, 1, 0, 1], dtype=np.int64))

    def test_movimentos_conforme_acoes(self, app_client):
        """Observações mudam conforme as ações (direita=+1, esquerda=-1)."""
        cliente = ClienteAmbiente(app_client, comprimento=20, max_passos=100)

        acoes_planejadas = [1, 1, 0, 1, 0]

        def politica(obs):
            return acoes_planejadas[len(rollout.acoes)]

        rollout = coletar_rollout(
            cliente=cliente,
            politica=politica,
            valor=lambda obs: 0.0,
            n_passos=5,
            gamma=0.99,
            gae_lambda=0.95,
        )

        # Verifica movimento conforme ações
        # obs[0]=0, ação[0]=1 -> obs[1]=1
        # obs[1]=1, ação[1]=1 -> obs[2]=2
        # obs[2]=2, ação[2]=0 -> obs[3]=1
        # obs[3]=1, ação[3]=1 -> obs[4]=2
        # obs[4]=2, ação[4]=0 -> obs[5]=1
        expected_obs = np.array([0, 1, 2, 1, 2, 1], dtype=np.int64)
        np.testing.assert_array_equal(rollout.observacoes, expected_obs)
```