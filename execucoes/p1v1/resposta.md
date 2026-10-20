```python
import pytest
import httpx
import numpy as np
from fastapi.testclient import TestClient
from ambiente.app import criar_app
from coletor.rollout import ClienteAmbiente, ErroAmbiente, ErroContrato, Transicao, Rollout, coletar_rollout


@pytest.fixture
def cliente_api():
    """Fixture que cria um TestClient novo para cada teste."""
    return TestClient(criar_app())


# === TESTES DA API - POST /episodios (status 201, 422) ===
class TestCriarEpisodio:
    def test_criar_episodio_sucesso(self, cliente_api):
        resposta = cliente_api.post("/episodios", json={"comprimento": 10, "max_passos": 100})
        assert resposta.status_code == 201
        corpo = resposta.json()
        assert isinstance(corpo["id"], str)
        assert isinstance(corpo["observacao"], int)
        assert isinstance(corpo["passos"], int)
        assert isinstance(corpo["encerrado"], bool)

    def test_criar_episodio_comprimento_limite_minimo_sucesso(self, cliente_api):
        resposta = cliente_api.post("/episodios", json={"comprimento": 2, "max_passos": 100})
        assert resposta.status_code == 201

    def test_criar_episodio_comprimento_limite_maximo_sucesso(self, cliente_api):
        resposta = cliente_api.post("/episodios", json={"comprimento": 20, "max_passos": 100})
        assert resposta.status_code == 201

    def test_criar_episodio_max_passos_limite_minimo_sucesso(self, cliente_api):
        resposta = cliente_api.post("/episodios", json={"comprimento": 10, "max_passos": 1})
        assert resposta.status_code == 201

    def test_criar_episodio_max_passos_limite_maximo_sucesso(self, cliente_api):
        resposta = cliente_api.post("/episodios", json={"comprimento": 10, "max_passos": 500})
        assert resposta.status_code == 201

    def test_criar_episodio_comprimento_abaixo_minimo_validacao(self, cliente_api):
        resposta = cliente_api.post("/episodios", json={"comprimento": 1, "max_passos": 100})
        assert resposta.status_code == 422

    def test_criar_episodio_comprimento_acima_maximo_validacao(self, cliente_api):
        resposta = cliente_api.post("/episodios", json={"comprimento": 21, "max_passos": 100})
        assert resposta.status_code == 422

    def test_criar_episodio_max_passos_abaixo_minimo_validacao(self, cliente_api):
        resposta = cliente_api.post("/episodios", json={"comprimento": 10, "max_passos": 0})
        assert resposta.status_code == 422

    def test_criar_episodio_max_passos_acima_maximo_validacao(self, cliente_api):
        resposta = cliente_api.post("/episodios", json={"comprimento": 10, "max_passos": 501})
        assert resposta.status_code == 422

    def test_criar_episodio_campo_faltante_validacao(self, cliente_api):
        resposta = cliente_api.post("/episodios", json={"comprimento": 10})
        assert resposta.status_code == 422


# === TESTES DA API - GET /episodios/{episodio_id} (status 200, 404, 422) ===
class TestConsultarEpisodio:
    def test_consultar_episodio_sucesso(self, cliente_api):
        criar_resp = cliente_api.post("/episodios", json={"comprimento": 10, "max_passos": 100})
        episodio_id = criar_resp.json()["id"]
        resposta = cliente_api.get(f"/episodios/{episodio_id}")
        assert resposta.status_code == 200
        corpo = resposta.json()
        assert corpo["id"] == episodio_id
        assert isinstance(corpo["observacao"], int)
        assert isinstance(corpo["passos"], int)
        assert isinstance(corpo["encerrado"], bool)

    def test_consultar_episodio_nao_existe_404(self, cliente_api):
        resposta = cliente_api.get("/episodios/inexistente")
        assert resposta.status_code == 404


# === TESTES DA API - DELETE /episodios/{episodio_id} (status 204, 404, 422) ===
class TestRemoverEpisodio:
    def test_remover_episodio_sucesso(self, cliente_api):
        criar_resp = cliente_api.post("/episodios", json={"comprimento": 10, "max_passos": 100})
        episodio_id = criar_resp.json()["id"]
        resposta = cliente_api.delete(f"/episodios/{episodio_id}")
        assert resposta.status_code == 204

    def test_remover_episodio_nao_existe_404(self, cliente_api):
        resposta = cliente_api.delete("/episodios/inexistente")
        assert resposta.status_code == 404

    def test_remover_episodio_duas_vezes_404(self, cliente_api):
        criar_resp = cliente_api.post("/episodios", json={"comprimento": 10, "max_passos": 100})
        episodio_id = criar_resp.json()["id"]
        cliente_api.delete(f"/episodios/{episodio_id}")
        resposta = cliente_api.delete(f"/episodios/{episodio_id}")
        assert resposta.status_code == 404


# === TESTES DA API - POST /episodios/{episodio_id}/passos (status 200, 404, 409, 422) ===
class TestDarPasso:
    def test_dar_passo_sucesso(self, cliente_api):
        criar_resp = cliente_api.post("/episodios", json={"comprimento": 10, "max_passos": 100})
        episodio_id = criar_resp.json()["id"]
        resposta = cliente_api.post(f"/episodios/{episodio_id}/passos", json={"acao": 0})
        assert resposta.status_code == 200
        corpo = resposta.json()
        assert isinstance(corpo["observacao"], int)
        assert isinstance(corpo["recompensa"], (int, float))
        assert isinstance(corpo["terminated"], bool)
        assert isinstance(corpo["truncated"], bool)

    def test_dar_passo_acao_esquerda_sucesso(self, cliente_api):
        criar_resp = cliente_api.post("/episodios", json={"comprimento": 10, "max_passos": 100})
        episodio_id = criar_resp.json()["id"]
        resposta = cliente_api.post(f"/episodios/{episodio_id}/passos", json={"acao": 0})
        assert resposta.status_code == 200

    def test_dar_passo_acao_direita_sucesso(self, cliente_api):
        criar_resp = cliente_api.post("/episodios", json={"comprimento": 10, "max_passos": 100})
        episodio_id = criar_resp.json()["id"]
        resposta = cliente_api.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})
        assert resposta.status_code == 200

    def test_dar_passo_episodio_nao_existe_404(self, cliente_api):
        resposta = cliente_api.post("/episodios/inexistente/passos", json={"acao": 0})
        assert resposta.status_code == 404

    def test_dar_passo_acao_negativa_validacao(self, cliente_api):
        criar_resp = cliente_api.post("/episodios", json={"comprimento": 10, "max_passos": 100})
        episodio_id = criar_resp.json()["id"]
        resposta = cliente_api.post(f"/episodios/{episodio_id}/passos", json={"acao": -1})
        assert resposta.status_code == 422

    def test_dar_passo_acao_invalida_validacao(self, cliente_api):
        criar_resp = cliente_api.post("/episodios", json={"comprimento": 10, "max_passos": 100})
        episodio_id = criar_resp.json()["id"]
        resposta = cliente_api.post(f"/episodios/{episodio_id}/passos", json={"acao": 2})
        assert resposta.status_code == 422

    def test_dar_passo_episodio_terminated_409(self, cliente_api):
        criar_resp = cliente_api.post("/episodios", json={"comprimento": 2, "max_passos": 100})
        episodio_id = criar_resp.json()["id"]
        for _ in range(2):
            cliente_api.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})
        resposta = cliente_api.post(f"/episodios/{episodio_id}/passos", json={"acao": 0})
        assert resposta.status_code == 409

    def test_dar_passo_episodio_truncated_409(self, cliente_api):
        criar_resp = cliente_api.post("/episodios", json={"comprimento": 100, "max_passos": 2})
        episodio_id = criar_resp.json()["id"]
        for _ in range(2):
            cliente_api.post(f"/episodios/{episodio_id}/passos", json={"acao": 1})
        resposta = cliente_api.post(f"/episodios/{episodio_id}/passos", json={"acao": 0})
        assert resposta.status_code == 409

    def test_dar_passo_corpo_faltante_validacao(self, cliente_api):
        criar_resp = cliente_api.post("/episodios", json={"comprimento": 10, "max_passos": 100})
        episodio_id = criar_resp.json()["id"]
        resposta = cliente_api.post(f"/episodios/{episodio_id}/passos", json={})
        assert resposta.status_code == 422


# === TESTES DO ClienteAmbiente ===
class TestClienteAmbiente:
    def test_cliente_criar_episodio_retorna_tupla(self, cliente_api):
        cliente = ClienteAmbiente(cliente_api, comprimento=10, max_passos=100)
        resultado = cliente.criar_episodio()
        assert isinstance(resultado, tuple) and len(resultado) == 2
        episodio_id, observacao = resultado
        assert isinstance(episodio_id, str) and isinstance(observacao, int)

    def test_cliente_passo_retorna_transicao(self, cliente_api):
        cliente = ClienteAmbiente(cliente_api, comprimento=10, max_passos=100)
        episodio_id, _ = cliente.criar_episodio()
        transicao = cliente.passo(episodio_id, 0)
        assert isinstance(transicao, Transicao)
        assert isinstance(transicao.observacao, int)
        assert isinstance(transicao.recompensa, float)
        assert isinstance(transicao.terminated, bool)
        assert isinstance(transicao.truncated, bool)

    def test_cliente_encerrar_remove_episodio(self, cliente_api):
        cliente = ClienteAmbiente(cliente_api, comprimento=10, max_passos=100)
        episodio_id, _ = cliente.criar_episodio()
        cliente.encerrar(episodio_id)
        resposta = cliente_api.get(f"/episodios/{episodio_id}")
        assert resposta.status_code == 404

    def test_cliente_passo_acao_0(self, cliente_api):
        cliente = ClienteAmbiente(cliente_api, comprimento=10, max_passos=100)
        episodio_id, _ = cliente.criar_episodio()
        transicao = cliente.passo(episodio_id, 0)
        assert isinstance(transicao.observacao, int)

    def test_cliente_passo_acao_1(self, cliente_api):
        cliente = ClienteAmbiente(cliente_api, comprimento=10, max_passos=100)
        episodio_id, _ = cliente.criar_episodio()
        transicao = cliente.passo(episodio_id, 1)
        assert isinstance(transicao.observacao, int)


# === TESTES DO coletar_rollout ===
class TestColetorRollout:
    def test_coletar_rollout_retorna_rollout(self, cliente_api):
        cliente = ClienteAmbiente(cliente_api, comprimento=10, max_passos=100)
        def politica(obs):
            return 1
        def valor(obs):
            return 0.0
        rollout = coletar_rollout(cliente, politica, valor, 5, 0.99, 0.95)
        assert isinstance(rollout, Rollout)

    def test_coletar_rollout_dimensoes_corretas(self, cliente_api):
        cliente = ClienteAmbiente(cliente_api, comprimento=10, max_passos=100)
        def politica(obs):
            return 1
        def valor(obs):
            return 0.0
        n_passos = 5
        rollout = coletar_rollout(cliente, politica, valor, n_passos, 0.99, 0.95)
        assert len(rollout.observacoes) == n_passos
        assert len(rollout.acoes) == n_passos
        assert len(rollout.recompensas) == n_passos
        assert len(rollout.values) == n_passos
        assert len(rollout.next_values) == n_passos
        assert len(rollout.terminated) == n_passos
        assert len(rollout.truncated) == n_passos
        assert len(rollout.advantages) == n_passos
        assert len(rollout.returns) == n_passos

    def test_coletar_rollout_dtypes_corretos(self, cliente_api):
        cliente = ClienteAmbiente(cliente_api, comprimento=10, max_passos=100)
        def politica(obs):
            return 1
        def valor(obs):
            return 0.0
        rollout = coletar_rollout(cliente, politica, valor, 5, 0.99, 0.95)
        assert rollout.observacoes.dtype == np.int64
        assert rollout.acoes.dtype == np.int64
        assert rollout.recompensas.dtype == np.float64
        assert rollout.values.dtype == np.float64
        assert rollout.next_values.dtype == np.float64
        assert rollout.terminated.dtype == np.bool_
        assert rollout.truncated.dtype == np.bool_
        assert rollout.advantages.dtype == np.float64
        assert rollout.returns.dtype == np.float64

    def test_coletar_rollout_com_reinicio_episodio(self, cliente_api):
        cliente = ClienteAmbiente(cliente_api, comprimento=2, max_passos=100)
        def politica(obs):
            return 1
        def valor(obs):
            return 0.0
        rollout = coletar_rollout(cliente, politica, valor, 10, 0.99, 0.95)
        assert len(rollout.observacoes) == 10
        assert np.any(rollout.terminated)

    def test_coletar_rollout_ultimo_passo_truncado(self, cliente_api):
        cliente = ClienteAmbiente(cliente_api, comprimento=100, max_passos=100)
        def politica(obs):
            return 1
        def valor(obs):
            return 0.0
        rollout = coletar_rollout(cliente, politica, valor, 5, 0.99, 0.95)
        assert rollout.truncated[-1]

    def test_coletar_rollout_encerra_episodio_aberto(self, cliente_api):
        cliente = ClienteAmbiente(cliente_api, comprimento=10, max_passos=100)
        def politica(obs):
            return 1
        def valor(obs):
            return 0.0
        rollout = coletar_rollout(cliente, politica, valor, 5, 0.99, 0.95)
        assert isinstance(rollout, Rollout)

    def test_coletar_rollout_n_passos_zero_erro(self, cliente_api):
        cliente = ClienteAmbiente(cliente_api, comprimento=10, max_passos=100)
        def politica(obs):
            return 1
        def valor(obs):
            return 0.0
        with pytest.raises(ValueError):
            coletar_rollout(cliente, politica, valor, 0, 0.99, 0.95)

    def test_coletar_rollout_n_passos_negativo_erro(self, cliente_api):
        cliente = ClienteAmbiente(cliente_api, comprimento=10, max_passos=100)
        def politica(obs):
            return 1
        def valor(obs):
            return 0.0
        with pytest.raises(ValueError):
            coletar_rollout(cliente, politica, valor, -1, 0.99, 0.95)


# === TESTES DE ERRO AMBIENTE (HTTP 5xx, timeout, conexão) ===
class TestErroAmbienteHttp500:
    def test_cliente_erro_http_500_criar_episodio(self):
        http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)), base_url="http://ambiente")
        cliente = ClienteAmbiente(http, comprimento=10, max_passos=100)
        with pytest.raises(ErroAmbiente) as exc_info:
            cliente.criar_episodio()
        assert exc_info.value.status == 500

    def test_cliente_erro_http_503_passo(self):
        def handler(request):
            if "/passos" not in request.url.path:
                return httpx.Response(201, json={"id": "ep1", "observacao": 0, "passos": 0, "encerrado": False})
            return httpx.Response(503)
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
        cliente = ClienteAmbiente(http, comprimento=10, max_passos=100)
        cliente.criar_episodio()
        with pytest.raises(ErroAmbiente) as exc_info:
            cliente.passo("ep1", 0)
        assert exc_info.value.status == 503

    def test_cliente_erro_http_502_encerrar(self):
        def handler(request):
            if request.method == "DELETE":
                return httpx.Response(502)
            return httpx.Response(201, json={"id": "ep1", "observacao": 0, "passos": 0, "encerrado": False})
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
        cliente = ClienteAmbiente(http, comprimento=10, max_passos=100)
        cliente.criar_episodio()
        with pytest.raises(ErroAmbiente) as exc_info:
            cliente.encerrar("ep1")
        assert exc_info.value.status == 502


# === TESTES DE TIMEOUT ===
class TestErroTimeout:
    def test_cliente_timeout_criar_episodio(self):
        http = httpx.Client(transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.TimeoutException())), base_url="http://ambiente")
        cliente = ClienteAmbiente(http, comprimento=10, max_passos=100)
        with pytest.raises(ErroAmbiente) as exc_info:
            cliente.criar_episodio()
        assert exc_info.value.status is None

    def test_cliente_timeout_passo(self):
        def handler(request):
            if "/passos" not in request.url.path:
                return httpx.Response(201, json={"id": "ep1", "observacao": 0, "passos": 0, "encerrado": False})
            raise httpx.TimeoutException()
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
        cliente = ClienteAmbiente(http, comprimento=10, max_passos=100)
        cliente.criar_episodio()
        with pytest.raises(ErroAmbiente) as exc_info:
            cliente.passo("ep1", 0)
        assert exc_info.value.status is None

    def test_cliente_timeout_encerrar(self):
        def handler(request):
            if request.method == "DELETE":
                raise httpx.TimeoutException()
            return httpx.Response(201, json={"id": "ep1", "observacao": 0, "passos": 0, "encerrado": False})
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
        cliente = ClienteAmbiente(http, comprimento=10, max_passos=100)
        cliente.criar_episodio()
        with pytest.raises(ErroAmbiente) as exc_info:
            cliente.encerrar("ep1")
        assert exc_info.value.status is None


# === TESTES DE FALHA DE CONEXÃO ===
class TestErroConexao:
    def test_cliente_conexao_falha_criar_episodio(self):
        http = httpx.Client(transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.ConnectError())), base_url="http://ambiente")
        cliente = ClienteAmbiente(http, comprimento=10, max_passos=100)
        with pytest.raises(ErroAmbiente) as exc_info:
            cliente.criar_episodio()
        assert exc_info.value.status is None

    def test_cliente_conexao_falha_passo(self):
        def handler(request):
            if "/passos" not in request.url.path:
                return httpx.Response(201, json={"id": "ep1", "observacao": 0, "passos": 0, "encerrado": False})
            raise httpx.ConnectError()
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
        cliente = ClienteAmbiente(http, comprimento=10, max_passos=100)
        cliente.criar_episodio()
        with pytest.raises(ErroAmbiente) as exc_info:
            cliente.passo("ep1", 0)
        assert exc_info.value.status is None

    def test_cliente_conexao_falha_encerrar(self):
        def handler(request):
            if request.method == "DELETE":
                raise httpx.ConnectError()
            return httpx.Response(201, json={"id": "ep1", "observacao": 0, "passos": 0, "encerrado": False})
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
        cliente = ClienteAmbiente(http, comprimento=10, max_passos=100)
        cliente.criar_episodio()
        with pytest.raises(ErroAmbiente) as exc_info:
            cliente.encerrar("ep1")
        assert exc_info.value.status is None


# === TESTES DE ERRO DE CONTRATO ===
class TestErroContrato:
    def test_cliente_contrato_criar_episodio_falta_campo(self):
        http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(201, json={"id": "ep1"})), base_url="http://ambiente")
        cliente = ClienteAmbiente(http, comprimento=10, max_passos=100)
        with pytest.raises(ErroContrato):
            cliente.criar_episodio()

    def test_cliente_contrato_criar_episodio_tipo_invalido(self):
        http = httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(201, json={"id": "ep1", "observacao": "str", "passos": 0, "encerrado": False})),
            base_url="http://ambiente"
        )
        cliente = ClienteAmbiente(http, comprimento=10, max_passos=100)
        with pytest.raises(ErroContrato):
            cliente.criar_episodio()

    def test_cliente_contrato_passo_falta_campo(self):
        def handler(request):
            if "/passos" not in request.url.path:
                return httpx.Response(201, json={"id": "ep1", "observacao": 0, "passos": 0, "encerrado": False})
            return httpx.Response(200, json={"observacao": 0})
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
        cliente = ClienteAmbiente(http, comprimento=10, max_passos=100)
        cliente.criar_episodio()
        with pytest.raises(ErroContrato):
            cliente.passo("ep1", 0)

    def test_cliente_contrato_passo_tipo_invalido(self):
        def handler(request):
            if "/passos" not in request.url.path:
                return httpx.Response(201, json={"id": "ep1", "observacao": 0, "passos": 0, "encerrado": False})
            return httpx.Response(200, json={"observacao": "str", "recompensa": 0.0, "terminated": False, "truncated": False})
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
        cliente = ClienteAmbiente(http, comprimento=10, max_passos=100)
        cliente.criar_episodio()
        with pytest.raises(ErroContrato):
            cliente.passo("ep1", 0)

    def test_coletor_contrato_criar_episodio(self):
        http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(201, json={})), base_url="http://ambiente")
        cliente = ClienteAmbiente(http, comprimento=10, max_passos=100)
        def politica(obs):
            return 1
        def valor(obs):
            return 0.0
        with pytest.raises(ErroContrato):
            coletar_rollout(cliente, politica, valor, 5, 0.99, 0.95)

    def test_coletor_contrato_passo(self):
        def handler(request):
            if "/passos" not in request.url.path:
                return httpx.Response(201, json={"id": "ep1", "observacao": 0, "passos": 0, "encerrado": False})
            return httpx.Response(200, json={})
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")
        cliente = ClienteAmbiente(http, comprimento=10, max_passos=100)
        def politica(obs):
            return 1
        def valor(obs):
            return 0.0
        with pytest.raises(ErroContrato):
            coletar_rollout(cliente, politica, valor, 5, 0.99, 0.95)


# === TESTES DO FLUXO DO COLETOR ===
class TestFluxoColetorCompleto:
    def test_coletor_fluxo_simples(self, cliente_api):
        cliente = ClienteAmbiente(cliente_api, comprimento=10, max_passos=100)
        def politica(obs):
            return 1
        def valor(obs):
            return 0.5
        rollout = coletar_rollout(cliente, politica, valor, 8, 0.99, 0.95)
        assert len(rollout.observacoes) == 8
        assert rollout.returns.shape == rollout.advantages.shape

    def test_coletor_com_multiplos_episodios(self, cliente_api):
        cliente = ClienteAmbiente(cliente_api, comprimento=3, max_passos=100)
        def politica(obs):
            return 1
        def valor(obs):
            return 0.0
        rollout = coletar_rollout(cliente, politica, valor, 20, 0.99, 0.95)
        assert len(rollout.observacoes) == 20
        assert np.sum(rollout.terminated) >= 5

    def test_coletor_next_values_usa_obs_final(self, cliente_api):
        cliente = ClienteAmbiente(cliente_api, comprimento=10, max_passos=100)
        def politica(obs):
            return 1
        def valor(obs):
            return float(obs)
        rollout = coletar_rollout(cliente, politica, valor, 5, 0.99, 0.95)
        assert rollout.next_values.dtype == np.float64
        assert len(rollout.next_values) == 5
```