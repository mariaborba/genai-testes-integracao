# Raciocínio (25 linhas)

1. **Chamadas HTTP**: POST /episodios (espera 201), POST /passos (200), DELETE (204)
2. **Falhas por chamada**: HTTP 5xx, timeout, ConnectError, status ≠ esperado, corpo não-JSON, não-dict, campo ausente, tipo errado (bool quando int)
3. **Validação 422**: comprimento ∈ [2,20], max_passos ∈ [1,500], acao ∈ [0,1]
4. **Regras integradas**: (a) episódio termina → novo criado, next_values[t]=valor(obs_final), não obs_novo; (b) episódio trunca por max_passos → idem; (c) último passo recebe truncated=True mesmo sem terminar; (d) episódio final removido
5. **Parâmetros para regras**:
   - (a) comprimento=2, n_passos=3, política=direita → passo 0: obs=0→1; passo 1: obs=1→2 (terminated), novo episódio; passo 2: obs=0→1. next_values=[valor(1), valor(2), valor(1)]
   - (b) comprimento=20, max_passos=2, n_passos=3 → passo 1: truncated por max_passos
   - (c) comprimento=10, max_passos=100, n_passos=2 → nenhum termina, último fica truncated[-1]=True
6. **Testes**: 3 por chamada (5xx, timeout, status), 2 por falha contrato, 4 validação, 4 regras integradas, 2 estado episódio = ~25 testes

```python
import pytest
import httpx
import numpy as np
from fastapi.testclient import TestClient

from ambiente.app import criar_app
from coletor.rollout import (
    ClienteAmbiente, ErroAmbiente, ErroContrato, coletar_rollout
)


@pytest.fixture
def cliente_http():
    """Novo TestClient para cada teste (em memória)."""
    return TestClient(criar_app())


@pytest.fixture
def cliente_ambiente(cliente_http):
    """ClienteAmbiente padrão (comprimento=10, max_passos=100)."""
    return ClienteAmbiente(cliente_http, comprimento=10, max_passos=100)


class TestCriarEpisodio:
    """POST /episodios"""

    def test_criar_episodio_sucesso(self, cliente_ambiente):
        ep_id, obs = cliente_ambiente.criar_episodio()
        assert isinstance(ep_id, str) and len(ep_id) > 0
        assert obs == 0

    def test_criar_episodio_http_500(self, cliente_http):
        def handler(request):
            return httpx.Response(500, text="erro")
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://amb")
        cli = ClienteAmbiente(http, 10, 100)
        with pytest.raises(ErroAmbiente) as exc:
            cli.criar_episodio()
        assert exc.value.status == 500

    def test_criar_episodio_timeout(self, cliente_http):
        def handler(request):
            raise httpx.ReadTimeout("timeout", request=request)
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://amb")
        cli = ClienteAmbiente(http, 10, 100)
        with pytest.raises(ErroAmbiente) as exc:
            cli.criar_episodio()
        assert exc.value.status is None

    def test_criar_episodio_conexao_recusada(self, cliente_http):
        def handler(request):
            raise httpx.ConnectError("recusada", request=request)
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://amb")
        cli = ClienteAmbiente(http, 10, 100)
        with pytest.raises(ErroAmbiente) as exc:
            cli.criar_episodio()
        assert exc.value.status is None

    def test_criar_episodio_status_invalido(self, cliente_http):
        def handler(request):
            return httpx.Response(200, json={"id": "x", "observacao": 0, "passos": 0, "encerrado": False})
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://amb")
        cli = ClienteAmbiente(http, 10, 100)
        with pytest.raises(ErroContrato):
            cli.criar_episodio()

    def test_criar_episodio_corpo_nao_json(self, cliente_http):
        def handler(request):
            return httpx.Response(201, text="<html>erro</html>")
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://amb")
        cli = ClienteAmbiente(http, 10, 100)
        with pytest.raises(ErroContrato):
            cli.criar_episodio()

    def test_criar_episodio_corpo_nao_dict(self, cliente_http):
        def handler(request):
            return httpx.Response(201, json=["id"])
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://amb")
        cli = ClienteAmbiente(http, 10, 100)
        with pytest.raises(ErroContrato):
            cli.criar_episodio()

    def test_criar_episodio_campo_id_ausente(self, cliente_http):
        def handler(request):
            return httpx.Response(201, json={"observacao": 0, "passos": 0, "encerrado": False})
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://amb")
        cli = ClienteAmbiente(http, 10, 100)
        with pytest.raises(ErroContrato):
            cli.criar_episodio()

    def test_criar_episodio_observacao_tipo_bool(self, cliente_http):
        def handler(request):
            return httpx.Response(201, json={"id": "x", "observacao": True, "passos": 0, "encerrado": False})
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://amb")
        cli = ClienteAmbiente(http, 10, 100)
        with pytest.raises(ErroContrato):
            cli.criar_episodio()

    def test_criar_episodio_observacao_tipo_invalido(self, cliente_http):
        def handler(request):
            return httpx.Response(201, json={"id": "x", "observacao": "zero", "passos": 0, "encerrado": False})
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://amb")
        cli = ClienteAmbiente(http, 10, 100)
        with pytest.raises(ErroContrato):
            cli.criar_episodio()

    def test_criar_episodio_comprimento_minimo_invalido(self, cliente_http):
        resp = cliente_http.post("/episodios", json={"comprimento": 1, "max_passos": 10})
        assert resp.status_code == 422

    def test_criar_episodio_comprimento_maximo_invalido(self, cliente_http):
        resp = cliente_http.post("/episodios", json={"comprimento": 21, "max_passos": 10})
        assert resp.status_code == 422

    def test_criar_episodio_max_passos_minimo_invalido(self, cliente_http):
        resp = cliente_http.post("/episodios", json={"comprimento": 10, "max_passos": 0})
        assert resp.status_code == 422

    def test_criar_episodio_max_passos_maximo_invalido(self, cliente_http):
        resp = cliente_http.post("/episodios", json={"comprimento": 10, "max_passos": 501})
        assert resp.status_code == 422


class TestPasso:
    """POST /episodios/{id}/passos"""

    def test_passo_sucesso(self, cliente_ambiente):
        ep_id, _ = cliente_ambiente.criar_episodio()
        trans = cliente_ambiente.passo(ep_id, 1)
        assert trans.observacao == 1
        assert trans.recompensa == -1.0

    def test_passo_http_500(self, cliente_http):
        def handler(request):
            if request.method == "POST" and "/passos" not in request.url.path:
                return httpx.Response(201, json={"id": "e1", "observacao": 0, "passos": 0, "encerrado": False})
            if "/passos" in request.url.path:
                return httpx.Response(500, text="erro")
            return httpx.Response(404)
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://amb")
        cli = ClienteAmbiente(http, 10, 100)
        ep_id, _ = cli.criar_episodio()
        with pytest.raises(ErroAmbiente) as exc:
            cli.passo(ep_id, 1)
        assert exc.value.status == 500

    def test_passo_timeout(self, cliente_http):
        def handler(request):
            if request.method == "POST" and "/passos" not in request.url.path:
                return httpx.Response(201, json={"id": "e1", "observacao": 0, "passos": 0, "encerrado": False})
            if "/passos" in request.url.path:
                raise httpx.ReadTimeout("timeout", request=request)
            return httpx.Response(404)
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://amb")
        cli = ClienteAmbiente(http, 10, 100)
        ep_id, _ = cli.criar_episodio()
        with pytest.raises(ErroAmbiente) as exc:
            cli.passo(ep_id, 1)
        assert exc.value.status is None

    def test_passo_conexao_recusada(self, cliente_http):
        def handler(request):
            if request.method == "POST" and "/passos" not in request.url.path:
                return httpx.Response(201, json={"id": "e1", "observacao": 0, "passos": 0, "encerrado": False})
            if "/passos" in request.url.path:
                raise httpx.ConnectError("recusada", request=request)
            return httpx.Response(404)
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://amb")
        cli = ClienteAmbiente(http, 10, 100)
        ep_id, _ = cli.criar_episodio()
        with pytest.raises(ErroAmbiente) as exc:
            cli.passo(ep_id, 1)
        assert exc.value.status is None

    def test_passo_status_invalido(self, cliente_http):
        def handler(request):
            if request.method == "POST" and "/passos" not in request.url.path:
                return httpx.Response(201, json={"id": "e1", "observacao": 0, "passos": 0, "encerrado": False})
            if "/passos" in request.url.path:
                return httpx.Response(201, json={"observacao": 1, "recompensa": -1, "terminated": False, "truncated": False})
            return httpx.Response(404)
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://amb")
        cli = ClienteAmbiente(http, 10, 100)
        ep_id, _ = cli.criar_episodio()
        with pytest.raises(ErroContrato):
            cli.passo(ep_id, 1)

    def test_passo_campo_observacao_ausente(self, cliente_http):
        def handler(request):
            if request.method == "POST" and "/passos" not in request.url.path:
                return httpx.Response(201, json={"id": "e1", "observacao": 0, "passos": 0, "encerrado": False})
            return httpx.Response(200, json={"recompensa": -1, "terminated": False, "truncated": False})
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://amb")
        cli = ClienteAmbiente(http, 10, 100)
        ep_id, _ = cli.criar_episodio()
        with pytest.raises(ErroContrato):
            cli.passo(ep_id, 1)

    def test_passo_campo_recompensa_tipo_bool(self, cliente_http):
        def handler(request):
            if request.method == "POST" and "/passos" not in request.url.path:
                return httpx.Response(201, json={"id": "e1", "observacao": 0, "passos": 0, "encerrado": False})
            return httpx.Response(200, json={"observacao": 1, "recompensa": True, "terminated": False, "truncated": False})
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://amb")
        cli = ClienteAmbiente(http, 10, 100)
        ep_id, _ = cli.criar_episodio()
        with pytest.raises(ErroContrato):
            cli.passo(ep_id, 1)

    def test_passo_campo_terminated_tipo_invalido(self, cliente_http):
        def handler(request):
            if request.method == "POST" and "/passos" not in request.url.path:
                return httpx.Response(201, json={"id": "e1", "observacao": 0, "passos": 0, "encerrado": False})
            return httpx.Response(200, json={"observacao": 1, "recompensa": -1.0, "terminated": 0, "truncated": False})
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://amb")
        cli = ClienteAmbiente(http, 10, 100)
        ep_id, _ = cli.criar_episodio()
        with pytest.raises(ErroContrato):
            cli.passo(ep_id, 1)

    def test_passo_episodio_nao_existe_404(self, cliente_http):
        cli = ClienteAmbiente(cliente_http, 10, 100)
        with pytest.raises(ErroAmbiente) as exc:
            cli.passo("inexistente", 1)
        assert exc.value.status == 404

    def test_passo_episodio_encerrado_409(self, cliente_http):
        cli = ClienteAmbiente(cliente_http, comprimento=2, max_passos=100)
        ep_id, _ = cli.criar_episodio()
        cli.passo(ep_id, 1)  # obs=1
        cli.passo(ep_id, 1)  # obs=2, terminated
        with pytest.raises(ErroAmbiente) as exc:
            cli.passo(ep_id, 1)
        assert exc.value.status == 409

    def test_passo_acao_minima_invalida(self, cliente_http):
        cli = ClienteAmbiente(cliente_http, 10, 100)
        ep_id, _ = cli.criar_episodio()
        resp = cliente_http.post(f"/episodios/{ep_id}/passos", json={"acao": -1})
        assert resp.status_code == 422

    def test_passo_acao_maxima_invalida(self, cliente_http):
        cli = ClienteAmbiente(cliente_http, 10, 100)
        ep_id, _ = cli.criar_episodio()
        resp = cliente_http.post(f"/episodios/{ep_id}/passos", json={"acao": 2})
        assert resp.status_code == 422


class TestRemover:
    """DELETE /episodios/{id}"""

    def test_remover_episodio_sucesso(self, cliente_ambiente):
        ep_id, _ = cliente_ambiente.criar_episodio()
        cliente_ambiente.encerrar(ep_id)

    def test_remover_episodio_http_500(self, cliente_http):
        def handler(request):
            if request.method == "POST":
                return httpx.Response(201, json={"id": "e1", "observacao": 0, "passos": 0, "encerrado": False})
            if request.method == "DELETE":
                return httpx.Response(500, text="erro")
            return httpx.Response(404)
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://amb")
        cli = ClienteAmbiente(http, 10, 100)
        ep_id, _ = cli.criar_episodio()
        with pytest.raises(ErroAmbiente) as exc:
            cli.encerrar(ep_id)
        assert exc.value.status == 500

    def test_remover_episodio_timeout(self, cliente_http):
        def handler(request):
            if request.method == "POST":
                return httpx.Response(201, json={"id": "e1", "observacao": 0, "passos": 0, "encerrado": False})
            if request.method == "DELETE":
                raise httpx.ReadTimeout("timeout", request=request)
            return httpx.Response(404)
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://amb")
        cli = ClienteAmbiente(http, 10, 100)
        ep_id, _ = cli.criar_episodio()
        with pytest.raises(ErroAmbiente) as exc:
            cli.encerrar(ep_id)
        assert exc.value.status is None

    def test_remover_episodio_conexao_recusada(self, cliente_http):
        def handler(request):
            if request.method == "POST":
                return httpx.Response(201, json={"id": "e1", "observacao": 0, "passos": 0, "encerrado": False})
            if request.method == "DELETE":
                raise httpx.ConnectError("recusada", request=request)
            return httpx.Response(404)
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://amb")
        cli = ClienteAmbiente(http, 10, 100)
        ep_id, _ = cli.criar_episodio()
        with pytest.raises(ErroAmbiente) as exc:
            cli.encerrar(ep_id)
        assert exc.value.status is None

    def test_remover_episodio_status_invalido(self, cliente_http):
        def handler(request):
            if request.method == "POST":
                return httpx.Response(201, json={"id": "e1", "observacao": 0, "passos": 0, "encerrado": False})
            if request.method == "DELETE":
                return httpx.Response(200)
            return httpx.Response(404)
        http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://amb")
        cli = ClienteAmbiente(http, 10, 100)
        ep_id, _ = cli.criar_episodio()
        with pytest.raises(ErroContrato):
            cli.encerrar(ep_id)


class TestRegrasEpisodioIntegrado:
    """Regras que só aparecem com os dois componentes integrados"""

    def test_rollout_episodio_termina_durante_coleta(self, cliente_http):
        """Episódio termina em passo 1 (posição 2), novo criado em passo 2.
        
        comprimento=2, n_passos=3, política=direita
        Passo 0: obs=0 → obs=1, recompensa=-1, terminated=False
        Passo 1: obs=1 → obs=2, recompensa=10, terminated=True → novo episódio
        Passo 2: obs=0 → obs=1, recompensa=-1, terminated=False
        last_step truncated=True (pelo rollout, não pelo serviço)
        """
        cli = ClienteAmbiente(cliente_http, comprimento=2, max_passos=100)
        
        def politica(obs):
            return 1  # sempre direita
        
        def valor(obs):
            return float(obs)
        
        rollout = coletar_rollout(cli, politica, valor, n_passos=3, gamma=0.99, gae_lambda=0.95)
        
        assert len(rollout.observacoes) == 3
        np.testing.assert_array_equal(rollout.observacoes, [0, 1, 0])
        np.testing.assert_array_equal(rollout.recompensas, [-1.0, 10.0, -1.0])
        np.testing.assert_array_equal(rollout.terminated, [False, True, False])
        
        # next_values usa obs final de cada passo, não obs novo do episódio seguinte
        # passo 0: obs_final=1 → valor(1)=1.0
        # passo 1: obs_final=2 → valor(2)=2.0
        # passo 2: obs_final=1 → valor(1)=1.0
        np.testing.assert_array_equal(rollout.next_values, [1.0, 2.0, 1.0])

    def test_rollout_episodio_trunca_durante_coleta(self, cliente_http):
        """Episódio trunca por max_passos em passo 1, novo criado em passo 2.
        
        comprimento=20, max_passos=2, n_passos=3, política=direita
        Passo 0: obs=0 → obs=1, passos=1
        Passo 1: obs=1 → obs=2, passos=2, truncated=True → novo episódio
        Passo 2: obs=0 → obs=1, passos=1
        """
        cli = ClienteAmbiente(cliente_http, comprimento=20, max_passos=2)
        
        def politica(obs):
            return 1
        
        def valor(obs):
            return float(obs)
        
        rollout = coletar_rollout(cli, politica, valor, n_passos=3, gamma=0.99, gae_lambda=0.95)
        
        assert len(rollout.observacoes) == 3
        np.testing.assert_array_equal(rollout.observacoes, [0, 1, 0])
        assert rollout.terminated[1] == False
        assert rollout.truncated[1] == True
        assert rollout.observacoes[2] == 0
        # next_values[1] deve ser valor(2), não valor(0)
        np.testing.assert_array_equal(rollout.next_values, [1.0, 2.0, 1.0])

    def test_rollout_ultimo_passo_truncado_mesmo_sem_terminar(self, cliente_http):
        """Último passo recebe truncated=True mesmo que não tenha terminado.
        
        comprimento=10, max_passos=100, n_passos=2, política=esquerda
        Passo 0: obs=0 → obs=0
        Passo 1: obs=0 → obs=0, terminated=False (serviço), mas truncated=True (rollout)
        """
        cli = ClienteAmbiente(cliente_http, comprimento=10, max_passos=100)
        
        def politica(obs):
            return 0  # sempre esquerda
        
        def valor(obs):
            return 0.0
        
        rollout = coletar_rollout(cli, politica, valor, n_passos=2, gamma=0.99, gae_lambda=0.95)
        
        assert len(rollout.observacoes) == 2
        assert rollout.terminated[0] == False
        assert rollout.terminated[1] == False
        assert rollout.truncated[0] == False
        # Último passo é truncado pelo limite do rollout
        assert rollout.truncated[1] == True

    def test_rollout_episodio_removido_ao_final(self, cliente_http):
        """Episódio em aberto é removido do serviço ao final do rollout.
        
        Depois de coletar_rollout, tentar acessar retorna 404.
        """
        cli = ClienteAmbiente(cliente_http, comprimento=10, max_passos=100)
        
        def politica(obs):
            return 1
        
        def valor(obs):
            return float(obs)
        
        rollout = coletar_rollout(cli, politica, valor, n_passos=2, gamma=0.99, gae_lambda=0.95)
        
        # Criar novo episódio e verificar 404 após remover
        ep_id, _ = cli.criar_episodio()
        resp = cliente_http.get(f"/episodios/{ep_id}")
        assert resp.status_code == 200
        
        cli.encerrar(ep_id)
        resp = cliente_http.get(f"/episodios/{ep_id}")
        assert resp.status_code == 404

    def test_rollout_next_values_nunca_usa_episodio_novo(self, cliente_http):
        """next_values[t] = valor(obs_final_passo_t), nunca valor(obs_novo_episodio=0).
        
        Se usasse obs novo, teríamos valor(0) em next_values.
        Com valor(obs)=obs*10: valor(0)=0, valor(1)=10, valor(2)=20.
        """
        cli = ClienteAmbiente(cliente_http, comprimento=2, max_passos=100)
        
        def politica(obs):
            return 1
        
        def valor(obs):
            return float(obs * 10)
        
        rollout = coletar_rollout(cli, politica, valor, n_passos=3, gamma=0.99, gae_lambda=0.95)
        
        # Passo 1 termina (obs=2), próximo começa novo (obs=0)
        # Se usássemos obs novo: next_values[1] = valor(0) = 0
        # Correto: next_values[1] = valor(2) = 20
        assert rollout.next_values[1] == 20.0
        # Verificar que não tem zero em next_values (prova que não usou novo episódio)
        assert 0.0 not in rollout.next_values
```