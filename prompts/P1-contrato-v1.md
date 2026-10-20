# P1 v1 — Contrato da API + fluxo do coletor (caixa-preta)

**Tipo de prompt:** baseado em especificação, no estilo da Figura 28 da Unidade III. A LLM recebe só a
especificação OpenAPI do serviço e a interface pública do coletor (assinaturas e docstrings, sem os
corpos). Papel no system; tarefa, instruções e contexto no user, como numa chamada de API.

## System

```text
Você é um engenheiro de testes experiente, especializado em testes funcionais, testes de API REST e testes de integração.
```

## User

```text
Crie um conjunto de testes de integração caixa-preta em pytest para a integração entre o serviço REST do
ambiente e o coletor de rollouts que o consome. Use a especificação OpenAPI delimitada por ##### para
saber como o serviço se comporta e a interface pública do coletor delimitada por ***** para saber como
o cliente usa o serviço. O conjunto deve cobrir a especificação, todas as respostas possíveis (status
code) e o fluxo do coletor sobre o serviço real. A resposta deve conter apenas código Python.

Siga estas cinco instruções no código gerado:
1. Um único arquivo pytest, Python 3.13. Use apenas pytest, httpx, numpy, a biblioteca padrão e
   `from fastapi.testclient import TestClient`. Importe o sistema com `from ambiente.app import criar_app`
   e `from coletor.rollout import ClienteAmbiente, ErroAmbiente, ErroContrato, Transicao, Rollout, coletar_rollout`.
2. O serviço real roda em memória: crie uma fixture que devolve `TestClient(criar_app())` novo a cada teste.
   O TestClient é um `httpx.Client`, então pode ser passado direto para `ClienteAmbiente(http, comprimento, max_passos)`.
3. Sem rede, sem `time.sleep`, sem servidor externo e sem dependência entre testes.
4. Para falhas que o serviço real não produz (HTTP 5xx, timeout, falha de conexão, corpo fora do contrato),
   use `httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")`.
5. Nomes de teste em português, no formato test_<o_que>_<condicao>.

#####
{OPENAPI}
#####

*****
{INTERFACE}
*****
```
