# P2 v1 — Casos de erro e falhas na fronteira (caixa-branca, chain-of-thought)

**Tipo de prompt:** baseado em código e em falhas, no estilo da Figura 29 (código-fonte no contexto)
combinado com chain-of-thought (Figura 11). O alvo não é cobrir a especificação, e sim o que quebra
quando os dois componentes conversam: erros HTTP, timeout, contrato violado e regras de estado do episódio.

## System

```text
Você é um engenheiro de testes experiente, especializado em testes de integração, injeção de falhas e cobertura de decisões.
```

## User

```text
Crie testes de integração caixa-branca em pytest focados em casos de erro e falhas na fronteira entre
o coletor de rollouts e o serviço REST do ambiente. O código dos dois componentes está delimitado por *****.

Pense passo a passo antes de escrever o código:
1. Liste cada chamada HTTP que o coletor faz e o que ele espera receber.
2. Para cada chamada, liste como ela pode falhar: erro HTTP 4xx ou 5xx, timeout, falha de conexão,
   corpo fora do contrato (campo ausente, tipo errado, status de sucesso diferente do esperado).
3. Liste as regras de estado do episódio que só aparecem quando os dois componentes estão integrados.
4. Para cada item das listas, escreva um teste cujo nome diga qual falha ou regra está sendo exercitada.
Escreva o raciocínio em no máximo 15 linhas e depois um único bloco ```python com o arquivo completo.

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

*****
# ambiente/app.py
{CODIGO_AMBIENTE}

# coletor/rollout.py
{CODIGO_COLETOR}
*****
```
