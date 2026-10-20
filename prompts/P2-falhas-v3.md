# P2 v3 — Casos de erro e falhas na fronteira (caixa-branca, chain-of-thought)

**Ajustes em relação à v2** (ver execucoes/p1v2* e p2v2*): instruções 10 a 12. A instrução 8 da v2
fez o modelo comparar elemento a elemento com `is True`, o que falha com `np.bool_` (11 das 34 falhas);
em p2v2b o cenário cruzava o fim do episódio, mas `valor` constante deixava o mutante C1 vivo; e
nenhuma suíte verificou a remoção dos episódios que o próprio rollout criou (C7 vivo nas 6 execuções).
O resto do prompt é igual à v2.

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
3. Liste as entradas inválidas que o serviço deve recusar com 422 (limites e tipos de cada campo).
4. Liste as regras de estado do episódio que só aparecem quando os dois componentes estão integrados,
   inclusive efeitos colaterais que só se veem consultando o serviço depois (GET de episódios removidos).
5. Para cada regra do item 4, escolha `comprimento`, `max_passos`, política e `n_passos` de modo que a
   condição aconteça de fato dentro do rollout, e calcule à mão os arrays esperados.
6. Para cada item das listas, escreva um teste cujo nome diga qual falha ou regra está sendo exercitada.
Escreva o raciocínio em no máximo 25 linhas e depois um único bloco ```python com o arquivo completo.

Siga estas instruções no código gerado:
1. Um único arquivo pytest, Python 3.13. Use apenas pytest, httpx, numpy, a biblioteca padrão e
   `from fastapi.testclient import TestClient`. Importe o sistema com `from ambiente.app import criar_app`
   e `from coletor.rollout import ClienteAmbiente, ErroAmbiente, ErroContrato, Transicao, Rollout, coletar_rollout`.
2. O serviço real roda em memória: crie uma fixture que devolve `TestClient(criar_app())` novo a cada teste.
   O TestClient é um `httpx.Client`, então pode ser passado direto para `ClienteAmbiente(http, comprimento, max_passos)`.
3. Sem rede, sem `time.sleep`, sem servidor externo e sem dependência entre testes.
4. Para falhas que o serviço real não produz (HTTP 5xx, timeout, falha de conexão, corpo fora do contrato),
   use `httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ambiente")`.
5. Nomes de teste em português, no formato test_<o_que>_<condicao>.
6. Exceções do httpx exigem mensagem e request: `raise httpx.ReadTimeout("tempo esgotado", request=request)`,
   `raise httpx.ConnectError("recusada", request=request)`.
7. Respeite os limites do schema: `comprimento` entre 2 e 20 e `max_passos` entre 1 e 500. Valores fora
   desses limites só aparecem em testes que esperam HTTP 422.
8. Oráculos: compare com valores literais que você calculou à mão (arrays inteiros, não só tamanho ou dtype).
   Erros do coletor: confira o tipo da exceção e `status`, nunca o texto da mensagem.
9. Todo teste de regra de fronteira de episódio precisa de um cenário em que o episódio realmente termina
   ou trunca DENTRO do rollout. Escreva num comentário em qual passo isso acontece.
10. Arrays do Rollout são numpy: compare com `np.testing.assert_array_equal(rollout.terminated, [False, True, False])`
    ou `.tolist() == [...]`. Nunca `rollout.terminated[i] is True` (np.bool_ não é o objeto True).
11. Em teste de next_values, use uma função `valor` que dê valores diferentes para cada observação
    (por exemplo, `valor = lambda obs: float(obs)`); com valor constante, o oráculo não distingue a
    observação final da observação do episódio seguinte.
12. Efeito colateral no serviço: guarde os ids que o PRÓPRIO rollout criou (envolva o `request` do
    TestClient com uma função que registra as respostas de POST /episodios) e confira com GET que todos
    foram removidos. Não crie um episódio novo para provar a remoção.

*****
# ambiente/app.py
{CODIGO_AMBIENTE}

# coletor/rollout.py
{CODIGO_COLETOR}
*****
```
