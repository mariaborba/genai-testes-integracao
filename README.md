# Testes de integração com GenAI: serviço REST de ambiente × coletor de rollouts

Experimento do fórum da Unidade III (Automação de Testes com IA Generativa). Dois tipos de prompt
geram testes de integração com uma LLM; cada suíte gerada é executada e medida por cobertura e por
um escore de mutação com mutantes de integração escritos **antes** da primeira geração.

Todos os dados são fictícios.

## O cenário de integração

```
coletor/rollout.py  ──HTTP──▶  ambiente/app.py (FastAPI)
  ClienteAmbiente                POST   /episodios              201 · 422
  coletar_rollout ──▶ rl/gae.py  GET    /episodios/{id}         200 · 404
                                 POST   /episodios/{id}/passos  200 · 404 · 409 · 422
                                 DELETE /episodios/{id}         204 · 404
```

- **`ambiente/`**: serviço REST de um ambiente de RL no estilo Gymnasium: um corredor em que cada
  passo vale -1 e chegar ao fim vale +10 (`terminated`); ao atingir `max_passos`, o episódio é
  interrompido (`truncated`). As duas coisas podem acontecer no mesmo passo.
- **`coletor/`**: o cliente. Fala com o serviço por HTTP, confere o contrato de cada resposta,
  reinicia episódios, monta os arrays e chama `compute_gae` (`rl/gae.py`, reaproveitado dos fóruns
  anteriores).
- **A regra que só existe na integração:** quando um episódio é truncado, `next_values` tem de usar a
  observação **final** dele, não a do episódio novo que o coletor cria em seguida. Cada componente
  pode estar certo isoladamente e o rollout sair errado.

## Os dois tipos de prompt

| | P1: contrato (caixa-preta) | P2: falhas na fronteira (caixa-branca + CoT) |
|---|---|---|
| Inspiração na Unidade III | Figura 28 (OpenAPI no prompt) | Figuras 29 e 11 (código no prompt, passo a passo) |
| Contexto | especificação OpenAPI + interface pública do coletor | código do serviço e do coletor |
| Objetivo | cobrir a especificação, todos os status e o fluxo do coletor | erros HTTP, timeout, contrato violado, regras de estado do episódio |
| Arquivos | [v1](prompts/P1-contrato-v1.md) · [v2](prompts/P1-contrato-v2.md) · [v3](prompts/P1-contrato-v3.md) | [v1](prompts/P2-falhas-v1.md) · [v2](prompts/P2-falhas-v2.md) · [v3](prompts/P2-falhas-v3.md) |

Cada versão traz no topo o que mudou e por quê. O texto exato enviado à LLM em cada execução está em
`execucoes/<rotulo>/prompt-enviado.txt`.

## Como a LLM foi executada

- **Modelo:** Claude Haiku 4.5 (`claude-haiku-4-5-20251001`), o mesmo usado nos exemplos da apostila.
- **Isolamento:** `claude -p` numa pasta vazia, **sem ferramentas** (não lê arquivos nem executa
  código) e sem sessão. O papel vai no *system prompt*, e a tarefa e o contexto vão na mensagem do
  usuário, como numa chamada de API (Figura 17).
- **Sem correção manual:** o arquivo de teste é o maior bloco ```python da resposta, salvo sem nenhuma
  edição.
- **Medição:** cobertura de linhas e decisões com a suíte inteira. O escore de mutação usa só os
  testes que passam sem mutação (a alternativa 2 da seção 3.7: descartar os que falham).

## Resultados

| Execução | Prompt | Testes | Passaram | Falharam | Cobertura | Mutantes mortos | Custo (US$) |
|---|---|---|---|---|---|---|---|
| p1v1 | P1 v1 | 55 | 47 | 8 | 95% | 7/18 | 0,119 |
| p1v1b | P1 v1 | 59 | 54 | 5 | 96% | 10/18 | 0,144 |
| p1v2a | P1 v2 | 57 | 51 | 6 | 96% | 12/18 | 0,131 |
| p1v2b | P1 v2 | 56 | 46 | 10 | 95% | 11/18 | 0,119 |
| p1v3a | P1 v3 | 69 | 66 | 3 | 96% | 14/18 | 0,168 |
| p1v3b | P1 v3 | — | — | — | — | resposta cortada | 0,259 |
| p1v3c | P1 v3 | — | — | — | — | resposta cortada | 0,240 |
| p2v1 | P2 v1 | 42 | 41 | 1 | 99% | 13/18 | 0,129 |
| p2v1b | P2 v1 | 45 | 41 | 4 | 99% | 10/18 | 0,111 |
| **p2v2a** | **P2 v2** | **36** | **36** | **0** | **98%** | **16/18** | **0,101** |
| **p2v2b** | **P2 v2** | **41** | **41** | **0** | **99%** | **14/18** | **0,123** |
| p2v3a | P2 v3 | 26 | 25 | 1 | 96% | 11/18 | 0,134 |
| p2v3b | P2 v3 | 36 | 35 | 1 | 98% | 14/18 | 0,078 |

**União P1 v2 + P2 v2** (mesma rodada): 17/18 (a) e 15/18 (b), contra 16 e 14 do melhor prompt
sozinho. Arquivos em `execucoes/uniao-*.txt`. **Custo total:** US$ 1,86 em 13 chamadas.

A tabela é gerada por `uv run python verificacao/resumo.py`.

### Mutantes (`verificacao/mutacoes.py`)

São 18, escritos antes da primeira geração. A suíte de referência, feita à mão
(`verificacao/test_referencia.py`), mata os 18; ela não entra no experimento e só prova que todos são
detectáveis.

- **Servidor:** S1 criação devolve 200 · S2 aceita passo após encerrar · S3 id inexistente vira 400 ·
  S4 nunca trunca · S5 trunca um passo tarde · S6 objetivo sem +10 · S7 aceita ação 2 ·
  S8 DELETE não remove · S9 truncagem esconde o término.
- **Coletor:** C1 `next_values` do episódio novo · C2 truncado vira terminado · C3 fim do rollout não
  trunca · C4 4xx passa em silêncio · C5 aceita bool em texto · C6 falha de rede escapa crua ·
  C7 não remove o episódio acabado · C8 não reinicia o episódio · C9 ignora o status de sucesso.

### Por que os testes falharam

Cada uma das 39 falhas foi lida. Nenhuma revelou defeito no sistema; todas são erro do teste gerado.

| Causa | Falhas | Exemplo |
|---|---|---|
| `rollout.terminated[i] is True` (`np.bool_` não é `True`) | 14 | surgiu na v2 e continuou na v3, mesmo com instrução explícita |
| Valor fora do schema (`comprimento=100`, limite 20) | 8 | o teste esperava 201 e recebeu 422 |
| API do httpx mal usada | 6 | `httpx.TimeoutException()` sem a mensagem obrigatória |
| Regra do ambiente inventada | 4 | recompensas ≥ 0; último passo terminado também marcado como truncado |
| Teste incompleto ou com erro de código | 4 | o comentário diz "deletamos o episódio", mas o teste não deleta |
| Cálculo à mão errado (off-by-one) | 2 | término esperado no passo 2; acontece no passo 1 |
| Asserção sobre o texto da mensagem | 1 | `"tipo inválido"`, quando a mensagem real é outra |

### O que os números mostram

1. **A cobertura não separou as suítes.** Ela ficou entre 95% e 99% em todas, enquanto o escore de
   mutação foi de 7 a 16. O P1 v1 cobriu 100% do serviço e matou só 7 mutantes.
2. **Um nome de teste não garante que a regra foi exercitada.** Em p2v1,
   `test_rollout_next_values_usa_observacao_transicao` roda 3 passos num corredor de 5 e o episódio
   nunca acaba, por isso o C1 sobrevive. Em p2v2b o cenário já cruzava o fim do episódio, mas
   `valor = lambda obs: 0.0` deixava o oráculo cego.
3. **Cada tipo de prompt achou coisas diferentes.** O P1 v2 recebeu a regra "terminated e truncated
   podem ser true no mesmo passo" em texto e matou o S9. O P2, que tinha o código, deixou o S9 vivo nas
   quatro execuções da v2 e da v3. Juntar as suítes deu o melhor escore.
4. **Mais instrução nem sempre ajudou.** A v3 do P1 gerou respostas que estouraram o limite de saída
   em 2 de 3 execuções (US$ 0,50 sem nenhum teste aproveitável). A v3 do P2 voltou a ter falhas e um
   escore médio menor que o da v2. A instrução 12, específica demais, foi aplicada ao objeto errado
   (`FastAPI` em vez de `TestClient`).
5. **O C7 é quase invisível.** Esse mutante não remove os episódios acabados no serviço, e só uma das
   11 suítes geradas o matou. Verificar isso exige consultar o estado do servidor depois do rollout.

**Viés a declarar:** os ajustes da v2 e da v3 foram guiados pelos mutantes que escaparam. O escore
dessas versões mede o quanto o prompt corrigiu falhas conhecidas, não a capacidade de achar defeitos
novos.

## Como reproduzir

```bash
uv sync
uv run pytest verificacao/test_referencia.py                # suíte de referência: 18 testes
uv run python verificacao/mutacoes.py verificacao/test_referencia.py   # 18/18
uv run python verificacao/executar.py prompts/P2-falhas-v2.md novo_rotulo  # nova geração (requer o CLI `claude`)
uv run python verificacao/executar.py --reavaliar p2v2a     # reexecuta só a avaliação
```

## Estrutura

```
ambiente/app.py            serviço REST (sistema em teste)
coletor/rollout.py         cliente + coleta + GAE (sistema em teste)
rl/gae.py                  compute_gae, reaproveitado dos fóruns 1 e 2
prompts/                   os dois tipos de prompt, v1 → v3, e o contexto (OpenAPI, interface)
execucoes/<rotulo>/        por execução: prompt enviado, resposta bruta, test_<rotulo>.py gerado,
                           meta.json (modelo, tokens, custo), pytest.txt, cobertura.txt, mutacoes.txt
verificacao/               executor, mutantes, suíte de referência, tabela de resumo
```
