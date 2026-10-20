"""Coleta de rollouts no serviço REST do ambiente e cálculo das vantagens com GAE.

O coletor é o lado cliente da integração: fala com o serviço por HTTP, confere
se cada resposta respeita o contrato e monta os arrays que ``compute_gae``
espera. A regra que mais importa na fronteira é a de ``terminated``/``truncated``:
em transição truncada, ``next_values`` usa a observação final do episódio, nunca
a observação do episódio novo criado em seguida.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import httpx
import numpy as np

from rl.gae import compute_gae


class ErroAmbiente(Exception):
    """O serviço respondeu com erro HTTP ou não respondeu (timeout, conexão).

    ``status`` é o código HTTP recebido, ou ``None`` se não houve resposta.
    """

    def __init__(self, mensagem: str, status: int | None) -> None:
        super().__init__(mensagem)
        self.status = status


class ErroContrato(Exception):
    """O serviço respondeu com sucesso, mas o corpo não segue o contrato."""


@dataclass(frozen=True)
class Transicao:
    observacao: int
    recompensa: float
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class Rollout:
    observacoes: np.ndarray
    acoes: np.ndarray
    recompensas: np.ndarray
    values: np.ndarray
    next_values: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


class ClienteAmbiente:
    """Cliente do serviço do ambiente. Recebe o ``httpx.Client`` já configurado (base_url, timeout)."""

    def __init__(self, http: httpx.Client, comprimento: int, max_passos: int) -> None:
        self._http = http
        self._comprimento = comprimento
        self._max_passos = max_passos

    def criar_episodio(self) -> tuple[str, int]:
        """Cria um episódio e devolve ``(id, observacao_inicial)``."""
        corpo = self._requisitar(
            "POST", "/episodios", 201, json={"comprimento": self._comprimento, "max_passos": self._max_passos}
        )
        return _campo(corpo, "id", str), _campo(corpo, "observacao", int)

    def passo(self, episodio_id: str, acao: int) -> Transicao:
        corpo = self._requisitar("POST", f"/episodios/{episodio_id}/passos", 200, json={"acao": acao})
        return Transicao(
            observacao=_campo(corpo, "observacao", int),
            recompensa=float(_campo(corpo, "recompensa", (int, float))),
            terminated=_campo(corpo, "terminated", bool),
            truncated=_campo(corpo, "truncated", bool),
        )

    def encerrar(self, episodio_id: str) -> None:
        self._requisitar("DELETE", f"/episodios/{episodio_id}", 204)

    def _requisitar(self, metodo: str, caminho: str, esperado: int, **kwargs) -> dict | None:
        try:
            resposta = self._http.request(metodo, caminho, **kwargs)
        except httpx.TimeoutException as erro:
            raise ErroAmbiente(f"{metodo} {caminho}: tempo esgotado", status=None) from erro
        except httpx.TransportError as erro:
            raise ErroAmbiente(f"{metodo} {caminho}: falha de conexão ({erro})", status=None) from erro

        if resposta.status_code >= 400:
            raise ErroAmbiente(
                f"{metodo} {caminho}: HTTP {resposta.status_code}: {resposta.text[:200]}", status=resposta.status_code
            )
        if resposta.status_code != esperado:
            raise ErroContrato(f"{metodo} {caminho}: esperado HTTP {esperado}, recebido {resposta.status_code}")
        if esperado == 204:
            return None
        try:
            corpo = resposta.json()
        except ValueError as erro:
            raise ErroContrato(f"{metodo} {caminho}: corpo não é JSON") from erro
        if not isinstance(corpo, dict):
            raise ErroContrato(f"{metodo} {caminho}: corpo deveria ser objeto JSON, recebido {type(corpo).__name__}")
        return corpo


def _campo(corpo: dict, nome: str, tipo: type | tuple[type, ...]):
    if nome not in corpo:
        raise ErroContrato(f"campo {nome!r} ausente na resposta")
    valor = corpo[nome]
    # bool é subclasse de int em Python: sem esta exclusão, true passaria como observação 1.
    if isinstance(valor, bool) and tipo is not bool:
        raise ErroContrato(f"campo {nome!r} deveria ser numérico, recebido bool")
    if not isinstance(valor, tipo):
        raise ErroContrato(f"campo {nome!r} com tipo inválido: {type(valor).__name__}")
    return valor


def coletar_rollout(
    cliente: ClienteAmbiente,
    politica: Callable[[int], int],
    valor: Callable[[int], float],
    n_passos: int,
    gamma: float,
    gae_lambda: float,
) -> Rollout:
    """Coleta ``n_passos`` transições, reiniciando o episódio quando ele acaba, e calcula o GAE.

    - Quando um episódio acaba (``terminated`` ou ``truncated``), o coletor o remove do
      serviço e cria outro, exceto no último passo do rollout.
    - ``next_values[t] = valor(observacao devolvida pelo passo t)``, inclusive quando o
      episódio acabou: é a observação final, não a inicial do episódio seguinte.
    - O último passo é marcado como ``truncated`` se não terminou, porque o rollout o
      interrompe de fora.
    - Ao final, o episódio em aberto é removido do serviço.

    Raises:
        ValueError: se ``n_passos`` não for inteiro >= 1, ou pelas regras de ``compute_gae``.
        ErroAmbiente: se o serviço responder erro HTTP ou não responder.
        ErroContrato: se uma resposta de sucesso violar o contrato.
    """
    if isinstance(n_passos, bool) or not isinstance(n_passos, int) or n_passos < 1:
        raise ValueError(f"n_passos deve ser inteiro >= 1; recebido {n_passos!r}")

    observacoes, acoes, recompensas, values, next_values, terminated, truncated = [], [], [], [], [], [], []
    episodio_id, observacao = cliente.criar_episodio()
    for t in range(n_passos):
        acao = politica(observacao)
        transicao = cliente.passo(episodio_id, acao)

        observacoes.append(observacao)
        acoes.append(acao)
        recompensas.append(transicao.recompensa)
        values.append(valor(observacao))
        next_values.append(valor(transicao.observacao))
        terminated.append(transicao.terminated)
        truncated.append(transicao.truncated)

        if (transicao.terminated or transicao.truncated) and t < n_passos - 1:
            cliente.encerrar(episodio_id)
            episodio_id, observacao = cliente.criar_episodio()
        else:
            observacao = transicao.observacao

    if not terminated[-1]:
        truncated[-1] = True
    cliente.encerrar(episodio_id)

    arrays = dict(
        observacoes=np.array(observacoes, dtype=np.int64),
        acoes=np.array(acoes, dtype=np.int64),
        recompensas=np.array(recompensas, dtype=np.float64),
        values=np.array(values, dtype=np.float64),
        next_values=np.array(next_values, dtype=np.float64),
        terminated=np.array(terminated, dtype=bool),
        truncated=np.array(truncated, dtype=bool),
    )
    advantages, returns = compute_gae(
        arrays["recompensas"],
        arrays["values"],
        arrays["next_values"],
        arrays["terminated"],
        arrays["truncated"],
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    return Rollout(**arrays, advantages=advantages, returns=returns)
