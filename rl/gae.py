"""Generalized Advantage Estimation (GAE) para um rollout de um único ambiente.

Função pura: não lê arquivos, não acessa rede, não usa relógio nem aleatoriedade,
e não altera os arrays recebidos.
"""

from __future__ import annotations

import numbers

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula as vantagens GAE e os retornos de um rollout em ordem temporal.

    O índice ``t`` refere-se à transição ``(s_t, a_t, r_t, s_{t+1})``:

    - ``delta_t = rewards[t] + gamma * next_values[t] * (1 - terminated[t]) - values[t]``
      (estado terminal não tem valor futuro; estado truncado tem, então usa
      ``next_values[t]`` normalmente);
    - ``advantages[t] = delta_t + gamma * gae_lambda * (1 - fim_t) * advantages[t+1]``,
      com ``fim_t = terminated[t] or truncated[t]``, para que a soma não atravesse
      fronteira de episódio;
    - no último passo do rollout, ``advantages[T-1] = delta_{T-1}``;
    - ``returns = advantages + values``.

    Args:
        rewards: recompensas ``r_t``, array 1-D de comprimento ``T >= 1``.
        values: ``V(s_t)`` estimado pelo crítico, mesmo comprimento.
        next_values: ``V(s_{t+1})``; em transição truncada, valor da observação
            final do episódio (antes do reset), já calculado por quem chama.
        terminated: flags de término em estado terminal do MDP, dtype booleano.
        truncated: flags de interrupção externa (limite de tempo, fim do rollout),
            dtype booleano.
        gamma: fator de desconto, em ``[0, 1]``.
        gae_lambda: fator de suavização do GAE, em ``[0, 1]``.

    Returns:
        Tupla ``(advantages, returns)``, ambos arrays 1-D float64 de comprimento ``T``.

    Raises:
        ValueError: se ``gamma`` ou ``gae_lambda`` não for um número real ou
            estiver fora de ``[0, 1]``; se algum array não for 1-D, não for
            conversível, tiver comprimento divergente de ``rewards`` ou ``T == 0``;
            se ``rewards``, ``values`` ou ``next_values`` contiver valor não
            finito (NaN ou ``±inf``); se ``terminated`` ou ``truncated`` não tiver
            dtype booleano.
    """
    gamma = _validate_unit_interval("gamma", gamma)
    gae_lambda = _validate_unit_interval("gae_lambda", gae_lambda)

    rewards = _validate_finite_float_array("rewards", rewards)
    values = _validate_finite_float_array("values", values)
    next_values = _validate_finite_float_array("next_values", next_values)
    terminated = _validate_bool_array("terminated", terminated)
    truncated = _validate_bool_array("truncated", truncated)

    steps = rewards.shape[0]
    if steps == 0:
        raise ValueError(
            "rewards deve ter comprimento T >= 1; recebido comprimento 0 "
            "(shape (0,))"
        )
    _validate_same_length_as_rewards("values", values, steps)
    _validate_same_length_as_rewards("next_values", next_values, steps)
    _validate_same_length_as_rewards("terminated", terminated, steps)
    _validate_same_length_as_rewards("truncated", truncated, steps)

    # Cópias float64 derivadas das flags: nada é escrito nos arrays de entrada.
    bootstrap = 1.0 - terminated.astype(np.float64)
    continues = 1.0 - (terminated | truncated).astype(np.float64)
    deltas = rewards + gamma * next_values * bootstrap - values

    advantages = np.empty(steps, dtype=np.float64)
    decay = gamma * gae_lambda
    # Recursão de trás para frente; no último passo o acumulado é zero, logo
    # advantages[T-1] == deltas[T-1].
    running = 0.0
    for t in range(steps - 1, -1, -1):
        running = deltas[t] + decay * continues[t] * running
        advantages[t] = running

    returns = advantages + values
    return advantages, returns


def _validate_unit_interval(name: str, value: object) -> float:
    """Exige um número real em [0, 1] e devolve-o convertido para float."""
    if not isinstance(value, numbers.Real):
        # Texto não é convertido silenciosamente: "0.5" indicaria erro de quem chama.
        raise ValueError(
            f"{name} deve ser um número real em [0, 1]; recebido tipo "
            f"{type(value).__name__}: {value!r}"
        )
    # Comparação e mensagem usam o valor como recebido (int, float ou escalar
    # NumPy), não o convertido, para o chamador reconhecer o que passou. NaN
    # falha nas duas comparações, portanto também é rejeitado aqui.
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} deve estar em [0, 1]; recebido: {value!r}")
    return float(value)


def _validate_finite_float_array(name: str, value: object) -> np.ndarray:
    """Converte para array 1-D float64 e exige que todos os valores sejam finitos."""
    array = _as_1d_array(name, value, np.float64)
    non_finite_positions = np.flatnonzero(~np.isfinite(array))
    if non_finite_positions.size > 0:
        # None dentro de uma sequência chega aqui como NaN e cai nesta regra.
        first_index = int(non_finite_positions[0])
        raise ValueError(
            f"{name} deve conter apenas valores finitos; recebido "
            f"{array[first_index]!r} no índice {first_index}"
        )
    return array


def _validate_bool_array(name: str, value: object) -> np.ndarray:
    """Exige um array 1-D de flags com dtype booleano."""
    array = _as_1d_array(name, value, dtype=None)
    if array.dtype != np.bool_:
        # Converter 0/1 em flag esconderia argumento trocado ou array errado:
        # terminated e truncated são semânticas distintas, não grandezas numéricas.
        raise ValueError(
            f"{name} deve ter dtype booleano; recebido dtype {array.dtype}"
        )
    return array


def _as_1d_array(name: str, value: object, dtype: type | None) -> np.ndarray:
    """Converte o valor para array 1-D, no dtype pedido quando houver um."""
    target_dtype = "" if dtype is None else f" de {np.dtype(dtype)}"
    try:
        # Quando o valor já é um array compatível, asarray não copia e o array
        # devolvido é o próprio do chamador. É seguro: nada neste módulo escreve
        # nos arrays validados; advantages, returns e os intermediários são novos.
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} não pôde ser convertido para array 1-D{target_dtype}; "
            f"recebido {_describe_value(value)}"
        ) from exc
    if array.ndim != 1:
        raise ValueError(
            f"{name} deve ser um array 1-D; recebido array com "
            f"{array.ndim} dimensões (shape {array.shape})"
        )
    return array


def _validate_same_length_as_rewards(name: str, array: np.ndarray, steps: int) -> None:
    """Exige que o array tenha o mesmo comprimento T de rewards."""
    if array.shape[0] != steps:
        raise ValueError(
            f"{name} deve ter o mesmo comprimento de rewards ({steps}); "
            f"recebido comprimento {array.shape[0]} (shape {array.shape})"
        )


def _describe_value(value: object) -> str:
    """Descreve o valor recebido por tipo e shape, sem expor o conteúdo do array."""
    try:
        shape = np.shape(value)
    except (TypeError, ValueError):
        # Sequência irregular (sublistas de comprimentos diferentes) não tem shape.
        shape = "indefinido"
    return f"tipo {type(value).__name__} com shape {shape}"
