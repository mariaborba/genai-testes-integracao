# coletor/rollout.py — interface pública (corpos omitidos)

class ErroAmbiente(Exception):
    """O serviço respondeu com erro HTTP ou não respondeu (timeout, conexão).

    ``status`` é o código HTTP recebido, ou ``None`` se não houve resposta.
    """
    status: int | None

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
    observacoes: np.ndarray   # int64
    acoes: np.ndarray         # int64
    recompensas: np.ndarray   # float64
    values: np.ndarray        # float64
    next_values: np.ndarray   # float64
    terminated: np.ndarray    # bool
    truncated: np.ndarray     # bool
    advantages: np.ndarray    # float64
    returns: np.ndarray       # float64

class ClienteAmbiente:
    """Cliente do serviço do ambiente. Recebe o ``httpx.Client`` já configurado (base_url, timeout)."""
    def __init__(self, http: httpx.Client, comprimento: int, max_passos: int) -> None: ...
    def criar_episodio(self) -> tuple[str, int]:
        """Cria um episódio e devolve ``(id, observacao_inicial)``."""
    def passo(self, episodio_id: str, acao: int) -> Transicao: ...
    def encerrar(self, episodio_id: str) -> None: ...

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

# rl/gae.py (já testado em separado)
def compute_gae(rewards, values, next_values, terminated, truncated, gamma, gae_lambda) -> tuple[np.ndarray, np.ndarray]:
    """delta_t = rewards[t] + gamma * next_values[t] * (1 - terminated[t]) - values[t]
    advantages[t] = delta_t + gamma * gae_lambda * (1 - fim_t) * advantages[t+1], fim_t = terminated[t] or truncated[t]
    advantages[T-1] = delta_{T-1};  returns = advantages + values"""
