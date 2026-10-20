"""Serviço REST do ambiente "corredor", com episódios no estilo do Gymnasium.

O agente começa na posição 0 de um corredor de comprimento ``comprimento`` e se
move uma casa por passo (0 = esquerda, 1 = direita; a posição nunca fica abaixo
de 0). Cada passo vale -1; chegar ao fim do corredor vale +10 e termina o
episódio (``terminated``). Atingir ``max_passos`` interrompe o episódio de fora
(``truncated``). As duas coisas podem acontecer no mesmo passo.

Os episódios ficam em memória, um dicionário por instância do app.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field, StrictInt

RECOMPENSA_PASSO = -1.0
RECOMPENSA_OBJETIVO = 10.0


class NovoEpisodio(BaseModel):
    comprimento: StrictInt = Field(ge=2, le=20, description="Posição do objetivo.")
    max_passos: StrictInt = Field(ge=1, le=500, description="Limite de passos antes da truncagem.")


class Acao(BaseModel):
    acao: StrictInt = Field(ge=0, le=1, description="0 = esquerda, 1 = direita.")


class EstadoEpisodio(BaseModel):
    id: str
    observacao: int = Field(description="Posição atual do agente.")
    passos: int
    encerrado: bool


class Transicao(BaseModel):
    observacao: int = Field(description="Posição depois do passo (observação final se o episódio acabou).")
    recompensa: float
    terminated: bool = Field(description="Chegou ao objetivo: não há valor futuro.")
    truncated: bool = Field(description="Atingiu max_passos: o estado seguinte ainda tem valor.")


@dataclass
class _Episodio:
    comprimento: int
    max_passos: int
    posicao: int = 0
    passos: int = 0
    encerrado: bool = False


def criar_app() -> FastAPI:
    app = FastAPI(title="Ambiente corredor", version="1.0.0")
    episodios: dict[str, _Episodio] = {}

    def buscar(episodio_id: str) -> _Episodio:
        episodio = episodios.get(episodio_id)
        if episodio is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"episódio {episodio_id!r} não existe")
        return episodio

    def estado(episodio_id: str, episodio: _Episodio) -> EstadoEpisodio:
        return EstadoEpisodio(
            id=episodio_id, observacao=episodio.posicao, passos=episodio.passos, encerrado=episodio.encerrado
        )

    @app.post("/episodios", status_code=status.HTTP_201_CREATED, response_model=EstadoEpisodio)
    def criar_episodio(corpo: NovoEpisodio, response: Response) -> EstadoEpisodio:
        episodio_id = uuid.uuid4().hex
        episodios[episodio_id] = _Episodio(comprimento=corpo.comprimento, max_passos=corpo.max_passos)
        response.headers["Location"] = f"/episodios/{episodio_id}"
        return estado(episodio_id, episodios[episodio_id])

    @app.get(
        "/episodios/{episodio_id}",
        response_model=EstadoEpisodio,
        responses={404: {"description": "Episódio não existe."}},
    )
    def consultar_episodio(episodio_id: str) -> EstadoEpisodio:
        return estado(episodio_id, buscar(episodio_id))

    @app.post(
        "/episodios/{episodio_id}/passos",
        response_model=Transicao,
        responses={
            404: {"description": "Episódio não existe."},
            409: {"description": "Episódio já encerrado (terminated ou truncated)."},
        },
    )
    def dar_passo(episodio_id: str, corpo: Acao) -> Transicao:
        episodio = buscar(episodio_id)
        if episodio.encerrado:
            raise HTTPException(status.HTTP_409_CONFLICT, f"episódio {episodio_id!r} já foi encerrado")

        episodio.posicao = max(0, episodio.posicao + (1 if corpo.acao == 1 else -1))
        episodio.passos += 1
        terminated = episodio.posicao == episodio.comprimento
        truncated = episodio.passos >= episodio.max_passos
        episodio.encerrado = terminated or truncated
        return Transicao(
            observacao=episodio.posicao,
            recompensa=RECOMPENSA_OBJETIVO if terminated else RECOMPENSA_PASSO,
            terminated=terminated,
            truncated=truncated,
        )

    @app.delete(
        "/episodios/{episodio_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={404: {"description": "Episódio não existe."}},
    )
    def remover_episodio(episodio_id: str) -> Response:
        buscar(episodio_id)
        del episodios[episodio_id]
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return app
