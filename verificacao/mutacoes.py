"""Quebra uma regra de integração por vez numa cópia do projeto e confere se uma suíte falha.

Uso: uv run python verificacao/mutacoes.py <arquivo de teste> [falhas.txt] [<outro teste> <falhas.txt> ...]

Com falhas.txt (linhas "FAILED <nodeid>"), os testes que já falham sem mutação ficam de fora.
Várias suítes juntas medem a união (complementaridade entre prompts).
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

RAIZ = pathlib.Path(__file__).resolve().parent.parent
APP, COLETOR = "ambiente/app.py", "coletor/rollout.py"
MUTACOES = {
    # lado servidor: o contrato HTTP
    "S1 criação devolve 200": (APP, "status_code=status.HTTP_201_CREATED, response_model", "status_code=200, response_model"),
    "S2 passo após encerrar": (APP, "if episodio.encerrado:", "if False:"),
    "S3 id inexistente vira 400": (APP, "raise HTTPException(status.HTTP_404_NOT_FOUND", "raise HTTPException(400"),
    "S4 nunca trunca": (APP, "truncated = episodio.passos >= episodio.max_passos", "truncated = False"),
    "S5 trunca um passo tarde": (APP, "truncated = episodio.passos >= episodio.max_passos", "truncated = episodio.passos > episodio.max_passos"),
    "S6 objetivo sem +10": (APP, "recompensa=RECOMPENSA_OBJETIVO if terminated", "recompensa=RECOMPENSA_PASSO if terminated"),
    "S7 aceita acao 2": (APP, 'acao: StrictInt = Field(ge=0, le=1,', 'acao: StrictInt = Field(ge=0, le=2,'),
    "S8 DELETE não remove": (APP, "        del episodios[episodio_id]\n", ""),
    "S9 truncado esconde terminado": (APP, "terminated = episodio.posicao == episodio.comprimento", "terminated = episodio.posicao == episodio.comprimento and episodio.passos < episodio.max_passos"),
    # lado cliente: o coletor
    "C1 next_value do episódio novo": (
        COLETOR,
        "            episodio_id, observacao = cliente.criar_episodio()\n",
        "            episodio_id, observacao = cliente.criar_episodio()\n            next_values[-1] = valor(observacao)\n",
    ),
    "C2 truncado vira terminado": (COLETOR, "terminated.append(transicao.terminated)", "terminated.append(transicao.terminated or transicao.truncated)"),
    "C3 fim do rollout não trunca": (COLETOR, "        truncated[-1] = True\n", "        pass\n"),
    "C4 4xx passa em silêncio": (COLETOR, "if resposta.status_code >= 400:", "if resposta.status_code >= 500:"),
    "C5 aceita bool em texto": (COLETOR, "    if not isinstance(valor, tipo):\n", "    if not isinstance(valor, tipo) and not isinstance(valor, str):\n"),
    "C6 falha de rede escapa crua": (
        COLETOR,
        "        try:\n            resposta = self._http.request(metodo, caminho, **kwargs)\n",
        "        resposta = self._http.request(metodo, caminho, **kwargs)\n        try:\n            pass\n",
    ),
    "C7 não remove episódio acabado": (COLETOR, "            cliente.encerrar(episodio_id)\n            episodio_id", "            episodio_id"),
    "C8 não reinicia episódio": (COLETOR, "and t < n_passos - 1:", "and False:"),
    "C9 ignora status de sucesso": (COLETOR, "if resposta.status_code != esperado:", "if False:"),
}
args = sys.argv[1:]
pares = [(pathlib.Path(args[i]).resolve(), args[i + 1] if i + 1 < len(args) else None) for i in range(0, len(args), 2)]
with tempfile.TemporaryDirectory() as pasta:
    copia = pathlib.Path(pasta) / "projeto"
    shutil.copytree(RAIZ, copia, ignore=shutil.ignore_patterns(".venv", "__pycache__", ".pytest_cache", "tests", "verificacao", "prompts", "execucoes", ".git"))
    (copia / "tests").mkdir()
    fora = []
    for suite, falhas in pares:
        shutil.copy(suite, copia / "tests" / suite.name)
        if falhas:
            for linha in pathlib.Path(falhas).read_text().split("\n"):
                if linha.startswith(("FAILED ", "ERROR ")):
                    fora += ["--deselect", "tests/" + linha.split()[1].split("/")[-1]]
    base = subprocess.run(["uv", "run", "--project", str(RAIZ), "pytest", "-q", "-p", "no:cacheprovider", *fora], cwd=copia, capture_output=True, text=True)
    print(f"sem mutação: {base.stdout.strip().splitlines()[-1]}")
    mortos = 0
    for nome, (arquivo, trecho, mutante) in MUTACOES.items():
        alvo = copia / arquivo
        original = alvo.read_text()
        assert original.count(trecho) == 1, f"trecho ausente ou ambíguo: {nome}"
        alvo.write_text(original.replace(trecho, mutante, 1))
        r = subprocess.run(["uv", "run", "--project", str(RAIZ), "pytest", "-q", "-x", "-p", "no:cacheprovider", *fora], cwd=copia, capture_output=True, text=True)
        alvo.write_text(original)
        morto = r.returncode != 0
        mortos += morto
        print(("PEGOU   " if morto else "ESCAPOU ") + nome)
    print(f"escore: {mortos}/{len(MUTACOES)}")
