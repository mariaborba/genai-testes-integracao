"""Monta um prompt, envia a uma LLM isolada e avalia a suíte gerada.

Uso: uv run python verificacao/executar.py <prompts/arquivo.md> <rotulo> [--modelo haiku]

O .md do prompt tem um bloco ```text sob "## System" e outro sob "## User". Marcadores
{OPENAPI}, {INTERFACE}, {CODIGO_AMBIENTE}, {CODIGO_COLETOR} e {CODIGO_TESTES_ANTERIOR}/{FALHAS}
são preenchidos com os arquivos do projeto. A LLM roda via `claude -p` numa pasta vazia, sem
ferramentas (não lê nem executa nada) e sem sessão: cada chamada é independente, como numa API.

Saída em execucoes/<rotulo>/: prompt enviado, resposta bruta, test_<rotulo>.py extraído, meta.json
(modelo, tokens, custo), pytest.txt, cobertura.txt e mutacoes.txt.
"""
import argparse
import json
import pathlib
import re
import subprocess
import tempfile

RAIZ = pathlib.Path(__file__).resolve().parent.parent


def bloco(md: str, titulo: str) -> str:
    m = re.search(rf"## {titulo}\s*\n```text\n(.*?)\n```", md, re.S)
    if not m:
        raise SystemExit(f"bloco ```text ausente sob '## {titulo}'")
    return m.group(1)


def preencher(texto: str, extras: dict[str, str]) -> str:
    valores = {
        "OPENAPI": (RAIZ / "prompts/contexto/openapi.json").read_text(),
        "INTERFACE": (RAIZ / "prompts/contexto/interface-coletor.py").read_text(),
        "CODIGO_AMBIENTE": (RAIZ / "ambiente/app.py").read_text(),
        "CODIGO_COLETOR": (RAIZ / "coletor/rollout.py").read_text(),
        **extras,
    }
    for chave, valor in valores.items():
        texto = texto.replace("{" + chave + "}", valor)
    return texto


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("rotulo")
    p.add_argument("--modelo", default="haiku")
    p.add_argument("--anterior", help="rótulo de uma execução anterior (prompts de correção)")
    a = p.parse_args()

    extras = {}
    if a.anterior:
        ant = RAIZ / "execucoes" / a.anterior
        extras["CODIGO_TESTES_ANTERIOR"] = (ant / f"test_{a.anterior}.py").read_text()
        extras["FALHAS"] = (ant / "falhas.txt").read_text()

    md = (RAIZ / a.prompt).read_text()
    system, user = preencher(bloco(md, "System"), extras), preencher(bloco(md, "User"), extras)
    destino = RAIZ / "execucoes" / a.rotulo
    destino.mkdir(parents=True, exist_ok=False)
    (destino / "prompt-enviado.txt").write_text(f"[system]\n{system}\n\n[user]\n{user}\n")

    with tempfile.TemporaryDirectory() as vazio:
        r = subprocess.run(
            ["claude", "-p", "--safe-mode", "--tools", "", "--no-session-persistence",
             "--model", a.modelo, "--system-prompt", system, "--output-format", "json"],
            input=user, capture_output=True, text=True, cwd=vazio, timeout=900,
        )
    dados = json.loads(r.stdout)
    resposta = dados["result"]
    (destino / "resposta.md").write_text(resposta)
    (destino / "meta.json").write_text(json.dumps({
        "prompt": a.prompt, "anterior": a.anterior,
        "modelos": list(dados.get("modelUsage", {})),
        "tokens_entrada": dados["usage"]["input_tokens"] + dados["usage"].get("cache_read_input_tokens", 0)
        + dados["usage"].get("cache_creation_input_tokens", 0),
        "tokens_saida": dados["usage"]["output_tokens"],
        "custo_usd": dados.get("total_cost_usd"),
    }, indent=1, ensure_ascii=False))

    blocos = re.findall(r"```python\n(.*?)```", resposta, re.S)
    if not blocos:
        raise SystemExit("resposta sem bloco ```python")
    teste = destino / f"test_{a.rotulo}.py"
    teste.write_text(max(blocos, key=len))
    avaliar(teste, destino)


def avaliar(teste: pathlib.Path, destino: pathlib.Path) -> None:
    uv = ["uv", "run", "--project", str(RAIZ)]
    py = subprocess.run(uv + ["pytest", str(teste), "-q", "-rfE", "--no-header", "-p", "no:cacheprovider",
                              "-o", "testpaths=", "--rootdir", str(RAIZ)],
                        cwd=RAIZ, capture_output=True, text=True)
    (destino / "pytest.txt").write_text(py.stdout + py.stderr)
    falhas = re.findall(r"^(?:FAILED|ERROR) .*$", py.stdout, re.M)
    (destino / "falhas.txt").write_text("\n".join(falhas) + "\n")
    cov = subprocess.run(uv + ["pytest", str(teste), "-q", "--no-header", "-p", "no:cacheprovider", "-o", "testpaths=",
                               "--rootdir", str(RAIZ), "--cov=ambiente", "--cov=coletor", "--cov-branch",
                               "--cov-report=term-missing"],
                         cwd=RAIZ, capture_output=True, text=True)
    (destino / "cobertura.txt").write_text(cov.stdout)
    # Com falhas, o escore mede só os testes que passam (a alternativa 2 da Unidade III).
    mut = subprocess.run(uv + ["python", "verificacao/mutacoes.py", str(teste), str(destino / "falhas.txt")],
                         cwd=RAIZ, capture_output=True, text=True)
    (destino / "mutacoes.txt").write_text(mut.stdout + mut.stderr)
    print(py.stdout.strip().splitlines()[-1] if py.stdout.strip() else py.stderr[-500:])


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["--reavaliar"]:
        rotulo = sys.argv[2]
        avaliar(RAIZ / "execucoes" / rotulo / f"test_{rotulo}.py", RAIZ / "execucoes" / rotulo)
    else:
        main()
