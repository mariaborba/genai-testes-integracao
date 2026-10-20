"""Tabela das execuções a partir dos arquivos em execucoes/. Uso: uv run python verificacao/resumo.py"""
import json
import pathlib
import re

RAIZ = pathlib.Path(__file__).resolve().parent.parent
print("| Execução | Prompt | Testes | Passaram | Falharam | Cobertura (linhas+decisões) | Mutantes mortos | Custo (US$) |")
print("|---|---|---|---|---|---|---|---|")
for pasta in sorted(p for p in (RAIZ / "execucoes").iterdir() if p.is_dir()):
    meta = json.loads((pasta / "meta.json").read_text())
    prompt = pathlib.Path(meta["prompt"]).stem
    if not (pasta / "pytest.txt").exists():
        print(f"| {pasta.name} | {prompt} | 0 | 0 | 0 | — | — | {meta['custo_usd']:.3f} |")
        continue
    ultima = (pasta / "pytest.txt").read_text().strip().splitlines()[-1]
    passou = int(m.group(1)) if (m := re.search(r"(\d+) passed", ultima)) else 0
    falhou = int(m.group(1)) if (m := re.search(r"(\d+) failed", ultima)) else 0
    total = re.search(r"^TOTAL.*?(\d+)%$", (pasta / "cobertura.txt").read_text(), re.M).group(1)
    escore = re.search(r"escore: (\d+/\d+)", (pasta / "mutacoes.txt").read_text()).group(1)
    print(f"| {pasta.name} | {prompt} | {passou + falhou} | {passou} | {falhou} | {total}% | {escore} | {meta['custo_usd']:.3f} |")
