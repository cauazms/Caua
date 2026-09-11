"""Carrega os prompts versionados de prompts.yaml."""

from pathlib import Path
from string import Formatter

import yaml


PROMPTS_FILE = Path(__file__).parent / "prompts.yaml"


def carregar_prompt(versao=None):
    """
    Devolve (versao, template) do prompts.yaml.

    Se versao for None, usa a chave default do arquivo.
    """

    if not PROMPTS_FILE.exists():
        raise SystemExit(
            f"Arquivo de prompts não encontrado: {PROMPTS_FILE}"
        )

    dados = yaml.safe_load(
        PROMPTS_FILE.read_text(encoding="utf-8")
    ) or {}

    prompts = dados.get("prompts") or {}

    if not prompts:
        raise SystemExit(
            f"Nenhum prompt definido em {PROMPTS_FILE.name}."
        )

    versao = versao or dados.get("default")

    if not versao:
        raise SystemExit(
            f"Defina PROMPT_VERSION no .env ou "
            f"a chave `default` em {PROMPTS_FILE.name}."
        )

    if versao not in prompts:
        disponiveis = ", ".join(sorted(prompts))

        raise SystemExit(
            f"Versão de prompt '{versao}' não existe em "
            f"{PROMPTS_FILE.name}.\n"
            f"Disponíveis: {disponiveis}"
        )

    template = (prompts[versao] or {}).get("template")

    if not template:
        raise SystemExit(
            f"A versão '{versao}' não tem o campo `template`."
        )

    # Verifica os placeholders existentes no YAML
    campos = {
        nome
        for _, nome, _, _ in Formatter().parse(template)
        if nome
    }

    return versao, template