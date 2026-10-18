"""Carrega os prompts versionados de prompts.yaml."""

from pathlib import Path
from string import Formatter
import yaml
from langchain_core.prompts import ChatPromptTemplate

PROMPTS_FILE = Path(__file__).parent / "prompts.yaml"

def carregar_prompt(versao=None):
    """Devolve (versao, template) do prompts.yaml."""
    if not PROMPTS_FILE.exists():
        raise SystemExit(f"Arquivo de prompts não encontrado: {PROMPTS_FILE}")

    dados = yaml.safe_load(PROMPTS_FILE.read_text(encoding="utf-8")) or {}
    prompts = dados.get("prompts") or {}

    if not prompts:
        raise SystemExit(f"Nenhum prompt definido em {PROMPTS_FILE.name}.")

    versao = versao or dados.get("default")
    if not versao:
        raise SystemExit(f"Defina PROMPT_VERSION no .env ou a chave default em {PROMPTS_FILE.name}.")

    if versao not in prompts:
        disponiveis = ", ".join(sorted(prompts))
        raise SystemExit(f"Versão de prompt '{versao}' não existe em {PROMPTS_FILE.name}.\nDisponíveis: {disponiveis}")

    template = (prompts[versao] or {}).get("template")
    if not template:
        raise SystemExit(f"A versão '{versao}' não tem o campo `template`.")

    return versao, template

def obter_chat_prompt_template(versao=None) -> ChatPromptTemplate:
    """Retorna ChatPromptTemplate configurado com o template versionado."""
    _, template_str = carregar_prompt(versao)
    return ChatPromptTemplate.from_template(template_str)
