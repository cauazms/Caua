import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from prompts import obter_chat_prompt_template, carregar_prompt

# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.getenv("MODEL", "openai/gpt-oss-20b")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))
PROMPT_VERSION = os.getenv("PROMPT_VERSION", None)

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
K_RETRIEVAL = 5

# ============================================================
# VALIDAÇÕES INICIAIS
# ============================================================

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY não foi encontrada no arquivo .env.")

INDEX_FAISS = VECTORSTORE_DIR / "index.faiss"
INDEX_PKL = VECTORSTORE_DIR / "index.pkl"

if not INDEX_FAISS.exists() or not INDEX_PKL.exists():
    raise FileNotFoundError(
        f"Índices FAISS não localizados em: {VECTORSTORE_DIR}. "
        "Execute primeiro: python build_vectorstore.py"
    )

# ============================================================
# RECURSOS (LLM E VECTORSTORE)
# ============================================================

@st.cache_resource
def carregar_modelo():
    return ChatOpenAI(
        model=MODEL,
        temperature=TEMPERATURE,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

@st.cache_resource
def carregar_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

llm = carregar_modelo()
vectorstore = carregar_vectorstore()

# ============================================================
# CADEIAS (HYDE + RESPOSTA FINAL)
# ============================================================

versao_ativa, _ = carregar_prompt(PROMPT_VERSION)
hyde_prompt = obter_chat_prompt_template(versao_ativa)
hyde_chain = hyde_prompt | llm | StrOutputParser()

answer_prompt = ChatPromptTemplate.from_template(
    """
Você é um assistente especializado na documentação do PI-V.

Sua resposta deve ser baseada EXCLUSIVAMENTE nos documentos recuperados pelo sistema.

REGRAS:
1. Responda à pergunta original do usuário.
2. Utilize somente as informações presentes no contexto.
3. NÃO invente informações.
4. NÃO utilize conhecimento externo.
5. NÃO utilize internet.
6. Se o contexto não possuir informação suficiente, diga claramente:
"A informação não foi encontrada na documentação disponibilizada."
7. Não trate a hipótese HyDE como fonte de informação. Ela serve apenas para localizar documentos.
8. Quando houver informações relevantes no contexto, explique de maneira clara e objetiva.
9. Se houver conflito entre documentos, informe que existe uma divergência.

Pergunta original:
{user_input}

Contexto recuperado dos documentos:
{context}

Histórico da conversa:
{history}

Resposta:
"""
)

answer_chain = answer_prompt | llm | StrOutputParser()

# ============================================================
# FUNÇÕES DE EXECUÇÃO
# ============================================================

def buscar_documentos_hyde(pergunta, k=5):
    inicio_hyde = time.perf_counter()
    hipotese = hyde_chain.invoke({"user_input": pergunta}).strip()
    tempo_hyde = time.perf_counter() - inicio_hyde

    if not hipotese:
        raise ValueError("O modelo não gerou uma hipótese HyDE.")

    inicio_faiss = time.perf_counter()
    documentos = vectorstore.similarity_search(hipotese, k=k)
    tempo_faiss = time.perf_counter() - inicio_faiss

    return documentos, hipotese, tempo_hyde, tempo_faiss

def formatar_contexto(documentos):
    partes = []
    for i, doc in enumerate(documentos, start=1):
        texto = doc.page_content.strip()
        fonte = doc.metadata.get("source", "Documento não identificado")
        partes.append(f"DOCUMENTO {i}\nFonte: {fonte}\n\n{texto}\n")
    return "\n".join(partes)

def formatar_historico(historico):
    if not historico:
        return "Nenhuma conversa anterior."
    partes = []
    for pergunta, resposta in historico[-5:]:
        partes.append(f"Usuário: {pergunta}\nAssistente: {resposta}")
    return "\n\n".join(partes)

def gerar_resposta(pergunta, documentos, historico):
    contexto = formatar_contexto(documentos)
    historico_texto = formatar_historico(historico)

    inicio_resposta = time.perf_counter()
    resposta = answer_chain.invoke({
        "user_input": pergunta,
        "context": contexto,
        "history": historico_texto
    }).strip()
    tempo_resposta = time.perf_counter() - inicio_resposta

    return resposta, tempo_resposta, contexto

def executar_consulta(pergunta, historico):
    inicio_total = time.perf_counter()
    documentos, hipotese, tempo_hyde, tempo_faiss = buscar_documentos_hyde(pergunta, k=K_RETRIEVAL)
    resposta, tempo_resposta, contexto = gerar_resposta(pergunta, documentos, historico)
    tempo_total = time.perf_counter() - inicio_total

    return {
        "pergunta": pergunta,
        "hipotese": hipotese,
        "documentos": documentos,
        "contexto": contexto,
        "resposta": resposta,
        "tempo_hyde": tempo_hyde,
        "tempo_faiss": tempo_faiss,
        "tempo_resposta": tempo_resposta,
        "tempo_total": tempo_total
    }

def calcular_precision_recall_mrr(documentos, fontes_relevantes):
    if not fontes_relevantes:
        return None, None, None

    fontes_relevantes_set = {
        Path(str(f)).name.lower() for f in fontes_relevantes if str(f).strip()
    }

    relevantes_encontrados = 0
    primeira_posicao = None

    for pos, doc in enumerate(documentos[:K_RETRIEVAL], start=1):
        nome_fonte = Path(str(doc.metadata.get("source", ""))).name.lower()
        if nome_fonte in fontes_relevantes_set:
            relevantes_encontrados += 1
            if primeira_posicao is None:
                primeira_posicao = pos

    qtd_recuperada = min(len(documentos), K_RETRIEVAL)
    precision = relevantes_encontrados / qtd_recuperada if qtd_recuperada > 0 else 0
    recall = relevantes_encontrados / len(fontes_relevantes_set) if fontes_relevantes_set else 0
    recall = min(recall, 1.0)
    mrr = 1 / primeira_posicao if primeira_posicao is not None else 0

    return precision, recall, mrr

# ============================================================
# INTERFACE STREAMLIT
# ============================================================

st.set_page_config(page_title="Chatbot PI-V (Spark + HyDE)", page_icon="⚡", layout="centered")

if "historico" not in st.session_state:
    st.session_state.historico = []
if "ultima_consulta" not in st.session_state:
    st.session_state.ultima_consulta = None
if "avaliacoes" not in st.session_state:
    st.session_state.avaliacoes = []

st.title("⚡ Chatbot PI-V")
st.caption(f"HyDE + FAISS + PySpark | Versão do Prompt: `{versao_ativa}`")

with st.sidebar:
    st.header("Sistema")
    st.write(f"**Modelo:** {MODEL}")
    st.write("**Ingestão:** Apache Spark (PySpark)")
    st.write("**Banco Vetorial:** FAISS")
    st.write(f"**Top-K:** {K_RETRIEVAL}")
    st.write(f"**Embeddings:** {EMBEDDING_MODEL}")
    st.divider()
    if st.button("Limpar conversa", use_container_width=True):
        st.session_state.historico = []
        st.session_state.ultima_consulta = None
        st.rerun()

aba_chat, aba_avaliacao = st.tabs(["💬 Chatbot", "📊 Avaliação"])

with aba_chat:
    for msg in st.session_state.historico:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if pergunta := st.chat_input("Digite sua pergunta sobre a documentação..."):
        with st.chat_message("user"):
            st.write(pergunta)

        try:
            with st.spinner("Consultando documentação indexada via Spark..."):
                # Monta histórico emparelhado corretamente
                pares_historico = []
                msgs = st.session_state.historico
                for i in range(0, len(msgs) - 1, 2):
                    if msgs[i]["role"] == "user" and msgs[i+1]["role"] == "assistant":
                        pares_historico.append((msgs[i]["content"], msgs[i+1]["content"]))

                resultado = executar_consulta(pergunta, pares_historico)

            st.session_state.historico.append({"role": "user", "content": pergunta})
            st.session_state.historico.append({"role": "assistant", "content": resultado["resposta"]})
            st.session_state.ultima_consulta = resultado

            with st.chat_message("assistant"):
                st.write(resultado["resposta"])
                with st.expander("🔎 Ver detalhes da recuperação"):
                    st.markdown("**Hipótese HyDE gerada:**")
                    st.code(resultado["hipotese"])
                    st.markdown("**Documentos recuperados (FAISS):**")
                    for i, doc in enumerate(resultado["documentos"], start=1):
                        fonte = doc.metadata.get("source", "Desconhecido")
                        st.write(f"{i}. {fonte}")

                    st.markdown("**Latência:**")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("HyDE", f"{resultado['tempo_hyde']:.3f}s")
                    c2.metric("FAISS", f"{resultado['tempo_faiss']:.3f}s")
                    c3.metric("Resposta", f"{resultado['tempo_resposta']:.3f}s")
                    c4.metric("Total", f"{resultado['tempo_total']:.3f}s")

        except Exception as erro:
            st.error(f"{type(erro).__name__}: {erro}")

with aba_avaliacao:
    st.header("📊 Avaliação de Recuperação e Assertividade")
    pergunta_teste = st.text_input("Pergunta de teste:")
    fontes_teste = st.text_input("Documento(s) esperado(s) (separados por ponto e vírgula):")

    if st.button("Executar avaliação", type="primary"):
        if not pergunta_teste.strip():
            st.warning("Insira uma pergunta.")
        else:
            try:
                with st.spinner("Executando teste RAG..."):
                    res = executar_consulta(pergunta_teste, [])

                st.markdown("**Resposta:**")
                st.write(res["resposta"])

                fontes_relevantes = [f.strip() for f in fontes_teste.split(";") if f.strip()]
                precision, recall, mrr = calcular_precision_recall_mrr(res["documentos"], fontes_relevantes)

                if precision is not None:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Precision@5", f"{precision:.2%}")
                    c2.metric("Recall@5", f"{recall:.2%}")
                    c3.metric("MRR", f"{mrr:.2%}")
            except Exception as e:
                st.error(f"Erro: {e}")
