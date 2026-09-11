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


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1"
)

MODEL = os.getenv(
    "MODEL",
    "openai/gpt-oss-20b"
)

TEMPERATURE = float(
    os.getenv("TEMPERATURE", "0.5")
)

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

VECTORSTORE_DIR = BASE_DIR / "vectorstore"

K_RETRIEVAL = 5


# ============================================================
# VALIDAÇÕES INICIAIS
# ============================================================

if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY não foi encontrada no arquivo .env."
    )

INDEX_FAISS = VECTORSTORE_DIR / "index.faiss"
INDEX_PKL = VECTORSTORE_DIR / "index.pkl"

if not INDEX_FAISS.exists():
    raise FileNotFoundError(
        f"Arquivo index.faiss não encontrado em: {INDEX_FAISS}"
    )

if not INDEX_PKL.exists():
    raise FileNotFoundError(
        f"Arquivo index.pkl não encontrado em: {INDEX_PKL}"
    )


# ============================================================
# MODELO
# ============================================================

@st.cache_resource
def carregar_modelo():
    return ChatOpenAI(
        model=MODEL,
        temperature=TEMPERATURE,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )


# ============================================================
# EMBEDDINGS + FAISS
# ============================================================

@st.cache_resource
def carregar_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


llm = carregar_modelo()
vectorstore = carregar_vectorstore()


# ============================================================
# HYDE
# ============================================================

hyde_prompt = ChatPromptTemplate.from_template(
    """
Você é o componente HyDE de um sistema RAG.

Sua única função é criar uma pequena hipótese documental
para auxiliar uma busca semântica em documentos sobre PI-V.

NÃO responda à pergunta do usuário.

NÃO converse com o usuário.

NÃO explique sua resposta.

NÃO invente números, valores, percentuais, artigos de lei,
datas, fórmulas ou informações específicas que não estejam
presentes na pergunta.

Use somente os conceitos e termos identificáveis diretamente
na pergunta.

Transforme a pergunta em um pequeno trecho técnico,
contendo palavras-chave e conceitos que provavelmente
apareceriam na documentação relacionada ao assunto.

O texto deve ser curto e objetivo.

A saída será usada SOMENTE como consulta para o mecanismo
de recuperação semântica.

Pergunta do usuário:

{user_input}

Gere somente a hipótese documental.
"""
)

hyde_chain = (
    hyde_prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# PROMPT DA RESPOSTA FINAL
# ============================================================

answer_prompt = ChatPromptTemplate.from_template(
    """
Você é um assistente especializado na documentação do PI-V.

Sua resposta deve ser baseada EXCLUSIVAMENTE
nos documentos recuperados pelo sistema.

REGRAS:

1. Responda à pergunta original do usuário.

2. Utilize somente as informações presentes no contexto.

3. NÃO invente informações.

4. NÃO utilize conhecimento externo.

5. NÃO utilize internet.

6. Se o contexto não possuir informação suficiente,
diga claramente:

"A informação não foi encontrada na documentação
disponibilizada."

7. Não trate a hipótese HyDE como fonte de informação.
Ela serve apenas para localizar documentos.

8. Quando houver informações relevantes no contexto,
explique de maneira clara e objetiva.

9. Se houver conflito entre documentos,
informe que existe uma divergência.

Pergunta original:

{user_input}

Contexto recuperado dos documentos:

{context}

Histórico da conversa:

{history}

Resposta:
"""
)

answer_chain = (
    answer_prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# FUNÇÃO DE RECUPERAÇÃO HYDE + FAISS
# ============================================================

def buscar_documentos_hyde(pergunta, k=5):

    # --------------------------------------------------------
    # ETAPA 1 — HYDE
    # --------------------------------------------------------

    inicio_hyde = time.perf_counter()

    hipotese = hyde_chain.invoke({
        "user_input": pergunta
    }).strip()

    tempo_hyde = time.perf_counter() - inicio_hyde

    if not hipotese:
        raise ValueError(
            "O modelo não gerou uma hipótese HyDE."
        )

    # --------------------------------------------------------
    # ETAPA 2 — FAISS
    # --------------------------------------------------------

    inicio_faiss = time.perf_counter()

    documentos = vectorstore.similarity_search(
        hipotese,
        k=k
    )

    tempo_faiss = time.perf_counter() - inicio_faiss

    return (
        documentos,
        hipotese,
        tempo_hyde,
        tempo_faiss
    )


# ============================================================
# FORMATAÇÃO DO CONTEXTO
# ============================================================

def formatar_contexto(documentos):

    partes = []

    for i, documento in enumerate(
        documentos,
        start=1
    ):

        texto = documento.page_content.strip()

        fonte = documento.metadata.get(
            "source",
            "Documento não identificado"
        )

        partes.append(
            f"""
DOCUMENTO {i}
Fonte: {fonte}

{texto}
"""
        )

    return "\n".join(partes)


# ============================================================
# HISTÓRICO
# ============================================================

def formatar_historico(historico):

    if not historico:
        return "Nenhuma conversa anterior."

    partes = []

    for pergunta, resposta in historico[-5:]:

        partes.append(
            f"Usuário: {pergunta}\n"
            f"Assistente: {resposta}"
        )

    return "\n\n".join(partes)


# ============================================================
# GERAÇÃO DA RESPOSTA
# ============================================================

def gerar_resposta(
    pergunta,
    documentos,
    historico
):

    contexto = formatar_contexto(
        documentos
    )

    historico_texto = formatar_historico(
        historico
    )

    inicio_resposta = time.perf_counter()

    resposta = answer_chain.invoke({
        "user_input": pergunta,
        "context": contexto,
        "history": historico_texto
    }).strip()

    tempo_resposta = (
        time.perf_counter()
        - inicio_resposta
    )

    return (
        resposta,
        tempo_resposta,
        contexto
    )


# ============================================================
# CONSULTA COMPLETA
# ============================================================

def executar_consulta(
    pergunta,
    historico
):

    inicio_total = time.perf_counter()

    (
        documentos,
        hipotese,
        tempo_hyde,
        tempo_faiss
    ) = buscar_documentos_hyde(
        pergunta,
        k=K_RETRIEVAL
    )

    (
        resposta,
        tempo_resposta,
        contexto
    ) = gerar_resposta(
        pergunta,
        documentos,
        historico
    )

    tempo_total = (
        time.perf_counter()
        - inicio_total
    )

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


# ============================================================
# MÉTRICAS DA METODOLOGIA DE AVALIAÇÃO
# ============================================================

def calcular_precision_recall_mrr(
    documentos,
    fontes_relevantes
):

    if not fontes_relevantes:
        return None, None, None

    fontes_relevantes = {
        Path(str(fonte))
        .name
        .lower()
        for fonte in fontes_relevantes
        if str(fonte).strip()
    }

    relevantes_encontrados = 0
    primeira_posicao = None

    for posicao, documento in enumerate(
        documentos[:K_RETRIEVAL],
        start=1
    ):

        fonte = documento.metadata.get(
            "source",
            ""
        )

        nome_fonte = (
            Path(str(fonte))
            .name
            .lower()
        )

        if nome_fonte in fontes_relevantes:

            relevantes_encontrados += 1

            if primeira_posicao is None:
                primeira_posicao = posicao

    quantidade_recuperada = min(
        len(documentos),
        K_RETRIEVAL
    )

    precision = (
        relevantes_encontrados
        / quantidade_recuperada
        if quantidade_recuperada > 0
        else 0
    )

    recall = (
        relevantes_encontrados
        / len(fontes_relevantes)
        if fontes_relevantes
        else 0
    )

    recall = min(recall, 1.0)

    mrr = (
        1 / primeira_posicao
        if primeira_posicao is not None
        else 0
    )

    return precision, recall, mrr


# ============================================================
# CONFIGURAÇÃO STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Chatbot PI-V",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# ESTADO DA APLICAÇÃO
# ============================================================

if "historico" not in st.session_state:
    st.session_state.historico = []

if "ultima_consulta" not in st.session_state:
    st.session_state.ultima_consulta = None

if "avaliacoes" not in st.session_state:
    st.session_state.avaliacoes = []


# ============================================================
# CABEÇALHO
# ============================================================

st.title("🤖 Chatbot PI-V")

st.caption(
    "Assistente de consulta à documentação do PI-V "
    "utilizando HyDE + RAG + FAISS."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Sistema")

    st.write(
        f"**Modelo:** {MODEL}"
    )

    st.write(
        "**Provedor:** OpenRouter"
    )

    st.write(
        "**Banco vetorial:** FAISS"
    )

    st.write(
        f"**Top-K:** {K_RETRIEVAL}"
    )

    st.write(
        f"**Embeddings:** {EMBEDDING_MODEL}"
    )

    st.divider()

    st.subheader("Arquitetura")

    st.code(
        """Pergunta
   ↓
HyDE
   ↓
Embedding
   ↓
FAISS
   ↓
Documentos
   ↓
LLM
   ↓
Resposta"""
    )

    if st.button(
        "Limpar conversa",
        use_container_width=True
    ):

        st.session_state.historico = []
        st.session_state.ultima_consulta = None

        st.rerun()


# ============================================================
# ABAS
# ============================================================

aba_chat, aba_avaliacao = st.tabs(
    [
        "💬 Chatbot",
        "📊 Avaliação"
    ]
)


# ============================================================
# ABA CHATBOT
# ============================================================

with aba_chat:

    # --------------------------------------------------------
    # HISTÓRICO VISUAL
    # --------------------------------------------------------

    for mensagem in st.session_state.historico:

        papel = mensagem["role"]
        conteudo = mensagem["content"]

        with st.chat_message(papel):
            st.write(conteudo)

    # --------------------------------------------------------
    # ENTRADA
    # --------------------------------------------------------

    pergunta = st.chat_input(
        "Digite sua pergunta sobre o PI-V..."
    )

    if pergunta:

        with st.chat_message("user"):
            st.write(pergunta)

        try:

            with st.spinner(
                "Consultando a documentação..."
            ):

                resultado = executar_consulta(
                    pergunta,
                    [
                        (
                            item["question"],
                            item["answer"]
                        )
                        for item in st.session_state.historico
                        if item["role"] == "assistant"
                    ]
                )

            resposta = resultado["resposta"]

            st.session_state.historico.append({
                "role": "user",
                "content": pergunta
            })

            st.session_state.historico.append({
                "role": "assistant",
                "content": resposta
            })

            st.session_state.ultima_consulta = resultado

            with st.chat_message("assistant"):

                st.write(resposta)

                with st.expander(
                    "🔎 Ver detalhes da consulta"
                ):

                    st.markdown(
                        "**Hipótese HyDE utilizada na recuperação:**"
                    )

                    st.code(
                        resultado["hipotese"]
                    )

                    st.markdown(
                        "**Documentos recuperados pelo FAISS:**"
                    )

                    for i, documento in enumerate(
                        resultado["documentos"],
                        start=1
                    ):

                        fonte = documento.metadata.get(
                            "source",
                            "Documento não identificado"
                        )

                        st.write(
                            f"{i}. {fonte}"
                        )

                    st.markdown(
                        "**Latência:**"
                    )

                    c1, c2, c3, c4 = st.columns(4)

                    c1.metric(
                        "HyDE",
                        f'{resultado["tempo_hyde"]:.3f}s'
                    )

                    c2.metric(
                        "FAISS",
                        f'{resultado["tempo_faiss"]:.3f}s'
                    )

                    c3.metric(
                        "Resposta",
                        f'{resultado["tempo_resposta"]:.3f}s'
                    )

                    c4.metric(
                        "Total",
                        f'{resultado["tempo_total"]:.3f}s'
                    )

        except Exception as erro:

            st.error(
                f"{type(erro).__name__}: {erro}"
            )


# ============================================================
# ABA AVALIAÇÃO
# ============================================================

with aba_avaliacao:

    st.header(
        "📊 Metodologia de avaliação"
    )

    st.markdown(
        """
O chatbot será avaliado em três dimensões:

### 1. Recuperação

**Precision@5**  
Proporção dos documentos recuperados que são relevantes.

**Recall@5**  
Proporção dos documentos relevantes que foram recuperados.

**MRR (Mean Reciprocal Rank)**  
Avalia a posição do primeiro documento relevante.

### 2. Qualidade da resposta

A resposta deve ser avaliada pelo responsável pelo projeto
em uma escala de **1 a 5**:

**1 — Incorreta**  
A resposta não responde à pergunta ou apresenta informação incorreta.

**2 — Pouco adequada**  
Possui muitos problemas ou informações insuficientes.

**3 — Parcialmente adequada**  
Responde parcialmente à pergunta.

**4 — Adequada**  
Responde corretamente e está de acordo com a documentação.

**5 — Excelente**  
Resposta correta, completa, clara e totalmente fundamentada
nos documentos recuperados.

Também devem ser observadas:

- Fidelidade ao contexto;
- Relevância da resposta;
- Ausência de informações inventadas.

### 3. Desempenho

São registrados automaticamente:

- Tempo de geração da hipótese HyDE;
- Tempo de recuperação no FAISS;
- Tempo de geração da resposta;
- Tempo total da consulta.
"""
    )

    st.divider()

    st.subheader(
        "Avaliar uma pergunta"
    )

    pergunta_avaliacao = st.text_input(
        "Pergunta de teste",
        placeholder=(
            "Digite uma pergunta cuja resposta esteja "
            "na documentação do PI-V"
        )
    )

    fontes_texto = st.text_input(
        "Documento(s) relevante(s), opcional",
        placeholder=(
            "Ex.: ICMS_SP_material_resumido_com_formulas_legislacao (1).docx"
        )
    )

    executar_avaliacao = st.button(
        "Executar avaliação",
        type="primary"
    )

    if executar_avaliacao:

        if not pergunta_avaliacao.strip():

            st.warning(
                "Digite uma pergunta de teste."
            )

        else:

            try:

                with st.spinner(
                    "Executando o teste..."
                ):

                    resultado = executar_consulta(
                        pergunta_avaliacao,
                        []
                    )

                st.subheader(
                    "Resultado"
                )

                st.markdown(
                    "**Resposta gerada:**"
                )

                st.write(
                    resultado["resposta"]
                )

                st.markdown(
                    "**Documentos recuperados:**"
                )

                for i, documento in enumerate(
                    resultado["documentos"],
                    start=1
                ):

                    fonte = documento.metadata.get(
                        "source",
                        "Documento não identificado"
                    )

                    st.write(
                        f"{i}. {fonte}"
                    )

                # ------------------------------------------------
                # MÉTRICAS DE RECUPERAÇÃO
                # ------------------------------------------------

                fontes_relevantes = [
                    fonte.strip()
                    for fonte in fontes_texto.split(";")
                    if fonte.strip()
                ]

                (
                    precision,
                    recall,
                    mrr
                ) = calcular_precision_recall_mrr(
                    resultado["documentos"],
                    fontes_relevantes
                )

                if precision is not None:

                    st.subheader(
                        "Métricas de recuperação"
                    )

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "Precision@5",
                        f"{precision:.2%}"
                    )

                    c2.metric(
                        "Recall@5",
                        f"{recall:.2%}"
                    )

                    c3.metric(
                        "MRR",
                        f"{mrr:.2%}"
                    )

                else:

                    st.info(
                        "Para calcular Precision@5, Recall@5 e MRR, "
                        "informe o nome do documento relevante."
                    )

                # ------------------------------------------------
                # AVALIAÇÃO MANUAL
                # ------------------------------------------------

                st.subheader(
                    "Avaliação da resposta"
                )

                nota = st.radio(
                    "Nota de assertividade",
                    options=[
                        1,
                        2,
                        3,
                        4,
                        5
                    ],
                    horizontal=True
                )

                st.caption(
                    "1 = incorreta | "
                    "2 = pouco adequada | "
                    "3 = parcial | "
                    "4 = adequada | "
                    "5 = excelente"
                )

                fidelidade = st.radio(
                    "A resposta está fiel aos documentos recuperados?",
                    [
                        "Sim",
                        "Parcialmente",
                        "Não"
                    ],
                    horizontal=True
                )

                relevancia = st.radio(
                    "A resposta responde diretamente à pergunta?",
                    [
                        "Sim",
                        "Parcialmente",
                        "Não"
                    ],
                    horizontal=True
                )

                comentario = st.text_area(
                    "Observação do avaliador",
                    placeholder=(
                        "Descreva brevemente por que a resposta "
                        "recebeu essa avaliação."
                    )
                )

                salvar = st.button(
                    "Registrar avaliação"
                )

                if salvar:

                    st.session_state.avaliacoes.append({
                        "nota": nota,
                        "fidelidade": fidelidade,
                        "relevancia": relevancia,
                        "latencia_total": resultado[
                            "tempo_total"
                        ]
                    })

                    st.success(
                        "Avaliação registrada nesta sessão."
                    )

            except Exception as erro:

                st.error(
                    f"{type(erro).__name__}: {erro}"
                )

    # --------------------------------------------------------
    # RESUMO DAS AVALIAÇÕES DA SESSÃO
    # --------------------------------------------------------

    if st.session_state.avaliacoes:

        st.divider()

        st.subheader(
            "Resumo das avaliações"
        )

        avaliacoes = (
            st.session_state.avaliacoes
        )

        media_nota = sum(
            item["nota"]
            for item in avaliacoes
        ) / len(avaliacoes)

        media_latencia = sum(
            item["latencia_total"]
            for item in avaliacoes
        ) / len(avaliacoes)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Testes realizados",
            len(avaliacoes)
        )

        c2.metric(
            "Nota média",
            f"{media_nota:.2f}/5"
        )

        c3.metric(
            "Latência média",
            f"{media_latencia:.3f}s"
        )


# ============================================================
# RODAPÉ
# ============================================================

st.divider()

st.caption(
    "PI-V • HyDE + RAG + FAISS • Interface web mínima"
)
