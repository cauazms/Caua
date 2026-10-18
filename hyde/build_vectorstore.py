import os
import glob
from pathlib import Path
from dotenv import load_dotenv

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
import docx2txt

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DOCS_DIR = BASE_DIR / "documentos"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def init_spark() -> SparkSession:
    """Inicializa a SparkSession local com suporte a processamento distribuído."""
    return (
        SparkSession.builder
        .appName("HyDE-FAISS-Spark-Ingestion")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

def extract_docx(file_path: str) -> str:
    """Extrai texto de arquivo .docx."""
    try:
        return docx2txt.process(file_path)
    except Exception as e:
        print(f"Erro ao extrair {file_path}: {e}")
        return ""

def main():
    print("=" * 60)
    print("Iniciando pipeline de ingestão com Apache Spark")
    print("=" * 60)

    docx_files = glob.glob(str(DOCS_DIR / "*.docx"))
    if not docx_files:
        raise FileNotFoundError(f"Nenhum arquivo .docx encontrado na pasta: {DOCS_DIR}")

    # 1. Leitura dos arquivos brutos
    print(f"Encontrados {len(docx_files)} documentos para processar.")
    records = []
    for f in docx_files:
        text = extract_docx(f)
        if text.strip():
            records.append((Path(f).name, text))

    # 2. Inicialização do Spark
    spark = init_spark()

    schema = StructType([
        StructField("source", StringType(), False),
        StructField("content", StringType(), False)
    ])

    df_docs = spark.createDataFrame(records, schema=schema)

    # 3. Chunking Paralelo distribuído nas partições do Spark
    def process_chunks_partition(rows):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )
        partition_chunks = []
        for row in rows:
            splits = splitter.split_text(row.content)
            for chunk in splits:
                partition_chunks.append((row.source, chunk))
        return partition_chunks

    rdd_chunks = df_docs.rdd.mapPartitions(process_chunks_partition)
    collected_chunks = rdd_chunks.collect()

    print(f"Apache Spark processou {len(collected_chunks)} chunks com sucesso.")
    spark.stop()

    # 4. Vetorização e criação do índice FAISS
    print(f"Carregando modelo de embeddings ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    texts = [item[1] for item in collected_chunks]
    metadatas = [{"source": item[0]} for item in collected_chunks]

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Construindo índice FAISS em: {VECTORSTORE_DIR}...")
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    # Persiste index.faiss e index.pkl
    vectorstore.save_local(str(VECTORSTORE_DIR))
    print("=" * 60)
    print("Índice FAISS gerado e persistido com sucesso via Spark!")
    print("=" * 60)

if __name__ == "__main__":
    main()
