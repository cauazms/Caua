import os
import glob
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
import docx2txt

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PERSIST_DIRECTORY = "vectorstore"
DOCS_PATH = "documentos"

def init_spark():
    return (
        SparkSession.builder
        .appName("HyDE-Ingestion-Spark")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

def extract_text_from_docx(file_path: str) -> str:
    try:
        return docx2txt.process(file_path)
    except Exception as e:
        print(f"Erro ao ler {file_path}: {e}")
        return ""

def main():
    print("Iniciando ingestao distribuida com Apache Spark...")
    spark = init_spark()

    docx_files = glob.glob(os.path.join(DOCS_PATH, "*.docx"))
    if not docx_files:
        print(f"Nenhum ficheiro .docx encontrado em '{DOCS_PATH}'.")
        return

    records = []
    for f in docx_files:
        content = extract_text_from_docx(f)
        if content.strip():
            records.append((os.path.basename(f), content))

    schema = StructType([
        StructField("filename", StringType(), False),
        StructField("content", StringType(), False)
    ])

    df_raw = spark.createDataFrame(records, schema=schema)

    def chunk_partition(rows):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = []
        for row in rows:
            splits = splitter.split_text(row.content)
            for chunk in splits:
                chunks.append((row.filename, chunk))
        return chunks

    rdd_chunks = df_raw.rdd.mapPartitions(chunk_partition)
    collected_chunks = rdd_chunks.collect()

    print(f"Total de fragmentos processados pelo Spark: {len(collected_chunks)}")

    texts = [item[1] for item in collected_chunks]
    metadatas = [{"source": item[0]} for item in collected_chunks]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print(f"A guardar indices em '{PERSIST_DIRECTORY}'...")
    Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=PERSIST_DIRECTORY
    )

    spark.stop()
    print("Ingestao com Spark concluida!")

if __name__ == "__main__":
    main()
