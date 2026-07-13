import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


from langchain_chroma import Chroma


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    os.path.dirname(CURRENT_DIR),
    "data",
    "sample-doc-long.txt"
)

CHROMA_PATH = os.path.join(
    os.path.dirname(CURRENT_DIR),
    "vector_db"
)


print("Loading TXT file...")

loader = TextLoader(DATA_FILE, encoding="utf-8")
documents = loader.load()

print(f"Loaded {len(documents)} document(s)")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


print("Loading Embedding Model...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)


print("Creating ChromaDB...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PATH
)

print("Ingestion Completed")