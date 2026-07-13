from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = "vector_db"

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)


def retrieve_documents(query: str, k: int = 3):

    results = vectorstore.similarity_search_with_score(
        query,
        k=k
    )

    return results