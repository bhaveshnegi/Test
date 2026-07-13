import redis

from app.retriever import retrieve_documents
from app.llm import get_llm


redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

llm = get_llm()


def ask_question(question: str):

    # Cache Check
    cache_key = f"qa:{question}"

    cached_answer = redis_client.get(cache_key)

    if cached_answer:
        return {
            "answer": cached_answer,
            "sources": "cache"
        }

    # Retrieve Documents
    results = retrieve_documents(question)

    if not results:
        return {
            "answer": "I don't know based on the provided documents.",
            "sources": 0
        }

    context = "\n\n".join(
        doc.page_content
        for doc, score in results
    )

    prompt = f"""
You are a helpful assistant.

Use ONLY the provided context.

If the answer is not present in the context,
reply exactly:

"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    answer = response.content

    # Store in Redis (1 hour)
    redis_client.setex(
        cache_key,
        3600,
        answer
    )

    return {
        "answer": answer,
        "sources": len(results)
    }