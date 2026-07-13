from langchain_ollama import ChatOllama


def get_llm():
    return ChatOllama(
        model="qwen2.5:0.5b",
        temperature=0
    )