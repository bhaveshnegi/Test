from fastapi import FastAPI
from pydantic import BaseModel

from app.rag_chain import ask_question


app = FastAPI(
    title="Simple RAG API",
)


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "Server Running"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    result = ask_question(
        request.question
    )

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"]
    }