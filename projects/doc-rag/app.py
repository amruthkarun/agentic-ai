from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from retriever import Retriever

app = FastAPI()
retriever = Retriever()

gen = pipeline("text2text-generation", model="google/flan-t5-small")

class QARequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/qa")
def qa(req: QARequest):
    hits = retriever.retrieve(req.query, req.top_k)

    contexts = []
    for h in hits:
        meta = h["meta"]
        contexts.append(f"{meta['doc_id']} ({meta['chunk_id']})")

    prompt = (
        "Context:\n" +
        "\n".join(contexts) +
        f"\n\nQuestion: {req.query}\nAnswer:"
    )

    out = gen(prompt, max_length=200)
    return {"answer": out[0]["generated_text"], "sources": contexts}

class SummarizeRequest(BaseModel):
    doc_ids: list

@app.post("/summarize")
def summarize(req: SummarizeRequest):
    joined = " ".join([f"Summary needed for {d}" for d in req.doc_ids])
    prompt = f"Summarize:\n{joined}"
    out = gen(prompt, max_length=150)
    return {"summary": out[0]["generated_text"]}
