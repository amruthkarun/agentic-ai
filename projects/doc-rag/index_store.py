import json
import pickle
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_index(docs_json: str, index_path: str, meta_path: str):
    model = SentenceTransformer(MODEL)

    with open(docs_json, "r", encoding="utf-8") as fh:
        docs = json.load(fh)

    texts = [d["text"] for d in docs]
    metas = [
        {"doc_id": d["doc_id"], "chunk_id": d["chunk_id"], "source_path": d["source_path"]}
        for d in docs
    ]

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)

    with open(meta_path, "wb") as fh:
        pickle.dump(metas, fh)

    print("Index + metadata saved.")

def load_index(index_path: str, meta_path: str):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as fh:
        metas = pickle.load(fh)

    model = SentenceTransformer(MODEL)

    return model, index, metas

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_json", default="docs.json")
    parser.add_argument("--index_path", default="vectorstore.faiss")
    parser.add_argument("--meta_path", default="vectorstore_meta.pkl")
    args = parser.parse_args()

    build_index(args.docs_json, args.index_path, args.meta_path)
