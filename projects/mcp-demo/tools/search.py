from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util

# lightweight semantic embedding model
EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def load_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""


def chunk_text(text: str, chunk_size: int = 350, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += (chunk_size - overlap)

    return chunks


def query(term: str, directory: str = "search_index", max_results: int = 5):
    """
    Semantic search over all text files in the directory.

    Parameters:
        term (str): user query (semantic meaning)
        directory (str): folder that contains .txt files
        max_results (int): number of top semantic matches to return

    Returns:
        JSON dictionary containing top semantic chunks across all files
    """

    Path(directory).mkdir(exist_ok=True)

    # gather all chunks across all documents
    all_chunks = []
    file_refs = []

    for path in Path(directory).glob("*.txt"):
        text = load_text_file(str(path))
        if not text:
            continue

        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        file_refs.extend([str(path)] * len(chunks))  # keep track of origin

    if not all_chunks:
        return {
            "query": term,
            "results": [],
            "message": f"No .txt documents found in {directory}"
        }

    # encode query + chunks
    q_embed = EMBED_MODEL.encode(term, convert_to_tensor=True)
    c_embed = EMBED_MODEL.encode(all_chunks, convert_to_tensor=True)

    # compute semantic similarity
    scores = util.cos_sim(q_embed, c_embed)[0]

    # pick top-k matches
    top_scores = scores.topk(k=min(max_results, len(all_chunks)))

    results = []
    for score, idx in zip(top_scores.values, top_scores.indices):
        idx = int(idx)
        results.append({
            "file": file_refs[idx],
            "score": float(score),
            "chunk": all_chunks[idx]
        })

    return {
        "query": term,
        "num_results": len(results),
        "results": results
    }
