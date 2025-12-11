from index_store import load_index

class Retriever:
    def __init__(self, index_path="vectorstore.faiss", meta_path="vectorstore_meta.pkl"):
        self.model, self.index, self.metas = load_index(index_path, meta_path)

    def retrieve(self, query: str, k: int = 5):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb, k)

        results = []
        for score, idx in zip(D[0], I[0]):
            results.append({
                "score": float(score),
                "meta": self.metas[idx]
            })

        return results

if __name__ == "__main__":
    r = Retriever()
    print(r.retrieve("Explain the main idea."))
