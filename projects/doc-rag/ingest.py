import os
import argparse
from pathlib import Path
from PyPDF2 import PdfReader
import json

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    text = []
    for p in reader.pages:
        try:
            text.append(p.extract_text() or "")
        except:
            pass
    return "\n".join(text)

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def main(input_dir: str, output_json: str):
    files = list(Path(input_dir).glob("**/*.pdf"))
    docs = []

    for f in files:
        text = extract_text_from_pdf(f)
        chunks = chunk_text(text)

        for i, c in enumerate(chunks):
            docs.append({
                "doc_id": f.stem,
                "chunk_id": f"{f.stem}_chunk_{i}",
                "text": c,
                "source_path": str(f)
            })

    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(docs, fh, indent=2)

    print(f"Wrote {len(docs)} chunks to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_json", default="docs.json")
    args = parser.parse_args()

    main(args.input_dir, args.output_json)
