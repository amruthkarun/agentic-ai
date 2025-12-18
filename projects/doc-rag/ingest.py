import argparse
import json
from pathlib import Path
from PyPDF2 import PdfReader
from tqdm import tqdm
from typing import Iterator, List

DEFAULT_CHUNK_SIZE = 1000   # words
DEFAULT_OVERLAP = 200       # words

def tokenize(text: str) -> List[str]:
    # Simple whitespace tokenizer; replace with a better tokenizer if needed
    return text.split()

def page_texts_from_pdf(path: Path) -> Iterator[str]:
    """Yield text page-by-page. Keeps only one page's text in memory at a time."""
    try:
        reader = PdfReader(str(path))
    except Exception as e:
        print(f"WARNING: failed to open {path}: {e}")
        return
    for p in reader.pages:
        try:
            yield p.extract_text() or ""
        except Exception:
            yield ""

def chunk_generator_from_tokens(tokens: List[str], chunk_size: int, overlap: int) -> Iterator[str]:
    """Yield chunks given a token list using overlap; memory usage bounded by tokens length passed in."""
    if not tokens:
        return
    start = 0
    n = len(tokens)
    while start < n:
        end = min(start + chunk_size, n)
        yield " ".join(tokens[start:end])
        if end == n:
            break
        # move start forward, leaving overlap tokens
        start = end - overlap
        if start < 0:
            start = 0

def chunk_text_stream(page_text_iter: Iterator[str], chunk_size: int, overlap: int) -> Iterator[str]:
    """
    Build chunks by reading pages sequentially, accumulating a rolling buffer of tokens.
    This avoids tokenizing the entire document at once.
    """
    buffer: List[str] = []
    for page_text in page_text_iter:
        if not page_text:
            continue
        page_tokens = tokenize(page_text)
        if not page_tokens:
            continue
        buffer.extend(page_tokens)

        # While we have at least one full chunk, yield it and keep the overlap
        while len(buffer) >= chunk_size:
            chunk = buffer[:chunk_size]
            yield " ".join(chunk)
            # keep overlap tokens for next chunk
            buffer = buffer[chunk_size - overlap:]
    # After all pages, flush remaining content as one chunk (if any)
    if buffer:
        yield " ".join(buffer)

def ingest_directory(input_dir: Path,
                     output_json: Path,
                     chunk_size: int = DEFAULT_CHUNK_SIZE,
                     overlap: int = DEFAULT_OVERLAP,
                     max_files: int = None,
                     max_file_size_mb: float = None):
    input_dir = Path(input_dir)
    files = list(input_dir.glob("**/*.pdf"))
    if max_files:
        files = files[:max_files]

    with output_json.open("w", encoding="utf-8") as out_f:
        for f in tqdm(files, desc="PDF files"):
            try:
                if max_file_size_mb is not None:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    if size_mb > max_file_size_mb:
                        print(f"Skipping {f} (size {size_mb:.1f} MB > {max_file_size_mb} MB)")
                        continue

                page_iter = page_texts_from_pdf(f)
                chunk_iter = chunk_text_stream(page_iter, chunk_size, overlap)
                chunk_count = 0
                for chunk in chunk_iter:
                    if not chunk.strip():
                        continue
                    record = {
                        "doc_id": f.stem,
                        "chunk_id": f"{f.stem}_chunk_{chunk_count}",
                        "text": chunk,
                        "source_path": str(f)
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False))
                    chunk_count += 1

                if chunk_count == 0:
                    # file had no extractable text
                    print(f"NOTE: no text extracted from {f}")

            except Exception as e:
                print(f"ERROR processing {f}: {e}")

    print(f"Ingestion complete. Output written to {output_json}")

def parse_args():
    p = argparse.ArgumentParser(description="PDF ingestion to JSONL")
    p.add_argument("--input_dir", required=True, help="Directory with PDFs")
    p.add_argument("--output_json", default="docs.json", help="Output JSON file")
    p.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size (words)")
    p.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="Chunk overlap (words)")
    p.add_argument("--max_files", type=int, default=None, help="Limit number of files to process")
    p.add_argument("--max_file_size_mb", type=float, default=None, help="Skip files larger than this size (MB)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    ingest_directory(
        input_dir=Path(args.input_dir),
        output_json=Path(args.output_json),
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        max_files=args.max_files,
        max_file_size_mb=args.max_file_size_mb
    )
