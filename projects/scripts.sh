# Env setup
conda create -n agentic-tutorial
conda activate agentic-tutorial
pip install -r projects/doc-rag/requirements.txt

# Doc-RAG
python projects/doc-rag/ingest.py --input_dir projects/doc-rag/example_data --output_json projects/doc-rag/docs.json
python projects/doc-rag/index_store.py --docs_json projects/doc-rag/docs.json --index_path projects/doc-rag/vectorstore.faiss --meta_path projects/doc-rag/vectorstore_meta.pkl
uvicorn projects.doc-rag.serve_fastapi:app --reload --port 8000


## MCP
python projects/mcp-demo/mcp_client.py projects/mcp-demo/manifest.yaml
cd projects/mcp=demo
python run.py
