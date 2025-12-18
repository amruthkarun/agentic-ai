# 🧠 Agentic AI — Agentic Workflows, MCP Tools & Document RAG

This repository provides a modular collection of **Agentic AI**, **MCP (Model Context Protocol)**, and **Document Understanding / RAG** projects.
Each project is self-contained yet designed to work together to form end-to-end intelligent systems capable of:

- 🔍 Semantic document search
- 📚 Retrieval-Augmented Generation (RAG)
- 🤖 Agentic tool execution
- 🧠 Context-aware routing via MCP
- 🧩 Summarization & question answering
- 📄 Automatic document ingestion and indexing

All components are open-source, lightweight, and easy to run locally.

---

# 🚀 Features

### 📄 **Document Understanding (Doc-RAG)**
- PDF & text ingestion → JSON chunks
- Vector store creation with FAISS
- FastAPI backend for semantic retrieval
- Query endpoint for RAG-ready pipelines

### 🤖 **Agentic MCP Demo**
- Tool routing via LLM
- Tools: reader, summarizer, semantic search, keyword search
- Manifest-based tool configuration
- Interactive agent shell

### 🔍 **Semantic Search Engine**
- Uses *MiniLM* sentence embeddings
- Multi-file retrieval
- Chunk-level ranking
- Answer extraction

### 🧱 **Modular, Extensible Design**
Minimal dependencies, clear separation of tools, routers, indexing, and pipelines.

---


# ⚙️ Environment Setup

To set up the environment:

```bash
conda create -n agentic-tutorial python=3.10 -y
conda activate agentic-tutorial
pip install -r projects/doc-rag/requirements.txt
```


## 1️⃣ Ingest Documents

Convert raw .txt or .pdf files to structured JSON:

```bash
python projects/doc-rag/ingest.py \
    --input_dir projects/doc-rag/example_data \
    --output_json projects/doc-rag/docs.json
```

## 2️⃣ Build Vector Index (FAISS)
```bash
python projects/doc-rag/index_store.py \
    --docs_json projects/doc-rag/docs.json \
    --index_path projects/doc-rag/vectorstore.faiss \
    --meta_path projects/doc-rag/vectorstore_meta.pkl
```
## 3️⃣ Start the FastAPI RAG Server
```bash
uvicorn projects.doc-rag.serve_fastapi:app --reload --port 8000
```
### Access:

API documentation: http://localhost:8000/docs

Query endpoint: /query

## 🤖 MCP Agent Demo

This project demonstrates using Model Context Protocol to build a fully agentic tool-use system.

Tools include:

📝 Summarizer (DistilBART/T5-small)

🔍 Search (keyword + semantic)

📄 Reader (load files)

🧠 Semantic question answering

## 1️⃣ Run the MCP Client
```bash
python projects/mcp-demo/mcp_client.py projects/mcp-demo/manifest.yaml
```
This loads the router LLM, active tool definitions, and context from the manifest.

## 2️⃣ Run the Interactive Agent

```bash
cd projects/mcp-demo
python run.py
```
---
### Example agent queries:

* summarize this text: Agentic AI refers to...
* What is agentc AI?
* what is retrieval augmented generation?

The agent automatically selects the correct tool.
