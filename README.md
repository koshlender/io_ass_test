# Multi-agent legal system (Supervisor + RAG + Critique)

This project implements a 3-agent architecture using **OpenAI tool-calling style** and a vLLM-hosted GPT-OSS model:

1. **Supervisor Agent**: decides whether a query should be answered directly or with retrieval (`direct` vs `rag`).
2. **RAG Agent**: retrieves context from a FAISS vector DB (Bharatiya Nyaya Sanhita + conversation memory) and answers with grounding.
3. **Critique Agent**: reviews the draft answer, checks route suitability, and produces final improved output.

## Features

- OpenAI-compatible API calls (`openai` client with `base_url` for vLLM).
- Tool-calling route decision by supervisor.
- FAISS vector database for:
  - BNS corpus chunks (downloaded from internet source).
  - Conversation turn embeddings (`User + Assistant`) for memory retrieval.
- Routing evaluation harness to test whether GPT-OSS routes correctly.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

```bash
export VLLM_BASE_URL="http://localhost:8000/v1"
export MODEL_NAME="gpt-oss"
export OPENAI_API_KEY="EMPTY"
```

## Run

### 1) Ingest Bharatiya Nyaya Sanhita from your local PDF into FAISS

```bash
export BNS_PDF_PATH="/absolute/path/to/BNS.pdf"
INGEST_BNS=true python multiagent_bns.py
```

### 2) Optional fallback: ingest from internet source

If `BNS_PDF_PATH` is not set, ingestion falls back to the internet source configured in code.

```bash
unset BNS_PDF_PATH
INGEST_BNS=true python multiagent_bns.py
```

### 3) Normal run (without re-ingestion)
### 1) Ingest Bharatiya Nyaya Sanhita into FAISS

```bash
INGEST_BNS=true python multiagent_bns.py
```

### 2) Normal run (without re-ingestion)

```bash
python multiagent_bns.py
```

## Notes

- Internet source for BNS currently used in code:
  - `https://en.wikisource.org/wiki/Bharatiya_Nyaya_Sanhita,_2023`
- FAISS files are persisted in `./vector_db/`.
- You can adapt `evaluate_routing()` with your own labeled benchmark to test supervisor reasoning quality more rigorously.
