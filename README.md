# Contract Analyser — Local AI Legal Document Intelligence

A multi-agent RAG system for analyzing legal contracts. Runs **entirely locally** using [Ollama](https://ollama.com).

## What it does

- **Clause extraction** — identifies and categorizes key clauses (liability, termination, IP, payment, etc.)
- **Risk scoring** — scores each clause 1–10 and explains why it may be unfavorable
- **Cross-document comparison** — diffs two contracts clause-by-clause and surfaces discrepancies
- **RAG Q&A** — ask natural language questions answered with citations to the actual contract text
- **Plain-English summary** — one-paragraph executive summary for non-lawyers

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           Router Agent                  │  ← decides which tool(s) to invoke
└────────────┬──────────────┬─────────────┘
             │              │
     ┌───────▼──────┐  ┌────▼──────────────┐
     │ RAG Retriever│  │  Analysis Agent   │
     │  (ChromaDB)  │  │  (Ollama LLM)     │
     └───────┬──────┘  └────┬──────────────┘
             │              │
     chunk + embed      structured JSON
     with metadata      output (clauses,
     (doc, page,        risks, diffs)
      clause_type)
             │
     ┌───────▼──────────────────────────────┐
     │         Tool Registry                │
     │  extract_clauses | score_risk        │
     │  compare_docs    | answer_question   │
     │  summarize                           │
     └──────────────────────────────────────┘
```

## Stack

| Component | Library | Why |
|-----------|---------|-----|
| LLM | Ollama (`llama3.2` or `mistral`) | Free, local, fast |
| Embeddings | `sentence-transformers` | Local embedding, no API |
| Vector store | ChromaDB | Lightweight local vector DB |
| PDF parsing | `pdfplumber` | Accurate text + layout extraction |
| CLI | `rich` + `typer` | Beautiful terminal UI |
| Web UI | `gradio` | Optional browser interface |

## Setup

### 1. Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2
```

### 2. Install Python dependencies

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run

```bash
# Interactive CLI
python -m src.cli

# Or with a specific contract
python -m src.cli analyze contracts/samples/nda_sample.txt

# Optional: web UI
python -m src.app
```
