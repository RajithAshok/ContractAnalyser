# Contract Analyzer

> A local, privacy-first AI system for analyzing legal contracts — no API keys, no cloud, no cost.

Built as a multi-agent RAG pipeline that extracts clauses, scores risk, compares documents side-by-side, and answers natural language questions grounded in the actual contract text. Everything runs on your machine using [Ollama](https://ollama.com) and a local vector database.

---

## Demo

<img width="800" alt="image" src="https://github.com/user-attachments/assets/b293293b-0a23-4380-9d02-ad8ee5c47f79" />


---

## Features

| Feature | Description |
|---|---|
| **Clause extraction** | Identifies and categorizes clauses — liability, termination, IP, payment, non-compete, arbitration, and more |
| **Risk scoring** | Scores each clause 0–10 and flags why it may be unfavorable for the reviewing party |
| **Document comparison** | Diffs two contracts clause-by-clause and recommends which is more favorable |
| **RAG Q&A** | Answers natural language questions with citations to the exact contract text |
| **Executive summary** | Plain-English overview of what the contract says and its overall risk level |
| **Web UI** | Optional Gradio browser interface alongside the terminal CLI |

---

## Architecture

The system uses a **router agent** that classifies the user's intent and dispatches to one of four specialized agents. Each agent is a separate module with its own prompt strategy and output schema.

```
User input
    │
    ▼
┌────────────────────────────────────────┐
│            Router Agent                │
│   (classifies intent → picks tool)     │
└────┬──────────┬──────────┬──────────┬──┘
     │          │          │          │
     ▼          ▼          ▼          ▼
Extractor   Risk        Comparator  Q&A Agent
Agent       Scorer      Agent       (RAG)
     │          │          │          │
     └──────────┴─────┬────┴──────────┘
                      │
              ChromaDB vector store
              (local embeddings via
               sentence-transformers)
```

**Design decisions worth noting:**

- **Semantic chunking**: The chunker splits on legal section boundaries (numbered headings, `ARTICLE`, `CLAUSE`, etc.) before falling back to sentence-aware overlap. This keeps clause context intact across chunk boundaries, which meaningfully improves retrieval precision compared to naive fixed-size splitting.
- **Structured JSON output**: Every agent prompts the LLM to return a strict JSON schema, validated with Pydantic. Malformed responses are caught and recovered from rather than crashing.
- **Local-only stack**: No data leaves the machine. Embeddings use `all-MiniLM-L6-v2` (~90MB, downloaded once), the vector store is a persistent ChromaDB on disk, and the LLM runs through Ollama.

---

## Stack

| Component | Library |
|---|---|
| LLM runtime | [Ollama](https://ollama.com) (`llama3.2:1b`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB (persistent, on-disk) |
| PDF parsing | `pdfplumber` |
| Data validation | Pydantic v2 |
| CLI | `rich` + `typer` |
| Web UI | Gradio |

---

## Getting started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running

### 1. Pull the model

```bash
ollama pull llama3.2:1b
```

### 2. Clone and install

```bash
git clone https://github.com/yourusername/contract-analyzer.git
cd contract-analyzer

python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Run

```bash
# Interactive CLI (recommended to start)
python -m src.cli

# Analyze a specific file directly
python -m src.cli analyze contracts/samples/nda_sample.txt

# Optional: browser UI at http://localhost:7860
python -m src.app
```

---

## CLI usage

```
▶ load contracts/samples/saas_agreement.txt
✓ Indexed 'saas_agreement' (9 chunks)

▶ extract
  Extracts and lists all clauses with risk scores

▶ risk
  Full risk report with severity buckets and negotiation recommendations

▶ ask What is the liability cap?
  RAG answer with citations from the contract

▶ compare contracts/samples/nda_sample.txt
  Clause-by-clause diff between the two loaded contracts

▶ list
  Show all indexed contracts

▶ use <docname>
  Switch the active document

▶ help
  Show all commands
```

Sample contracts are included in `contracts/samples/` to test with immediately.

---

## Project structure

```
contract-analyzer/
├── src/
│   ├── agents/
│   │   ├── router.py        # intent classifier + tool dispatcher
│   │   ├── extractor.py     # clause extraction agent
│   │   ├── risk_scorer.py   # risk bucketing + recommendations
│   │   ├── comparator.py    # cross-document diff agent
│   │   └── qa.py            # RAG question answering
│   ├── rag/
│   │   ├── chunker.py       # section-aware contract chunker
│   │   ├── embedder.py      # local sentence-transformer embeddings
│   │   └── retriever.py     # ChromaDB indexing + similarity search
│   ├── models.py            # Pydantic schemas (Clause, Risk, Diff, QA)
│   ├── ollama_client.py     # Ollama HTTP wrapper + JSON parsing
│   ├── cli.py               # rich/typer terminal interface
│   └── app.py               # Gradio web UI
├── contracts/
│   └── samples/
│       ├── nda_sample.txt
│       └── saas_agreement.txt
├── tests/
│   ├── conftest.py
│   ├── test_chunker.py
│   ├── test_models.py
│   └── test_client.py
├── requirements.txt
└── README.md
```

---

## Running tests

```bash
pytest tests/ -v
```

The test suite covers chunking logic, Pydantic schema validation, risk scoring, and JSON parsing — all without requiring Ollama to be running.

---

## Privacy

All processing happens locally. No contract text, embeddings, or query data is sent to any external service. The vector store is written to `.chroma_db/` in the project directory and can be deleted at any time to clear the index.
