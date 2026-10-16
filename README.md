# Contract Analyzer

> A local, privacy-first AI system for analyzing legal contracts — no API keys, no cloud, no cost.

Built as a multi-agent RAG pipeline that extracts clauses, scores risk, compares documents side-by-side, and answers natural language questions grounded in the actual contract text. Fully containerized with Docker, orchestrated with Kubernetes, provisioned with Terraform, and shipped with a GitHub Actions CI/CD pipeline.

---

## Demo

<!-- Run `risk` on one of the sample contracts and screenshot the terminal output -->
<!-- Save the screenshot as assets/demo.png -->

<img width="800" alt="image" src="https://github.com/user-attachments/assets/31189d66-263f-4ef0-8bed-78b4537070e9" />

---

## Features

| Feature | Description |
|---|---|
| **Clause extraction** | Identifies and categorizes clauses — liability, termination, IP, payment, non-compete, arbitration, and more |
| **Risk scoring** | Scores each clause 0–10 and flags why it may be unfavorable for the reviewing party |
| **Document comparison** | Diffs two contracts clause-by-clause and recommends which is more favorable |
| **RAG Q&A** | Answers natural language questions with citations to the exact contract text |
| **Executive summary** | Plain-English overview of what the contract says and its overall risk level |

---

## Architecture

```
GitHub push
    → GitHub Actions CI/CD
        → runs 27-test suite
        → builds Docker image
        → pushes to GitHub Container Registry
        → validates Terraform config

Local infrastructure (Terraform → Kubernetes → Docker):
    ┌─────────────────────────────────────────┐
    │            Router Agent                 │
    │   (classifies intent → picks tool)     │
    └────┬──────────┬──────────┬─────────────┘
         │          │          │          │
         ▼          ▼          ▼          ▼
    Extractor   Risk        Comparator  Q&A Agent
    Agent       Scorer      Agent       (RAG)
         │          │          │          │
                  ChromaDB (vector store)
                  Ollama (local LLM)
```

---

## Stack

| Layer | Technology |
|---|---|
| LLM runtime | Ollama (`llama3.2:1b`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB |
| Data validation | Pydantic v2 |
| CLI | `rich` + `typer` |
| Containerization | Docker + Docker Compose |
| Orchestration | Kubernetes (minikube) |
| Infrastructure as code | Terraform |
| CI/CD | GitHub Actions + GitHub Container Registry |

---

## Prerequisites

Before running this project make sure you have the following installed:

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) — with WSL2 backend enabled
- [minikube](https://minikube.sigs.k8s.io/docs/start/) — local Kubernetes cluster
- [kubectl](https://kubernetes.io/docs/tasks/tools/) — Kubernetes CLI
- [Terraform](https://developer.hashicorp.com/terraform/install) — infrastructure provisioning
- Python 3.11+

---

## Running the project

There are three ways to run this project. Pick one.

---

### Option A — Run locally (no Docker, simplest)

Best for development and making code changes.

**1. Install Ollama**

Download from https://ollama.com/download and install it.

**2. Start Ollama and pull the model**

```powershell
ollama serve
```

Open a new terminal and run:

```powershell
ollama pull llama3.2:1b
```

**3. Set up Python environment**

```powershell
cd C:\path\to\ContractAnalyser
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**4. Run the CLI**

```powershell
python -m src.cli
```

**5. Use it**

```
▶ load contracts/samples/saas_agreement.txt
▶ risk
▶ ask What is the liability cap?
▶ compare contracts/samples/nda_sample.txt
▶ help
```

**Shut down**

```powershell
# Press Ctrl+C or type exit in the CLI
# Stop Ollama
taskkill /IM ollama.exe /F
# Deactivate Python environment
deactivate
```

---

### Option B — Run with Docker Compose (recommended for demos)

Runs all three services together — the analyzer, Ollama, and ChromaDB — in containers.

**1. Make sure Docker Desktop is running**

Open Docker Desktop from the Start menu and wait for it to fully load.

**2. Start all services**

```powershell
cd C:\path\to\ContractAnalyser
docker compose up
```

Wait until you see all three services report healthy. This takes about a minute on first run.

**3. Pull the model (first time only)**

Open a second PowerShell window:

```powershell
docker exec ollama ollama pull llama3.2:1b
```

Wait for the download to complete.

**4. Open the CLI**

```powershell
docker exec -it contract-analyzer python -m src.cli
```

**5. Use it**

```
▶ load contracts/samples/saas_agreement.txt
▶ risk
▶ ask What is the liability cap?
▶ compare contracts/samples/nda_sample.txt
▶ help
```

**Shut down**

```powershell
# Stop and remove containers (keeps volumes — models and data are preserved)
docker compose down

# Stop and remove everything including volumes (deletes downloaded models and ChromaDB data)
docker compose down -v
```

---

### Option C — Run with Kubernetes and Terraform (full stack)

Provisions the entire infrastructure declaratively with Terraform on a local minikube cluster.

**1. Make sure Docker Desktop is running**

Open Docker Desktop from the Start menu and wait for it to fully load.

**2. Start minikube**

```powershell
minikube start --driver=docker
```

Wait until you see `Done! kubectl is now configured to use "minikube"`.

**3. Build the image inside minikube**

```powershell
cd C:\path\to\ContractAnalyser
minikube image build -t contract-analyzer:latest .
```

**4. Provision infrastructure with Terraform**

```powershell
cd terraform
terraform init
terraform apply
```

Type `yes` when prompted. Terraform will create the namespace, ConfigMap, PersistentVolumeClaims, StatefulSets, Services, and Deployment.

**5. Verify everything is running**

```powershell
kubectl get all -n contract-analyzer
```

All pods should show `Running` and `1/1 Ready`. If they are still starting wait 30 seconds and run it again.

**6. Pull the model (first time only)**

```powershell
kubectl exec -n contract-analyzer ollama-0 -- ollama pull llama3.2:1b
```

Wait for the download to complete.

**7. Copy sample contracts into the pod**

First get the analyzer pod name:

```powershell
kubectl get pods -n contract-analyzer -l app=contract-analyzer
```

Then copy the files (replace `<pod-name>` with the name from above):

```powershell
kubectl exec -n contract-analyzer deployment/contract-analyzer -- mkdir -p /app/contracts/samples

kubectl cp contracts\samples\nda_sample.txt contract-analyzer\<pod-name>:/app/contracts/samples/nda_sample.txt

kubectl cp contracts\samples\saas_agreement.txt contract-analyzer\<pod-name>:/app/contracts/samples/saas_agreement.txt
```

**8. Open the CLI**

```powershell
kubectl exec -it -n contract-analyzer deployment/contract-analyzer -- python -m src.cli
```

**9. Use it**

```
▶ load contracts/samples/saas_agreement.txt
▶ risk
▶ ask What is the liability cap?
▶ compare contracts/samples/nda_sample.txt
▶ help
```

**Shut down**

```powershell
# Remove all Kubernetes resources provisioned by Terraform
cd terraform
terraform destroy

# Type yes when prompted

# Stop minikube
minikube stop

# Optional — delete the minikube cluster entirely (frees disk space)
minikube delete
```

---

## CLI commands

| Command | What it does |
|---|---|
| `load <file>` | Index a `.txt` or `.pdf` contract |
| `extract` | Extract and list all clauses with risk scores |
| `risk` | Full risk report with severity buckets and negotiation recommendations |
| `ask <question>` | RAG question answering with citations from the contract |
| `compare <file>` | Clause-by-clause diff between two contracts |
| `list` | Show all indexed contracts |
| `use <docname>` | Switch the active document |
| `delete <docname>` | Remove a document from the index |
| `help` | Show all commands |
| `exit` | Quit |

---

## Running tests

Tests run without Ollama or any running service — pure unit tests.

```powershell
# Local
venv\Scripts\activate
pytest tests/ -v

# Inside Docker
docker exec contract-analyzer pytest tests/ -v

# Inside Kubernetes
kubectl exec -n contract-analyzer deployment/contract-analyzer -- pytest tests/ -v
```

---

## Project structure

```
ContractAnalyser/
├── .github/
│   └── workflows/
│       ├── ci.yml           # runs tests and builds image on every push
│       └── cd.yml           # pushes image to GHCR and validates Terraform on merge to main
├── contracts/
│   └── samples/
│       ├── nda_sample.txt
│       └── saas_agreement.txt
├── k8s/                     # raw Kubernetes manifests (reference — Terraform is used to apply)
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── ollama-pvc.yaml
│   ├── chromadb-pvc.yaml
│   ├── ollama.yaml
│   ├── chromadb.yaml
│   └── analyzer.yaml
├── src/
│   ├── agents/
│   │   ├── router.py        # intent classifier and tool dispatcher
│   │   ├── extractor.py     # clause extraction agent
│   │   ├── risk_scorer.py   # risk bucketing and recommendations
│   │   ├── comparator.py    # cross-document diff agent
│   │   └── qa.py            # RAG question answering
│   ├── rag/
│   │   ├── chunker.py       # section-aware contract chunker
│   │   ├── embedder.py      # local sentence-transformer embeddings
│   │   └── retriever.py     # ChromaDB indexing and similarity search
│   ├── models.py            # Pydantic schemas
│   ├── ollama_client.py     # Ollama HTTP wrapper
│   ├── cli.py               # terminal interface
│   └── app.py               # optional Gradio web UI
├── terraform/
│   ├── providers.tf         # Kubernetes provider config
│   ├── variables.tf         # configurable values
│   ├── namespace.tf
│   ├── configmap.tf
│   ├── storage.tf           # PersistentVolumeClaims
│   ├── ollama.tf            # StatefulSet and Service
│   ├── chromadb.tf          # StatefulSet and Service
│   └── analyzer.tf          # Deployment
├── tests/
│   ├── conftest.py
│   ├── test_chunker.py
│   ├── test_models.py
│   └── test_client.py
├── .dockerignore
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Privacy

All processing happens locally. No contract text, embeddings, or query data is sent to any external service. The vector store is written to `.chroma_db/` locally or to a PersistentVolume in Kubernetes, and can be deleted at any time to clear the index.

---
