FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by pdfplumber and chromadb
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker caches this layer separately
# so pip install doesn't re-run on every code change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# ChromaDB persistence folder
RUN mkdir -p /app/.chroma_db

# Contracts folder
RUN mkdir -p /app/contracts/samples

# Default command — runs the CLI
CMD ["python", "-m", "src.cli"]