# ── Base image ────────────────────────────────────────────────────────────────
# Python 3.11 slim — keeps the image lean
FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy API source ───────────────────────────────────────────────────────────
COPY api/ ./

# ── Copy adapter weights ──────────────────────────────────────────────────────
# Place your downloaded LoRA adapter folder at ./adapter before building
COPY adapter/ ./adapter/

# ── Environment defaults (override with docker run -e) ────────────────────────
ENV BASE_MODEL_ID=codellama/CodeLlama-7b-Instruct-hf
ENV ADAPTER_PATH=./adapter
ENV MAX_NEW_TOKENS=256
ENV TEMPERATURE=0.1

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
