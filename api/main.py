"""
FastAPI Inference Endpoint
==========================
Serves the fine-tuned CodeLlama-7B model as a REST API.

Endpoints:
    POST /generate   — generate SQL or PySpark code
    GET  /health     — liveness check
    GET  /           — API info
"""

from contextlib import asynccontextmanager
from typing import Optional
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model_loader import load_model, generate


# ── App state ─────────────────────────────────────────────────────────────────
model_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, release on shutdown."""
    print("[startup] Loading model — this takes ~30s on first run ...")
    model_state["model"], model_state["tokenizer"] = load_model()
    print("[startup] Model loaded. API is ready.")
    yield
    model_state.clear()
    print("[shutdown] Model released.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="LLM SQL & PySpark Code Generator",
    description=(
        "Fine-tuned CodeLlama-7B that generates production-ready SQL queries "
        "and PySpark code from plain English instructions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    instruction: str = Field(
        ...,
        min_length=10,
        max_length=500,
        example="Write a SQL query to find the top 10 customers by total revenue this quarter.",
    )
    input_context: Optional[str] = Field(
        default="",
        max_length=300,
        example="Table: orders (customer_id, order_date, revenue)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "instruction": "Write a SQL query to find the top 10 customers by total revenue.",
                "input_context": "Table: orders (customer_id, order_date, revenue)",
            }
        }


class GenerateResponse(BaseModel):
    generated_code: str
    instruction:    str
    input_context:  str
    latency_ms:     float


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "name":        "LLM SQL & PySpark Code Generator",
        "version":     "1.0.0",
        "author":      "Rakesh Akula",
        "description": "Fine-tuned CodeLlama-7B for data engineering code generation.",
        "endpoints": {
            "generate": "POST /generate",
            "health":   "GET  /health",
            "docs":     "GET  /docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    loaded = "model" in model_state and model_state["model"] is not None
    return HealthResponse(status="ok" if loaded else "loading", model_loaded=loaded)


@app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
def generate_code(request: GenerateRequest):
    """
    Generate SQL or PySpark code from a plain English instruction.

    - **instruction**: describe the data task in plain English
    - **input_context**: optional table schema or DataFrame columns to guide the model
    """
    if "model" not in model_state:
        raise HTTPException(status_code=503, detail="Model is still loading. Try again in a moment.")

    try:
        t0   = time.perf_counter()
        code = generate(
            model       = model_state["model"],
            tokenizer   = model_state["tokenizer"],
            instruction = request.instruction,
            input_context = request.input_context or "",
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        return GenerateResponse(
            generated_code = code,
            instruction    = request.instruction,
            input_context  = request.input_context or "",
            latency_ms     = latency_ms,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
