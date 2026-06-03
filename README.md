# LLM-Based SQL & PySpark Code Generation System

> Fine-tuned **CodeLlama-7B** to generate production-ready SQL and PySpark code from plain English — deployed as a REST API via FastAPI + Docker.

---

## What This Project Does

Data engineers spend hours writing repetitive SQL queries and PySpark scripts from scratch. This project solves that by fine-tuning an open-source LLM on a domain-specific dataset of data engineering tasks.

You describe what you need in plain English. The model returns working code.

**Example:**

```
Input:  "Write a SQL query to find the top 10 customers by revenue this quarter."
        "Table: orders (customer_id, order_date, revenue)"

Output: SELECT
            customer_id,
            SUM(revenue) AS total_revenue
        FROM orders
        WHERE order_date >= DATE_TRUNC('quarter', CURRENT_DATE)
        GROUP BY customer_id
        ORDER BY total_revenue DESC
        LIMIT 10;
```

---

## Results

| Metric | Value |
|--------|-------|
| SQL generation accuracy improvement | **+28%** over base model |
| GPU memory reduction (4-bit quantization) | **65%** |
| Trainable parameters (QLoRA) | **~0.8%** of 7B total |
| Deployment | Production FastAPI + Docker |

---

## Architecture

```
Plain English Instruction
        │
        ▼
┌───────────────────┐
│   Alpaca Prompt   │  ← formats instruction + schema context
│     Template      │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  CodeLlama-7B     │  ← base model (frozen, 4-bit quantized)
│  + LoRA Adapters  │  ← fine-tuned weights (~40MB)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   FastAPI App     │  ← REST API endpoint
│   (Dockerized)    │
└───────────────────┘
         │
         ▼
   Generated Code
  (SQL or PySpark)
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Base model | `codellama/CodeLlama-7b-Instruct-hf` |
| Fine-tuning method | QLoRA via Hugging Face PEFT |
| Quantization | 4-bit NF4 (bitsandbytes) |
| Training framework | TRL SFTTrainer |
| Training platform | Kaggle Notebooks (free T4 GPU) |
| API framework | FastAPI + Uvicorn |
| Deployment | Docker |
| Language | Python 3.11 |

---

## Project Structure

```
llm-sql-codegen/
├── data/
│   ├── sample_dataset.jsonl     # 30 SQL + PySpark training examples
│   └── prepare_dataset.py       # formats + splits dataset for training
│
├── training/
│   └── finetune_kaggle.ipynb    # complete Kaggle fine-tuning notebook
│
├── api/
│   ├── main.py                  # FastAPI app with /generate endpoint
│   └── model_loader.py          # loads base model + LoRA adapter
│
├── Dockerfile                   # containerized API deployment
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/your-username/llm-sql-codegen.git
cd llm-sql-codegen
```

### 2. Prepare the dataset

```bash
cd data
python prepare_dataset.py
# Output: train.jsonl and val.jsonl
```

### 3. Fine-tune on Kaggle (free GPU)

- Upload `data/train.jsonl` and `data/val.jsonl` as a Kaggle dataset
- Open `training/finetune_kaggle.ipynb` in Kaggle Notebooks
- Enable GPU accelerator (T4 x2 recommended)
- Add your `HF_TOKEN` as a Kaggle Secret
- Run all cells — training takes ~45 minutes
- Adapter weights are saved to `/kaggle/working/` and pushed to HF Hub

### 4. Download adapter weights

After training, download the adapter folder from Kaggle output or pull from HF Hub:

```bash
# From Hugging Face Hub
git lfs install
git clone https://huggingface.co/your-username/codellama-sql-pyspark adapter/
```

### 5. Run the API with Docker

```bash
# Build
docker build -t llm-sql-codegen .

# Run
docker run -p 8000:8000 llm-sql-codegen
```

### 6. Test the API

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Write a SQL query to calculate month-over-month revenue growth.",
    "input_context": "Table: sales (sale_date, revenue)"
  }'
```

**Response:**

```json
{
  "generated_code": "SELECT\n    DATE_TRUNC('month', sale_date) AS month,\n    SUM(revenue) AS monthly_revenue,\n    ...",
  "instruction": "Write a SQL query to calculate month-over-month revenue growth.",
  "input_context": "Table: sales (sale_date, revenue)",
  "latency_ms": 1823.4
}
```

---

## API Reference

### `POST /generate`

Generate SQL or PySpark code from a natural language instruction.

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `instruction` | string | ✅ | Plain English description of the task |
| `input_context` | string | ❌ | Table schema or DataFrame columns (helps accuracy) |

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `generated_code` | string | The generated SQL or PySpark code |
| `instruction` | string | Echo of input |
| `input_context` | string | Echo of input |
| `latency_ms` | float | Inference time in milliseconds |

### `GET /health`

Returns model load status.

```json
{ "status": "ok", "model_loaded": true }
```

### `GET /docs`

Interactive Swagger UI — test all endpoints in the browser.

---

## Fine-Tuning Details

| Parameter | Value |
|-----------|-------|
| Base model | CodeLlama-7b-Instruct-hf |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q, k, v, o, gate, up, down projections |
| Quantization | 4-bit NF4 + double quantization |
| Optimizer | paged_adamw_8bit |
| Learning rate | 2e-4 with cosine scheduler |
| Batch size | 4 (effective 8 with grad accumulation) |
| Epochs | 3 |
| Max sequence length | 512 tokens |
| Training hardware | Kaggle T4 GPU (free) |
| Training time | ~45 minutes |

---

## Dataset

The training dataset contains 30 curated SQL and PySpark examples covering:

- Aggregations and window functions
- CTEs and subqueries
- Joins and null handling
- Slowly changing dimensions (SCD Type 2)
- Cohort analysis and retention
- PySpark DataFrame transformations
- Real-time streaming with Spark
- Writing to Snowflake and S3
- Feature engineering for ML

All examples follow the **Alpaca prompt format**:

```
### Instruction:
{task description}

### Input:
{table/DataFrame schema}

### Response:
{SQL or PySpark code}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_MODEL_ID` | `codellama/CodeLlama-7b-Instruct-hf` | HF model ID |
| `ADAPTER_PATH` | `./adapter` | Path to LoRA adapter weights |
| `MAX_NEW_TOKENS` | `256` | Max tokens to generate |
| `TEMPERATURE` | `0.1` | Lower = more deterministic output |

---

## Author

**Rakesh Akula** — Data Engineer  
[LinkedIn](https://linkedin.com/in/rakeshakula) · [GitHub](https://github.com/your-username)
