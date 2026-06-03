"""
Dataset Preparation Script
===========================
Loads raw JSONL examples, formats them into the Alpaca prompt template,
and splits into train/validation sets ready for fine-tuning.
"""

import json
import random
from pathlib import Path


# ── Alpaca-style prompt template ──────────────────────────────────────────────
PROMPT_TEMPLATE = """Below is an instruction that describes a data engineering task.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


def load_jsonl(filepath: str) -> list[dict]:
    """Load all records from a JSONL file."""
    records = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_prompt(record: dict) -> str:
    """Format a single record into the Alpaca prompt template."""
    return PROMPT_TEMPLATE.format(
        instruction=record.get("instruction", ""),
        input=record.get("input", ""),
        output=record.get("output", ""),
    )


def split_dataset(records: list, train_ratio: float = 0.9, seed: int = 42):
    """Split records into train and validation sets."""
    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def save_jsonl(records: list, filepath: str):
    """Save records to a JSONL file."""
    with open(filepath, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"  Saved {len(records)} records → {filepath}")


def main():
    data_dir = Path(__file__).parent

    # ── Load raw dataset ──────────────────────────────────────────────────────
    raw_path = data_dir / "sample_dataset.jsonl"
    records = load_jsonl(str(raw_path))
    print(f"\n Loaded {len(records)} raw examples from {raw_path.name}")

    # ── Format prompts ────────────────────────────────────────────────────────
    formatted = [{"text": format_prompt(r)} for r in records]
    print(f" Formatted {len(formatted)} examples into Alpaca prompt template")

    # ── Split ─────────────────────────────────────────────────────────────────
    train_data, val_data = split_dataset(formatted, train_ratio=0.9)
    print(f" Train: {len(train_data)} | Validation: {len(val_data)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_jsonl(train_data, str(data_dir / "train.jsonl"))
    save_jsonl(val_data,   str(data_dir / "val.jsonl"))

    print("\n Dataset preparation complete!")
    print("  Next step: Upload train.jsonl and val.jsonl to Kaggle as a dataset.")


if __name__ == "__main__":
    main()
