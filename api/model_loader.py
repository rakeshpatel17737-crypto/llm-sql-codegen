"""
Model Loader
============
Loads the fine-tuned CodeLlama-7B adapter on top of the base model
and exposes a single generate() function for the FastAPI app.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Config (override via environment variables) ───────────────────────────────
BASE_MODEL_ID  = os.getenv("BASE_MODEL_ID",  "codellama/CodeLlama-7b-Instruct-hf")
ADAPTER_PATH   = os.getenv("ADAPTER_PATH",   "./adapter")   # local folder or HF repo
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
TEMPERATURE    = float(os.getenv("TEMPERATURE", "0.1"))

# ── Prompt template (must match training template exactly) ────────────────────
PROMPT_TEMPLATE = """\
Below is an instruction that describes a data engineering task.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""


def _build_bnb_config() -> BitsAndBytesConfig:
    """4-bit NF4 quantization — same config used during training."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_model():
    """
    Load base model in 4-bit + LoRA adapter weights.
    Returns (model, tokenizer) ready for inference.
    """
    print(f"[loader] Loading tokenizer from {BASE_MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"[loader] Loading base model in 4-bit ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=_build_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"[loader] Attaching LoRA adapter from {ADAPTER_PATH} ...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    device = next(model.parameters()).device
    print(f"[loader] Model ready on {device}")
    return model, tokenizer


def generate(model, tokenizer, instruction: str, input_context: str = "") -> str:
    """
    Generate SQL or PySpark code from a natural language instruction.

    Args:
        model:          Loaded PEFT model
        tokenizer:      Matching tokenizer
        instruction:    Natural language task description
        input_context:  Optional table/DataFrame schema context

    Returns:
        Generated code as a string
    """
    prompt = PROMPT_TEMPLATE.format(
        instruction=instruction,
        input=input_context,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    # Decode and strip the prompt prefix
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response  = full_text.split("### Response:")[-1].strip()
    return response
