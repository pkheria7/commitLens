"""
CommitLens — gradio.Server mode
================================
- Serves custom index.html at GET /
- Exposes process_repo via @app.api() for the JS frontend to call
- Mellum 2 (6-bit, CPU-resident) handles per-file summaries via batched GPU inference
- Groq llama-70b handles the final report (fast, no GPU cost)
- <think>...</think> blocks stripped from all Mellum outputs
- Per-file output is tightly constrained to 3-5 bullet points max
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

import spaces
import torch
from fastapi.responses import HTMLResponse
from gradio import Server
from groq import Groq
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from commitlens import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("commitlens")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_REPO_ID   = "JetBrains/Mellum2-12B-A2.5B-Thinking"
GROQ_MODEL      = "llama-3.3-70b-versatile"   # fast Groq-hosted 70B
BATCH_TOKEN_BUDGET = 7000   # estimated input tokens; above this → sequential

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Tight, bullet-constrained prompt → short output → fewer tokens generated
SUMMARY_SYSTEM_PROMPT = """\
You are a senior code reviewer. Given a git diff for ONE file, output EXACTLY:
- 1 sentence: what changed (be specific, name functions/classes if relevant)
- 1 sentence: likely reason for the change
- Up to 3 bullet points of notable patterns, risks, or issues (skip if none)

Rules:
- Total response MUST be under 120 words
- No preamble, no "Sure!", no restating the filename
- No internal reasoning, no <think> blocks — final answer only
- Use plain text, no markdown headers
"""

FINAL_SYSTEM_PROMPT = """\
You are a technical writer producing a commit review report.

Given per-file summaries, write a structured markdown report with these exact sections:

## Commit Overview
One paragraph (3-5 sentences) summarising the overall intent of the commit.

## Changes Per File
A sub-section per file (### `filename`) with 2-4 bullet points.

## Key Takeaways
3-5 bullets: cross-cutting concerns, risks, follow-up actions.

Rules:
- Total report MUST be under 400 words
- No filler phrases ("In conclusion", "It is worth noting")
- Output markdown only — no preamble, no explanation
"""

# ---------------------------------------------------------------------------
# Global model state — CPU-resident between requests
# ---------------------------------------------------------------------------

_model:     AutoModelForCausalLM | None = None
_tokenizer: AutoTokenizer        | None = None


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (multiline) produced by thinking models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_filename(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith("Filename :"):
            return line.split(":", 1)[1].strip()
    return "unknown"


# ---------------------------------------------------------------------------
# Startup: load Mellum 2 in 6-bit NF4 into CPU RAM
# Runs ONCE before app.launch(), outside any @spaces.GPU context.
# ---------------------------------------------------------------------------

def load_model_on_startup() -> None:
    """
    Load Mellum 2 into CPU RAM with 6-bit NF4 double quantization.
    device_map='cpu' keeps weights off-GPU until a @spaces.GPU call fires,
    satisfying ZeroGPU's requirement that GPU allocation only happens inside
    decorated functions.
    """
    global _model, _tokenizer

    log.info("=== STARTUP: loading tokenizer (%s) ===", MODEL_REPO_ID)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)
    if _tokenizer.pad_token_id is None:
        _tokenizer.pad_token_id = _tokenizer.eos_token_id
    log.info("Tokenizer ready. pad_token_id=%s", _tokenizer.pad_token_id)

    log.info("=== STARTUP: loading model in 6-bit NF4 on CPU ===")
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,   # NF4 + double quant ≈ effective 6-bit
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO_ID,
        quantization_config=quant_cfg,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )
    _model.eval()
    log.info("=== STARTUP: model ready on CPU ===")


# ---------------------------------------------------------------------------
# Mellum inference (called inside @spaces.GPU)
# ---------------------------------------------------------------------------

def _build_mellum_prompt(user_content: str) -> str:
    """Apply Mellum's chat template to a single user turn."""
    return _tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def _generate_batch(prompts: list[str]) -> list[str]:
    """
    Single batched model.generate() call for all prompts.
    Left-padding aligns sequences for parallel decode.
    max_new_tokens is kept low (256) because SUMMARY_SYSTEM_PROMPT
    instructs the model to stay under 120 words.
    """
    log.info("Batch inference: %d prompts", len(prompts))
    _tokenizer.padding_side = "left"
    enc = _tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=3072,          # cap input so batch fits in VRAM
    ).to("cuda")

    log.info("Input shape: %s", enc.input_ids.shape)
    with torch.no_grad():
        out = _model.generate(
            **enc,
            max_new_tokens=2048,   # ≈ 200 words — enough for our tight prompt
            use_cache=True,
            do_sample=False,
            pad_token_id=_tokenizer.pad_token_id,
        )

    results = []
    for seq in out:
        new_tok = seq[enc.input_ids.shape[1]:]
        text = _tokenizer.decode(new_tok, skip_special_tokens=True)
        results.append(_strip_thinking(text))
    return results


def _generate_sequential(prompts: list[str]) -> list[str]:
    """Fallback single-prompt inference when batch would OOM."""
    log.info("Sequential inference: %d prompts", len(prompts))
    _tokenizer.padding_side = "right"
    results = []
    for i, prompt in enumerate(prompts):
        log.info("  [%d/%d]", i + 1, len(prompts))
        enc = _tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = _model.generate(
                **enc,
                max_new_tokens=1024,
                use_cache=True,
                do_sample=False,
                pad_token_id=_tokenizer.pad_token_id,
            )
        text = _tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        results.append(_strip_thinking(text))
    return results


def _smart_generate(prompts: list[str]) -> list[str]:
    """
    Route to batch or sequential based on estimated token count.
    Catches OOM and retries sequentially.
    """
    estimated_tokens = sum(len(p) for p in prompts) // 4
    use_sequential = (len(prompts) == 1) or (estimated_tokens > BATCH_TOKEN_BUDGET)

    if use_sequential:
        log.info("Routing to sequential (est. %d tokens)", estimated_tokens)
        return _generate_sequential(prompts)

    try:
        return _generate_batch(prompts)
    except torch.cuda.OutOfMemoryError:
        log.warning("Batch OOM — retrying sequentially")
        torch.cuda.empty_cache()
        return _generate_sequential(prompts)


# ---------------------------------------------------------------------------
# Groq final report (pure API call — no GPU needed)
# ---------------------------------------------------------------------------

def _generate_final_report_groq(per_file_summaries: list[dict]) -> str:
    """
    Send all per-file summaries to Groq llama-3.3-70b and get back
    a structured markdown commit report. Fast (~2-4 s) and free of GPU cost.

    Reads GROQ_API_KEY from environment (set as a HF Space secret).
    """
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    # Format per-file summaries as a clean user message
    user_content = "\n\n".join(
        f"### `{f['name']}`\n{f['summary']}"
        for f in per_file_summaries
    )

    log.info("Calling Groq %s for final report (%d files) ...", GROQ_MODEL, len(per_file_summaries))
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": FINAL_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        max_tokens=600,       # 400-word cap + small buffer
        temperature=0.2,      # low temp for consistent, factual output
    )

    report = response.choices[0].message.content.strip()
    log.info("Groq report received (%d chars)", len(report))
    return report


# ---------------------------------------------------------------------------
# gradio.Server app
# ---------------------------------------------------------------------------

app = Server()


@app.get("/", response_class=HTMLResponse)
async def homepage():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.api(name="process_repo")
@spaces.GPU(duration=240)      # reduced from 300 — summaries are now much shorter
def process_repo(repo_url: str, token: str) -> dict:
    """
    Full pipeline:
      1. run_pipeline()  → raw per-file prompts            (CPU, fast)
      2. Mellum 2 batch  → per-file summaries (≤120 words) (GPU, batched)
      3. Groq 70B        → final markdown report (≤400 words) (API, ~3 s)

    Returns: { "files": [{"name": str, "summary": str}], "report": str }
    """
    log.info("=== process_repo: %s ===", repo_url)
    _model.to("cuda")   # move model to GPU for Mellum inference; stays in GPU until next @spaces.GPU call or app shutdown
    # Step 1 — fetch diff and build prompts
    prompts = run_pipeline(repo_url, token.strip() or None)
    log.info("Got %d file prompts from pipeline", len(prompts))
    if not prompts:
        raise ValueError("No source-code files changed in the latest commit.")

    fnames = [_extract_filename(p) for p in prompts]

    # Step 2 — per-file summaries via Mellum 2 on GPU
    mellum_prompts = [_build_mellum_prompt(p) for p in prompts]
    summaries = _smart_generate(mellum_prompts)

    file_results = [
        {"name": n, "summary": s}
        for n, s in zip(fnames, summaries)
    ]
    log.info("Per-file summaries done")

    # Step 3 — final report via Groq (outside GPU, but still inside @spaces.GPU
    # context — that's fine, the Groq call is pure HTTP and doesn't touch CUDA)
    final_report = _generate_final_report_groq(file_results)

    log.info("Pipeline complete — %d files", len(file_results))
    return {"files": file_results, "report": final_report}


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------

load_model_on_startup()   # weights land in CPU RAM; GPU untouched until first request

if __name__ == "__main__":
    log.info("Starting CommitLens ...")
    app.launch()
