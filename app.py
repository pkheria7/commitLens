"""
CommitLens — gradio.Server mode
================================
- Serves custom index.html at GET /
- Exposes process_repo via @app.api() for the JS frontend to call
- Uses Mellum 2 for per-file summaries + final markdown report
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from fastapi.responses import HTMLResponse
from gradio import Server
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import spaces
from commitlens import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("commitlens")

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

MODEL_REPO_ID = "JetBrains/Mellum2-12B-A2.5B-Thinking"

SUMMARY_SYSTEM_PROMPT = (
    "You are an expert code reviewer. Analyze the following commit change context "
    "and provide a concise, clear summary of what changed in this file, why it might "
    "have changed, and any notable patterns or potential issues."
)

FINAL_SYSTEM_PROMPT = (
    "You are a technical documentation expert. Given the following per-file summaries "
    "of code changes from a commit, produce a well-formatted markdown document that "
    "provides a comprehensive overview of the commit. Include sections for:\n"
    "- Commit Overview\n"
    "- Summary of Changes (per file)\n"
    "- Key Takeaways / Impact\n\n"
    "Use clear markdown formatting (headings, code blocks, bullet lists)."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None


def _get_llm():
    global _model, _tokenizer
    if _model is None:
        log.info("Starting model load from %s ...", MODEL_REPO_ID)
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        log.info("Loading tokenizer ...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)
        log.info("Tokenizer loaded.")

        log.info("Loading model with 8-bit quantization ...")
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO_ID,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        log.info("Model loaded successfully.")
    return _model, _tokenizer


def _extract_filename(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith("Filename :"):
            name = line.split(":", 1)[1].strip()
            return name
    return "unknown"


def _generate_response(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    model, tokenizer = _get_llm()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        use_cache=True,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True
    )
    return response.strip()


def _summarize(prompt: str) -> str:
    return _generate_response(SUMMARY_SYSTEM_PROMPT, prompt, max_tokens=1024)


def _final_md(combined: str) -> str:
    return _generate_response(FINAL_SYSTEM_PROMPT, combined, max_tokens=2048)


# ---------------------------------------------------------------------------
# gradio.Server app
# ---------------------------------------------------------------------------

app = Server()


@app.get("/", response_class=HTMLResponse)
async def homepage():
    """Serve the custom cinematic frontend."""
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.api(name="process_repo")
@spaces.GPU(duration=300)
def process_repo(repo_url: str, token: str) -> dict:
    """
    Run the CommitLens pipeline and return structured results.

    Returns:
        {
          "files": [{"name": str, "summary": str}, ...],
          "report": str   # final markdown
        }
    """
    log.info("Pipeline started for repo: %s", repo_url)
    prompts = run_pipeline(repo_url, token.strip() or None)
    log.info("CommitLens pipeline returned %d prompts.", len(prompts))

    if not prompts:
        raise ValueError("No source-code files changed in the latest commit.")

    file_results = []
    per_file_md_parts = []

    for i, prompt in enumerate(prompts):
        fname = _extract_filename(prompt)
        log.info("Processing file %d/%d: %s", i + 1, len(prompts), fname)
        summary = _summarize(prompt)
        file_results.append({"name": fname, "summary": summary})
        per_file_md_parts.append(f"## `{fname}`\n\n{summary}")
        log.info("Finished file %d/%d: %s", i + 1, len(prompts), fname)

    combined = "\n\n---\n\n".join(per_file_md_parts)
    final_report = _final_md(combined)
    log.info("Pipeline finished successfully.")

    return {"files": file_results, "report": final_report}


if __name__ == "__main__":
    log.info("Starting CommitLens with gradio.Server ...")
    app.launch()
