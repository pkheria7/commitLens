"""
CommitLens — Gradio UI
=======================
- Asks for a GitHub repo URL (and optional PAT)
- Runs commitlens.run_pipeline() to get per-file prompts
- Feeds each prompt to Mellum 2 for a per-file summary
- Combines all summaries and asks the model for a final .md report
- Displays everything in the browser
"""

from __future__ import annotations

import logging
import sys

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
        # 8-bit quantization is required to bypass the 16GB ZeroGPU CPU RAM limit
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

        log.info("Loading tokenizer ...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)
        log.info("Tokenizer loaded.")

        # flash_attention_2 removed. PyTorch will automatically use native SDPA.
        log.info("Loading model with 8-bit quantization and device_map='auto' ...")
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO_ID,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16, # ZeroGPU RTX 6000 natively supports bfloat16
        )
        log.info("Model loaded successfully.")
    return _model, _tokenizer


def _extract_filename(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith("Filename :"):
            name = line.split(":", 1)[1].strip()
            log.debug("Extracted filename: %s", name)
            return name
    log.warning("Could not extract filename from prompt")
    return "unknown"


def _generate_response(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    log.info("Generating response (max_tokens=%d) ...", max_tokens)
    model, tokenizer = _get_llm()
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    # Format the prompt using the model's chat template
    log.debug("Applying chat template ...")
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    log.debug("Tokenizing input ...")
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    log.debug("Input shape: %s", inputs.input_ids.shape)
    
    log.info("Running model.generate ...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        use_cache=True,     
        do_sample=False,    
        pad_token_id=tokenizer.eos_token_id 
    )
    log.info("Generation complete.")
    
    # Decode and return just the generated response
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    log.debug("Response length: %d characters", len(response))
    return response.strip()


def _summarize(prompt: str) -> str:
    log.info("Summarizing file ...")
    result = _generate_response(SUMMARY_SYSTEM_PROMPT, prompt, max_tokens=1024)
    log.info("File summarization done.")
    return result


def _final_md(combined: str) -> str:
    log.info("Generating final markdown report ...")
    result = _generate_response(FINAL_SYSTEM_PROMPT, combined, max_tokens=2048)
    log.info("Final markdown report generated.")
    return result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@spaces.GPU(duration=300)
def process_repo(repo_url: str, token: str, progress: gr.Progress = gr.Progress()):
    try:
        log.info("Pipeline started for repo: %s", repo_url)
        progress(0, desc="Running CommitLens pipeline...")
        prompts = run_pipeline(repo_url, token.strip() or None)
        log.info("CommitLens pipeline returned %d prompts.", len(prompts))

        if not prompts:
            log.warning("No source-code files changed in the latest commit.")
            raise ValueError("No source-code files changed in the latest commit.")

        per_file_md_parts = []
        for i, prompt in enumerate(prompts):
            fname = _extract_filename(prompt)
            log.info("Processing file %d/%d: %s", i + 1, len(prompts), fname)
            progress(
                (i + 1) / (len(prompts) + 1),
                desc=f"Summarizing [{i+1}/{len(prompts)}] {fname}...",
            )
            summary = _summarize(prompt)
            per_file_md_parts.append(f"## `{fname}`\n\n{summary}")
            log.info("Finished file %d/%d: %s", i + 1, len(prompts), fname)

        combined = "\n\n---\n\n".join(per_file_md_parts)
        log.info("All per-file summaries combined (%d characters).", len(combined))

        progress(0.95, desc="Generating final markdown report...")
        final_md = _final_md(combined)

        log.info("Pipeline finished successfully.")
        return combined, final_md

    except gr.Error:
        raise
    except Exception as e:
        log.error("Pipeline failed: %s", e, exc_info=True)
        raise gr.Error(str(e))


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

with gr.Blocks(title="CommitLens", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# CommitLens — AI-Powered Commit Analysis")

    with gr.Row():
        repo_url = gr.Textbox(
            label="GitHub Repository URL",
            placeholder="https://github.com/owner/repo",
            scale=2,
        )
        token = gr.Textbox(
            label="GitHub Token (for private repos)",
            type="password",
            placeholder="ghp_... or leave empty for public repos",
            scale=1,
        )

    run_btn = gr.Button("Run Analysis", variant="primary", size="lg")

    with gr.Row():
        per_file_out = gr.Markdown(label="Per-File Summaries")
        final_out = gr.Markdown(label="Final Report (.md)")

    run_btn.click(
        fn=process_repo,
        inputs=[repo_url, token],
        outputs=[per_file_out, final_out],
    )

if __name__ == "__main__":
    log.info("Starting Gradio app ...")
    demo.launch()