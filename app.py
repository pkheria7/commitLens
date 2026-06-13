"""
CommitLens — Gradio UI (Dedicated T4 Medium Version)
"""

from __future__ import annotations

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from commitlens import run_pipeline

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
        # 8-bit quantization is mandatory to fit inside the T4's 16GB VRAM
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)
        
        # CHANGED: torch_dtype set to float16 (T4 does not support bfloat16)
        # CHANGED: Removed Flash Attention (T4 relies on native SDPA instead)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO_ID,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16, 
        )
    return _model, _tokenizer

def _extract_filename(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith("Filename :"):
            return line.split(":", 1)[1].strip()
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
        pad_token_id=tokenizer.eos_token_id 
    )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return response.strip()

def _summarize(prompt: str) -> str:
    return _generate_response(SUMMARY_SYSTEM_PROMPT, prompt, max_tokens=1024)

def _final_md(combined: str) -> str:
    return _generate_response(FINAL_SYSTEM_PROMPT, combined, max_tokens=2048)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

# CHANGED: Removed @spaces.GPU entirely. The T4 is dedicated and always active.
def process_repo(repo_url: str, token: str, progress: gr.Progress = gr.Progress()):
    try:
        progress(0, desc="Running CommitLens pipeline...")
        prompts = run_pipeline(repo_url, token.strip() or None)

        if not prompts:
            raise ValueError("No source-code files changed in the latest commit.")

        per_file_md_parts = []
        for i, prompt in enumerate(prompts):
            fname = _extract_filename(prompt)
            progress(
                (i + 1) / (len(prompts) + 1),
                desc=f"Summarizing [{i+1}/{len(prompts)}] {fname}...",
            )
            summary = _summarize(prompt)
            per_file_md_parts.append(f"## `{fname}`\n\n{summary}")

        combined = "\n\n---\n\n".join(per_file_md_parts)

        progress(0.95, desc="Generating final markdown report...")
        final_md = _final_md(combined)

        return combined, final_md

    except Exception as e:
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
    demo.launch()