---
title: CommitLens
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.0"
python_version: "3.12"
app_file: app.py
pinned: false
---

# CommitLens — AI-Powered Commit Analysis

Analyses the latest commit of any GitHub repository and generates per-file summaries plus a final markdown report using [JetBrains Mellum 2](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Thinking-GGUF-Q8_0).

## How it works

1. **Enter a GitHub repo URL** (public or private)
2. **Optionally provide a GitHub token** for higher rate limits or private repos
3. **Click *Run Analysis*** — the app:
   - Fetches the two most recent commits via the GitHub API
   - Filters for source-code files (`.py`, `.js`, `.ts`, `.go`, `.rs`, etc.)
   - Builds a structured prompt per file (commit metadata + before/after code + diff)
   - Sends each prompt to **Mellum 2** for a concise per-file summary
   - Combines all summaries and asks the model again for a comprehensive markdown report
4. **View results** — per-file summaries and the final `.md` report side by side

## Model

This Space loads the **Mellum2-12B-A2.5B-Thinking-Q8_0** GGUF model (~7 GB) via `llama-cpp-python` on startup. First load may take several minutes.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Gradio application entry point |
| `commitlens.py` | GitHub API pipeline (fetch, diff, filter, prompt builder) |
| `requirements.txt` | Python dependencies |

## Requirements

- `requests`
- `gradio>=4.0.0`
- `llama-cpp-python>=0.2.0`
