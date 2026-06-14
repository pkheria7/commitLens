---
title: CommitLens
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.18.0
python_version: '3.12'
app_file: app.py
pinned: true
license: mit
short_description: urn any Git commit into a human-readable engineering report.
---

# CommitLens — AI-Powered Code Review Pipeline

**CommitLens** is a high-performance information extraction and analysis pipeline that transforms raw GitHub diffs into structured, human-readable engineering reports. It uses a hybrid LLM approach: **JetBrains Mellum 2** for deep per-file analysis and **Groq-hosted Llama 3.3** for lightning-fast synthesis.

![CommitLens UI](https://img.shields.io/badge/UI-Custom_HTML/CSS-blue)
![Backend](https://img.shields.io/badge/Backend-Python_/_FastAPI-green)
![LLM](https://img.shields.io/badge/LLM-Mellum_2_+_Groq_Llama_3.3-orange)

## 🚀 Key Features

- **Automated Diff Extraction**: Fetches the two latest commits from any GitHub repository and generates semantic diffs.
- **Top-Impact Filtering**: Automatically identifies and prioritizes the most significant changes (top 2 files by lines changed) to ensure high-signal reviews.
- **Hybrid LLM Pipeline**:
  - **Mellum 2 (12B)**: Performs surgical, per-file code analysis. Optimized with 6-bit NF4 quantization for efficient GPU utilization.
  - **Groq (Llama 3.3 70B)**: Generates a high-level executive summary and key takeaways in milliseconds.
- **Cinematic UI**: A bespoke, low-latency frontend featuring a custom particle engine, real-time status tracking, and a "git-graph" hero visualization.

## 🛠 Tech Stack

- **Core**: Python 3.12, FastAPI, Gradio (Server Mode).
- **ML/Inference**: `transformers`, `bitsandbytes` (4-bit/6-bit quantization), `torch`, `spaces` (ZeroGPU).
- **APIs**: GitHub REST API, Groq Cloud API.
- **Frontend**: Vanilla JavaScript (ES6+), HTML5 Canvas, CSS3 Grid/Flexbox.

## 📂 Project Structure

| File | Purpose |
|------|---------|
| `app.py` | Main application server; manages model lifecycle and GPU/API orchestration. |
| `commitlens.py` | Data pipeline; handles GitHub API interaction, file filtering, and prompt engineering. |
| `index.html` | Custom-built, high-fidelity frontend with interactive Git visualizations. |
| `requirements.txt` | Dependency manifest (requests, gradio, torch, transformers, etc.). |

## ⚙️ How It Works

1. **Extraction**: The `GitHubClient` fetches commit metadata and raw patches.
2. **Filtering**: Files are filtered by extension (keeping source code, ignoring binaries/locks) and sorted by impact.
3. **Mellum Analysis**: The pipeline builds structured prompts containing "Before", "After", and "Diff" blocks. Mellum 2 generates concise summaries for each file.
4. **Groq Synthesis**: Per-file summaries are batched and sent to Groq for a final structured Markdown report including a "Commit Overview" and "Key Takeaways".

## 🛠 Setup & Usage

### Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   ```bash
   export GROQ_API_KEY="your_groq_api_key"
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

### CLI Mode
You can also run the extraction pipeline directly:
```bash
python commitlens.py <github_repo_url> --token <optional_pat> --print-prompts
```

## 📄 License
MIT
