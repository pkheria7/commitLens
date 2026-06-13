"""
CommitLens — Information Extraction Pipeline
=============================================
Fetches the two latest commits from a GitHub repository, diffs them,
and returns a list of per-file structured prompts ready for Melum 2.

Usage:
    python commitlens.py <github_repo_url> [--token <PAT>]

Example:
    python commitlens.py https://github.com/psf/requests --token ghp_xxx
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import requests  # pip install requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GITHUB_API = "https://api.github.com"

# Source-code extensions to keep (lower-cased, with leading dot)
KEEP_EXTENSIONS: set[str] = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".cpp", ".c", ".h", ".hpp",
    ".go", ".rs", ".php", ".rb", ".cs",
    ".swift", ".kt", ".scala", ".r", ".m",
    ".sh", ".bash", ".zsh",
}

# Config / infra filenames to keep regardless of extension
KEEP_FILENAMES: set[str] = {
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "requirements.txt",
    "package.json",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "cargo.toml",
    "go.mod",
    ".env.example",
    "makefile",
}

# Extensions/patterns to ignore explicitly
IGNORE_EXTENSIONS: set[str] = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".pdf", ".docx", ".xlsx",
    ".zip", ".tar", ".gz", ".bz2", ".7z",
    ".exe", ".dll", ".so", ".dylib",
    ".woff", ".woff2", ".ttf", ".eot",
    ".lock",          # package-lock.json, yarn.lock, Cargo.lock, etc.
    ".map",           # source maps
    ".min.js",        # minified assets (handled via endswith below)
}

IGNORE_FILENAME_PATTERNS: tuple[str, ...] = (
    ".min.js",
    ".min.css",
    "-lock.json",
    ".lock",
    ".pb",            # protobuf binaries
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FileContext:
    filename: str
    status: str          # added | modified | removed | renamed
    additions: int
    deletions: int
    patch: Optional[str]
    before_content: Optional[str] = None   # full file at old commit
    after_content: Optional[str] = None    # full file at new commit


@dataclass
class CommitContext:
    message: str
    author: str
    timestamp: str
    old_sha: str
    new_sha: str
    total_additions: int
    total_deletions: int
    total_changed_files: int
    files: list[FileContext] = field(default_factory=list)


# ---------------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------------

class GitHubClient:
    def __init__(self, token: Optional[str] = None):
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/vnd.github+json"
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def _get(self, url: str, params: Optional[dict] = None) -> dict | list:
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_raw(self, url: str) -> str:
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text

    # ---- domain methods ---------------------------------------------------

    def latest_two_shas(self, owner: str, repo: str) -> tuple[str, str]:
        """Return (old_sha, new_sha) for the two most recent commits."""
        commits = self._get(
            f"{GITHUB_API}/repos/{owner}/{repo}/commits",
            params={"per_page": 2},
        )
        if len(commits) < 2:
            raise ValueError("Repository has fewer than 2 commits.")
        new_sha: str = commits[0]["sha"]
        old_sha: str = commits[1]["sha"]
        return old_sha, new_sha

    def compare(self, owner: str, repo: str, old: str, new: str) -> dict:
        return self._get(
            f"{GITHUB_API}/repos/{owner}/{repo}/compare/{old}...{new}"
        )

    def file_content_at(self, owner: str, repo: str, path: str, ref: str) -> Optional[str]:
        """Return raw file content at a specific commit ref, or None if missing."""
        try:
            meta = self._get(
                f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}",
                params={"ref": ref},
            )
            download_url: str = meta.get("download_url", "")
            if not download_url:
                return None
            return self.get_raw(download_url)
        except requests.HTTPError:
            return None


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _should_keep(filename: str) -> bool:
    lower = filename.lower()
    basename = lower.split("/")[-1]

    # Explicit ignore patterns take priority
    for pat in IGNORE_FILENAME_PATTERNS:
        if lower.endswith(pat):
            return False

    # Keep by exact filename match (infra / config files)
    if basename in KEEP_FILENAMES:
        return True

    # Keep by extension
    for ext in KEEP_EXTENSIONS:
        if lower.endswith(ext):
            return True

    # Ignore by extension
    for ext in IGNORE_EXTENSIONS:
        if lower.endswith(ext):
            return False

    # Default: skip unknown binary-looking files
    return False


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def parse_repo_url(url: str) -> tuple[str, str]:
    """Extract (owner, repo) from a GitHub URL."""
    parsed = urlparse(url.rstrip("/"))
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Cannot parse owner/repo from URL: {url}")
    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    return owner, repo


def fetch_commit_context(
    client: GitHubClient,
    owner: str,
    repo: str,
) -> CommitContext:
    """Steps 2–6: fetch SHAs, comparison, file contents."""

    # Step 2 — latest two SHAs
    old_sha, new_sha = client.latest_two_shas(owner, repo)

    # Step 3 — comparison
    comparison = client.compare(owner, repo, old_sha, new_sha)

    commit_meta = comparison["commits"][-1]["commit"]
    message: str = commit_meta["message"]
    author: str = commit_meta["author"]["name"]
    timestamp: str = commit_meta["author"]["date"]

    stats = comparison.get("stats", {})
    total_additions: int = stats.get("additions", 0)
    total_deletions: int = stats.get("deletions", 0)
    raw_files: list[dict] = comparison.get("files", [])
    total_changed_files: int = len(raw_files)

    # Step 4 — filter files
    filtered: list[dict] = [f for f in raw_files if _should_keep(f["filename"])]

    # --- NEW: Sort by total changes (additions + deletions) descending and take top 2 ---
    filtered = sorted(
        filtered, 
        key=lambda x: x.get("additions", 0) + x.get("deletions", 0), 
        reverse=True
    )[:2]

    # Step 5 + 6 — build FileContext, fetch before/after content
    file_contexts: list[FileContext] = []
    for f in filtered:
        filename: str = f["filename"]
        status: str = f.get("status", "modified")

        fc = FileContext(
            filename=filename,
            status=status,
            additions=f.get("additions", 0),
            deletions=f.get("deletions", 0),
            patch=f.get("patch"),
        )

        # Fetch full file content for semantic context (Step 6)
        if status != "added":
            fc.before_content = client.file_content_at(owner, repo, filename, old_sha)
        if status != "removed":
            fc.after_content = client.file_content_at(owner, repo, filename, new_sha)

        file_contexts.append(fc)

    return CommitContext(
        message=message,
        author=author,
        timestamp=timestamp,
        old_sha=old_sha,
        new_sha=new_sha,
        total_additions=total_additions,
        total_deletions=total_deletions,
        total_changed_files=total_changed_files,
        files=file_contexts,
    )


# ---------------------------------------------------------------------------
# Step 7 — Build per-file prompts
# ---------------------------------------------------------------------------

def build_prompts(ctx: CommitContext) -> list[str]:
    """
    Return one structured prompt string per changed file.
    Each prompt contains:
      - commit-level header (message, author, timestamp, stats)
      - file-specific info (status, additions/deletions)
      - before/after content (if available)
      - the diff patch
    """
    prompts: list[str] = []

    commit_header = (
        "=== COMMIT METADATA ===\n"
        f"Message   : {ctx.message}\n"
        f"Author    : {ctx.author}\n"
    )

    for fc in ctx.files:
        sections: list[str] = [commit_header]

        # File identity
        sections.append(
            "=== FILE ===\n"
            f"Filename : {fc.filename}\n"
        )

        # Before content
        if fc.before_content is not None:
            sections.append(
                "=== BEFORE CODE ===\n"
                f"{fc.before_content}\n"
            )
        else:
            sections.append("=== BEFORE CODE ===\n(file did not exist)\n")

        # After content
        # if fc.after_content is not None:
        #     sections.append(
        #         "=== AFTER CODE ===\n"
        #         f"{fc.after_content}\n"
        #     )
        # else:
        #     sections.append("=== AFTER CODE ===\n(file was deleted)\n")

        # Diff patch
        if fc.patch:
            sections.append(
                "=== DIFF ===\n"
                f"{fc.patch}\n"
            )
        else:
            sections.append("=== DIFF ===\n(no patch available)\n")

        prompts.append("\n".join(sections))

    return prompts


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    repo_url: str,
    token: Optional[str] = None,
) -> list[str]:
    """
    Full CommitLens pipeline.

    Parameters
    ----------
    repo_url : str
        GitHub repository URL, e.g. ``https://github.com/owner/repo``
    token : str, optional
        GitHub personal access token for authenticated requests (higher rate
        limits, private repos).

    Returns
    -------
    list[str]
        One prompt string per changed source-code file.
    """
    owner, repo = parse_repo_url(repo_url)
    client = GitHubClient(token=token)
    ctx = fetch_commit_context(client, owner, repo)
    return build_prompts(ctx)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="CommitLens: extract per-file commit prompts for Melum 2",
    )
    parser.add_argument("repo_url", help="GitHub repository URL")
    parser.add_argument(
        "--token", "-t",
        default=None,
        help="GitHub Personal Access Token (optional but recommended)",
    )
    parser.add_argument(
        "--print-prompts", "-p",
        action="store_true",
        help="Print all generated prompts to stdout",
    )
    args = parser.parse_args()

    try:
        prompts = run_pipeline(args.repo_url, token=args.token)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[CommitLens] Generated {len(prompts)} prompt(s).\n")

    if args.print_prompts:
        for i, prompt in enumerate(prompts, 1):
            print(f"{'='*60}")
            print(f"PROMPT {i} / {len(prompts)}")
            print(f"{'='*60}")
            print(prompt)
            print()
    else:
        for i, prompt in enumerate(prompts, 1):
            # Print just the file header so the caller sees what was captured
            first_line = [
                line for line in prompt.splitlines()
                if line.startswith("Filename :")
            ]
            label = first_line[0] if first_line else f"File {i}"
            print(f"  [{i}] {label}")

    # Return value is available when imported as a module
    return prompts


if __name__ == "__main__":
    _cli()
