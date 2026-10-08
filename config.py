"""Shared settings for the AgentDojo POC (NVIDIA NIM / Nemotron)."""

import logging
import os
from pathlib import Path

import openai
from dotenv import load_dotenv
from rich.logging import RichHandler

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")  # real environment variables still take priority

BASE_URL = "https://integrate.api.nvidia.com/v1"
API_KEY_ENVS = ("QWEN_NVIDIA_API_KEY", "NVIDIA_API_KEY")  # first one that is set is used
SMALL_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"  # default for phases 2-5
LARGE_MODEL = "nvidia/nemotron-3-super-120b-a12b"  # second model for the Phase 7 comparison
BENCHMARK_VERSION = "v1.2"
LOGDIR = PROJECT_DIR / "logs"  # one JSON log per run
SLEEP_BETWEEN_TASKS = 2.0  # seconds; keeps free-tier runs under the rate limit


def get_api_key() -> str:
    for name in API_KEY_ENVS:
        key = os.environ.get(name, "").strip()
        if key and key != "your-key-here":
            return key
    raise SystemExit(f"No NVIDIA NIM API key. Set {' or '.join(API_KEY_ENVS)} in .env (see .env.example).")


def make_client() -> openai.OpenAI:
    """OpenAI client pointed at NIM's OpenAI-compatible endpoint. Used by every phase script."""
    # NIM occasionally hangs a request for minutes before returning 504; fail fast and retry instead.
    return openai.OpenAI(api_key=get_api_key(), base_url=BASE_URL, timeout=60)


def setup_logging() -> None:
    """Show AgentDojo's per-message chat log in the console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(show_path=False, markup=True)],
    )
    for noisy in ("httpx", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
