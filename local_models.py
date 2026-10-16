"""Local (Hugging Face) models for AgentDojo, served with vLLM on a single GPU (e.g. Colab T4).

What this module does:
  * LOCAL_MODELS         the models we test, and how vLLM must serve each one
  * download_model()     downloads a model from Hugging Face into a folder (e.g. on Google Drive)
  * VllmServer           starts / stops a vLLM server that exposes the model as an
                         OpenAI-compatible API at http://localhost:<port>/v1
  * build_local_pipeline()  the same AgentDojo agent as nim_pipeline.py, pointed at that server

vLLM speaks the same API as NVIDIA NIM, so the agent itself is unchanged; only the address differs.
"""

import shlex
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import openai
from agentdojo.agent_pipeline import AgentPipeline, InitQuery, OpenAILLM, SystemMessage, ToolsExecutionLoop, ToolsExecutor
from agentdojo.agent_pipeline.llms.openai_llm import _function_to_openai, _message_to_openai, _openai_to_assistant_message
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.models import MODEL_NAMES
from agentdojo.types import ChatMessage
from openai._types import NOT_GIVEN

from nim_pipeline import SYSTEM_PROMPTS, _to_nim_message

# A file that marks a finished download, so an interrupted download is retried next time.
DOWNLOAD_COMPLETE_MARKER = ".download_complete"

# Settings shared by every model. The Colab T4 has no bfloat16 support, hence "half" (float16),
# and --enforce-eager skips CUDA-graph capture, which is unreliable on that older GPU.
COMMON_VLLM_ARGS = ("--dtype", "half", "--enforce-eager", "--gpu-memory-utilization", "0.90")


@dataclass(frozen=True)
class LocalModel:
    """One model to test, and everything needed to download and serve it."""

    key: str  # short name used on the command line, e.g. "qwen"
    hf_repo: str  # Hugging Face repository to download
    served_name: str  # model name in API requests; also the AgentDojo pipeline name and log folder
    prose_name: str  # how AgentDojo's stronger attacks address the model ("... to you, Qwen")
    vllm_args: tuple[str, ...] = ()  # model-specific vLLM flags; "{model_dir}" is replaced by the model folder
    extra_files: dict[str, str] = field(default_factory=dict)  # file name -> URL, saved next to the model
    request_extras: dict = field(default_factory=dict)  # extra fields sent with every chat request


LOCAL_MODELS: dict[str, LocalModel] = {
    "qwen": LocalModel(
        key="qwen",
        hf_repo="Qwen/Qwen3-8B-AWQ",  # 4-bit, about 6 GB
        served_name="qwen3-8b-awq",
        prose_name="Qwen",
        vllm_args=("--enable-auto-tool-choice", "--tool-call-parser", "hermes"),
        # Qwen3 "thinks" before answering by default; turning that off keeps runs fast on a T4.
        request_extras={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    "llama": LocalModel(
        key="llama",
        hf_repo="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",  # 4-bit, about 6 GB, not gated
        served_name="llama-3.1-8b-instruct-awq",
        prose_name="Llama",
        vllm_args=(
            "--enable-auto-tool-choice", "--tool-call-parser", "llama3_json",
            "--chat-template", "{model_dir}/tool_chat_template_llama3.1_json.jinja",
        ),
        # vLLM's recommended chat template for Llama 3.1 tool calling.
        extra_files={
            "tool_chat_template_llama3.1_json.jinja":
                "https://raw.githubusercontent.com/vllm-project/vllm/main/examples/tool_chat_template_llama3.1_json.jinja",
        },
    ),
    "nemotron": LocalModel(
        key="nemotron",
        hf_repo="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",  # about 8 GB; the only Nemotron 3 small enough for a T4
        served_name="nemotron-3-nano-4b",
        prose_name="Nemotron",
        # Flags from the model card. The reasoning-parser plugin file ships inside the HF repo.
        vllm_args=(
            "--trust-remote-code", "--mamba_ssm_cache_dtype", "float32",
            "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
            "--reasoning-parser-plugin", "{model_dir}/nano_v3_reasoning_parser.py", "--reasoning-parser", "nano_v3",
        ),
        request_extras={"chat_template_kwargs": {"enable_thinking": False}},
    ),
}


def get_model(key: str) -> LocalModel:
    if key not in LOCAL_MODELS:
        raise ValueError(f"Unknown model {key!r}; choose from {list(LOCAL_MODELS)}")
    return LOCAL_MODELS[key]


# ---------- downloading ----------

def download_model(model: LocalModel, models_dir: Path) -> Path:
    """Download the model into models_dir/<key> (skipped if already complete). Returns the folder."""
    from huggingface_hub import snapshot_download  # only needed where models are downloaded (Colab)

    model_dir = models_dir / model.key
    if (model_dir / DOWNLOAD_COMPLETE_MARKER).exists():
        print(f"[download] {model.key}: already downloaded in {model_dir}")
        return model_dir

    print(f"[download] {model.key}: downloading {model.hf_repo} -> {model_dir} (this can take a while)")
    snapshot_download(repo_id=model.hf_repo, local_dir=model_dir)  # resumes partial downloads
    for file_name, url in model.extra_files.items():
        urllib.request.urlretrieve(url, model_dir / file_name)
    (model_dir / DOWNLOAD_COMPLETE_MARKER).write_text(model.hf_repo)
    print(f"[download] {model.key}: done")
    return model_dir


def copy_to_fast_disk(model_dir: Path, fast_disk_dir: Path) -> Path:
    """Loading weights straight from Google Drive is very slow; copy them to the local disk first."""
    local_dir = fast_disk_dir / model_dir.name
    if not (local_dir / DOWNLOAD_COMPLETE_MARKER).exists():
        print(f"[copy] {model_dir} -> {local_dir}")
        shutil.copytree(model_dir, local_dir, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(DOWNLOAD_COMPLETE_MARKER, ".cache"))
        (local_dir / DOWNLOAD_COMPLETE_MARKER).write_text("copied")  # written last, so a cut-off copy is redone
    return local_dir


# ---------- serving ----------

class VllmServer:
    """Runs `vllm serve` in the background for one model. Use it as a context manager:

        with VllmServer(model, model_dir) as server:
            pipeline = build_local_pipeline(model, server.base_url)
    """

    def __init__(
        self,
        model: LocalModel,
        model_dir: Path,
        log_file: Path,
        vllm_command: str = "vllm",
        port: int = 8000,
        max_model_len: int = 32768,
        max_parallel_requests: int = 8,
        startup_timeout_s: int = 1200,
    ) -> None:
        self.model = model
        self.model_dir = model_dir
        self.log_file = log_file
        self.vllm_command = vllm_command
        self.port = port
        self.max_model_len = max_model_len
        self.max_parallel_requests = max_parallel_requests
        self.startup_timeout_s = startup_timeout_s
        self.process: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"

    def command(self) -> list[str]:
        model_specific = [arg.replace("{model_dir}", str(self.model_dir)) for arg in self.model.vllm_args]
        return [
            *shlex.split(self.vllm_command), "serve", str(self.model_dir),
            "--served-model-name", self.model.served_name,
            "--port", str(self.port),
            "--max-model-len", str(self.max_model_len),
            "--max-num-seqs", str(self.max_parallel_requests),
            *COMMON_VLLM_ARGS,
            *model_specific,
        ]

    def start(self) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"[server] starting vLLM for {self.model.key} (log: {self.log_file})")
        log_handle = self.log_file.open("w", encoding="utf-8")
        self.process = subprocess.Popen(self.command(), stdout=log_handle, stderr=subprocess.STDOUT)
        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        """Poll the /v1/models endpoint until the model is loaded, or fail with the log's last lines."""
        deadline = time.time() + self.startup_timeout_s
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"vLLM exited while starting {self.model.key}:\n{self.log_tail()}")
            try:
                with urllib.request.urlopen(f"{self.base_url}/models", timeout=5) as response:
                    if response.status == 200:
                        print(f"[server] {self.model.key} is ready at {self.base_url}")
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass  # still loading
            time.sleep(5)
        self.stop()
        raise TimeoutError(f"vLLM did not start {self.model.key} within {self.startup_timeout_s}s:\n{self.log_tail()}")

    def log_tail(self, line_count: int = 30) -> str:
        if not self.log_file.exists():
            return "(no log)"
        lines = self.log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-line_count:])

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        print(f"[server] stopping {self.model.key}")
        self.process.terminate()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        time.sleep(5)  # give the GPU a moment to release memory before the next model loads

    def __enter__(self) -> "VllmServer":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()


# ---------- the AgentDojo agent ----------

class LocalLLM(OpenAILLM):
    """AgentDojo's OpenAILLM for a local vLLM server: NIM-style message fixes, plus per-model
    request extras (e.g. turning off "thinking") and a cap on reply length."""

    def __init__(self, client: openai.OpenAI, model: str, request_extras: dict, max_reply_tokens: int = 2048) -> None:
        super().__init__(client, model, temperature=0.0)  # temperature 0 = repeatable results
        self.request_extras = request_extras
        self.max_reply_tokens = max_reply_tokens

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        api_messages = [_to_nim_message(_message_to_openai(m, self.model)) for m in messages]
        api_tools = [_function_to_openai(tool) for tool in runtime.functions.values()]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            tools=api_tools or NOT_GIVEN,
            tool_choice="auto" if api_tools else NOT_GIVEN,
            temperature=self.temperature,
            max_tokens=self.max_reply_tokens,
            extra_body=self.request_extras or None,
        )
        output = _openai_to_assistant_message(completion.choices[0].message)
        return query, runtime, env, [*messages, output], extra_args


def build_local_pipeline(model: LocalModel, base_url: str) -> AgentPipeline:
    """Same chain as nim_pipeline.build_pipeline:
    SystemMessage -> InitQuery -> LLM -> ToolsExecutionLoop([ToolsExecutor, LLM])."""
    # vLLM ignores the API key, but the OpenAI client requires one. Long timeout: a busy T4 is slow.
    client = openai.OpenAI(api_key="not-needed", base_url=base_url, timeout=600, max_retries=2)
    llm = LocalLLM(client, model.served_name, model.request_extras)
    pipeline = AgentPipeline([
        SystemMessage(SYSTEM_PROMPTS["default"]),
        InitQuery(),
        llm,
        ToolsExecutionLoop([ToolsExecutor(), llm]),
    ])
    pipeline.name = model.served_name  # becomes the log folder name
    MODEL_NAMES.setdefault(model.served_name, model.prose_name)  # lets important_instructions / tool_knowledge work
    return pipeline
