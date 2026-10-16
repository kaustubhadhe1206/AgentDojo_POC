# AgentDojo POC (NVIDIA NIM / Nemotron)

Tests prompt-injection attacks against NVIDIA NIM models (`nvidia/nemotron-3.5-lightning-30b-a3b` by default),
with and without the AgentDojo benchmark. Every run is recorded in `experiment_tracker.xlsx`.

**How to run everything: see [TESTING.md](TESTING.md).**

## Files

| File | Role |
|---|---|
| `phase1_check_env.py` | Run: setup check (no API call) |
| `phase2_normal_usage.py` | Run: plain model call → *Normal usage* tab |
| `phase3_quick_attack_test.py` | Run: one-call attack → *Quick attack test (homemade)* tab |
| `phase4_no_attack_test.py` | Run: AgentDojo task, no attack → *No-attack test (AgentDojo)* tab |
| `phase5_full_attack_test.py` | Run: AgentDojo task with attack → *Full attack test (AgentDojo)* tab |
| `phase6_local_models_benchmark.py` | Run (on Colab): Qwen, Llama and Nemotron from Hugging Face, 50 attacked + 18 no-attack tests each, run in parallel → CSVs on Drive |
| `colab/local_models_benchmark.ipynb` | The Colab notebook that drives phase 6 (Drive, install, download, run, results) |
| `local_models.py` | Helper: the local models, model download, vLLM server, AgentDojo agent for local models |
| `local_test_plan.py` | Helper: the fixed list of tests every local model runs (`python local_test_plan.py` prints it) |
| `tracker.py` | Helper: builds the Excel sheet (runs automatically; `python tracker.py` to refresh by hand) |
| `nim_pipeline.py` | Helper: the AgentDojo agent (used by phases 4 and 5) |
| `config.py` | Helper: endpoint, models, API client |
| `custom_attacks.py` | Helper: extra attacker goals (`injection_task_14` edit file, `_15` share file) |
| `setup_env.ps1`, `requirements.txt`, `.env.example` | Setup |
| `agentdojo-plan.pdf` | The original plan |

## Things worth knowing

- **AgentDojo is the attacker and the judge, not a defense.** It hides the attack in its fake data, runs the agent, then checks whether the attacker's goal happened. `attack_success=False` means the model ignored the attack on its own.
- **`--agent vulnerable`** is a control agent told to obey instructions found in tool outputs, so you can see what a successful attack looks like.
- **Fixes made to AgentDojo's OpenAI wrapper for NIM** (in `nim_pipeline.py`):
  - System messages are sent as `system` instead of `developer`.
  - Message content is sent as plain text.
  - Pipelines are named without the `nvidia/` prefix (a `/` breaks AgentDojo's result cache).
  - Nemotron is registered by name so the `important_instructions` and `tool_knowledge` attacks work.
- **Qwen models are retired on NIM** (`HTTP 410`), which is why the project uses Nemotron.
