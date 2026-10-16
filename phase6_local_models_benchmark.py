"""Phase 6: benchmark several local Hugging Face models with AgentDojo, running tests in parallel.

For each model, one after another (a T4 GPU holds one model at a time):
  1. start a vLLM server for it (OpenAI-compatible API on localhost)
  2. run the whole test plan against it, many tests at once (vLLM batches them on the GPU)
  3. stop the server and move on to the next model
At the end, write per-test results and a per-model summary as CSV files.

Results are cached: AgentDojo writes one JSON log per test, and finished tests are skipped
on the next run. So if Colab disconnects, just run the same command again to continue.

Typical use (see colab/local_models_benchmark.ipynb):
  python phase6_local_models_benchmark.py --download-only --models-dir /content/drive/MyDrive/agentdojo/models
  python phase6_local_models_benchmark.py --models-dir ... --output-dir /content/drive/MyDrive/agentdojo/output
  python phase6_local_models_benchmark.py --plan-only          # just print the tests, no GPU needed
"""

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from agentdojo.agent_pipeline import AgentPipeline
from agentdojo.attacks import load_attack  # importing this also registers the built-in attacks
from agentdojo.attacks.attack_registry import ATTACKS
from agentdojo.benchmark import run_task_with_injection_tasks, run_task_without_injection_tasks
from agentdojo.logging import LOGGER_STACK, OutputLogger
from agentdojo.task_suite.load_suites import get_suite

import config
from local_models import (
    LOCAL_MODELS,
    LocalModel,
    VllmServer,
    build_local_pipeline,
    copy_to_fast_disk,
    download_model,
    get_model,
)
from local_test_plan import DEFAULT_ATTACKS, DEFAULT_SUITES, PlannedTest, build_test_plan, describe_plan

DEFAULT_OUTPUT_DIR = config.PROJECT_DIR / "local_benchmark_output"


@dataclass
class TestResult:
    model: str
    test: PlannedTest
    utility: bool | None  # did the agent complete the user's task? None = the run crashed
    attack_success: bool | None  # did the attacker's goal happen? None for baseline runs or crashes
    error: str = ""
    duration_s: float = 0.0
    log_file: str = ""


class QuietLogger(OutputLogger):
    """Saves AgentDojo's JSON logs like OutputLogger, but prints nothing: with many tests running
    at once, the usual per-message console output would be unreadable."""

    def log(self, *args, **kwargs) -> None:
        pass


def log_path(logdir: Path, pipeline_name: str, test: PlannedTest) -> Path:
    """Where AgentDojo writes (and later looks for) the JSON log of this test."""
    attack_folder = test.attack or "none"
    return logdir / pipeline_name / test.suite / test.user_task / attack_folder / f"{test.injection_task or 'none'}.json"


def read_log(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def is_finished(logdir: Path, pipeline_name: str, test: PlannedTest) -> bool:
    log = read_log(log_path(logdir, pipeline_name, test))
    return log is not None and "utility" in log


def run_one_test(pipeline: AgentPipeline, test: PlannedTest, logdir: Path, force_rerun: bool) -> TestResult:
    """Run a single test in a worker thread and return its scores."""
    # AgentDojo keeps its active loggers in one list that all threads share by default. Giving each
    # worker its own list stops parallel tests from writing into each other's logs.
    LOGGER_STACK.set([])
    started = time.time()
    suite = get_suite(config.BENCHMARK_VERSION, test.suite)
    user_task = suite.get_user_task_by_id(test.user_task)
    utility: bool | None = None
    attack_success: bool | None = None
    error = ""
    try:
        with QuietLogger(str(logdir)):
            if test.is_baseline:
                utility, _ = run_task_without_injection_tasks(
                    suite, pipeline, user_task, logdir, force_rerun, config.BENCHMARK_VERSION
                )
            else:
                attack = load_attack(test.attack, suite, pipeline)
                utility_by_pair, success_by_pair = run_task_with_injection_tasks(
                    suite, pipeline, user_task, attack, logdir, force_rerun,
                    [test.injection_task], config.BENCHMARK_VERSION,
                )
                pair = (test.user_task, test.injection_task)
                # In AgentDojo's results "security=True" means the ATTACK SUCCEEDED.
                utility, attack_success = utility_by_pair[pair], success_by_pair[pair]
    except Exception as exc:  # one broken test must not stop the other 60+
        error = f"{type(exc).__name__}: {exc}"

    # AgentDojo records some failures (e.g. "prompt longer than the context window") in the log
    # and scores them utility=False / security=True. Don't count those as successful attacks.
    path = log_path(logdir, pipeline.name, test)
    log = read_log(path) or {}
    if log.get("error") and not error:
        error = str(log["error"])
    if error:
        attack_success = None
    return TestResult(
        model=pipeline.name,
        test=test,
        utility=utility if not error else None,
        attack_success=attack_success,
        error=error[:500],
        duration_s=round(log.get("duration") or (time.time() - started), 1),
        log_file=str(path),
    )


def run_plan_in_parallel(
    pipeline: AgentPipeline, plan: list[PlannedTest], logdir: Path, workers: int, force_rerun: bool
) -> list[TestResult]:
    """Run every planned test against one model, `workers` tests at a time."""
    results: list[TestResult] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one_test, pipeline, test, logdir, force_rerun): test for test in plan}
        for finished_count, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if result.error:
                outcome = f"ERROR {result.error[:120]}"
            elif result.test.is_baseline:
                outcome = f"utility={result.utility}"
            else:
                outcome = f"utility={result.utility} attack_success={result.attack_success}"
            minutes = (time.time() - started) / 60
            print(f"[{pipeline.name}] {finished_count}/{len(plan)} ({minutes:.1f} min) {result.test.label()}: {outcome}",
                  flush=True)
    return results


def benchmark_model(model: LocalModel, plan: list[PlannedTest], args: argparse.Namespace) -> list[TestResult]:
    """Start the model's server (unless --server-url is given), run the plan, stop the server."""
    logdir = args.output_dir / "logs"

    if args.server_url:  # a server is already running (e.g. started by hand); just use it
        pipeline = build_local_pipeline(model, args.server_url)
        return run_plan_in_parallel(pipeline, plan, logdir, args.workers, args.force_rerun)

    # Skip loading the model at all if every test already has a log (e.g. after a Colab reconnect).
    if not args.force_rerun and all(is_finished(logdir, model.served_name, test) for test in plan):
        print(f"[{model.key}] all {len(plan)} tests already done; reading cached results")
        pipeline = build_local_pipeline(model, "http://localhost:1/v1")  # never called: results are cached
        return run_plan_in_parallel(pipeline, plan, logdir, args.workers, force_rerun=False)

    model_dir = download_model(model, args.models_dir)
    fast_disk_dir = None
    if args.fast_disk_dir:
        fast_disk_dir = copy_to_fast_disk(model_dir, args.fast_disk_dir)
        model_dir = fast_disk_dir

    server = VllmServer(
        model, model_dir,
        log_file=args.output_dir / "server_logs" / f"{model.key}.log",
        vllm_command=args.vllm_command,
        port=args.port,
        max_model_len=args.max_model_len,
        max_parallel_requests=args.workers,
    )
    try:
        with server:
            pipeline = build_local_pipeline(model, server.base_url)
            return run_plan_in_parallel(pipeline, plan, logdir, args.workers, args.force_rerun)
    finally:
        if fast_disk_dir is not None and args.delete_fast_disk_copy:
            import shutil
            shutil.rmtree(fast_disk_dir, ignore_errors=True)  # free local disk for the next model


# ---------- reporting ----------

def percent(values: list[bool]) -> str:
    return f"{100 * sum(values) / len(values):.1f}%" if values else "-"


def summarize(results: list[TestResult]) -> list[dict]:
    """One summary row per model: baseline utility, utility under attack, attack success rate,
    plus the attack success rate per attack style and per suite."""
    summary_rows = []
    attacks = sorted({r.test.attack for r in results if r.test.attack})
    suites = sorted({r.test.suite for r in results})
    for model in dict.fromkeys(r.model for r in results):  # keeps the order models were run in
        model_results = [r for r in results if r.model == model and not r.error]
        baseline = [r.utility for r in model_results if r.test.is_baseline]
        attacked = [r for r in model_results if not r.test.is_baseline]
        row = {
            "model": model,
            "baseline_utility": percent(baseline),
            "utility_under_attack": percent([r.utility for r in attacked]),
            "attack_success_rate": percent([r.attack_success for r in attacked]),
            "attacked_runs": len(attacked),
            "baseline_runs": len(baseline),
            "errors": sum(1 for r in results if r.model == model and r.error),
        }
        for attack in attacks:
            row[f"ASR {attack}"] = percent([r.attack_success for r in attacked if r.test.attack == attack])
        for suite in suites:
            row[f"ASR {suite}"] = percent([r.attack_success for r in attacked if r.test.suite == suite])
        summary_rows.append(row)
    return summary_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_results(results: list[TestResult], output_dir: Path) -> list[dict]:
    results_dir = output_dir / "results"
    test_rows = [
        {
            "model": r.model, "suite": r.test.suite, "user_task": r.test.user_task,
            "injection_task": r.test.injection_task or "", "attack": r.test.attack or "none",
            "utility": r.utility, "attack_success": "" if r.attack_success is None else r.attack_success,
            "error": r.error, "duration_s": r.duration_s, "log_file": r.log_file,
        }
        for r in sorted(results, key=lambda r: (r.model, r.test.suite, r.test.attack or "", r.test.user_task))
    ]
    summary_rows = summarize(results)
    write_csv(results_dir / "test_results.csv", test_rows)
    write_csv(results_dir / "model_summary.csv", summary_rows)
    print(f"\nSaved {results_dir / 'test_results.csv'} and {results_dir / 'model_summary.csv'}")
    return summary_rows


def print_summary(summary_rows: list[dict]) -> None:
    print("\n=== Model comparison (errors excluded) ===")
    print(f"{'model':<28} {'baseline utility':>16} {'utility@attack':>15} {'attack success':>15} {'errors':>7}")
    for row in summary_rows:
        print(f"{row['model']:<28} {row['baseline_utility']:>16} {row['utility_under_attack']:>15} "
              f"{row['attack_success_rate']:>15} {row['errors']:>7}")


# ---------- command line ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6: benchmark local models with AgentDojo, in parallel")
    parser.add_argument("--models", nargs="+", default=list(LOCAL_MODELS), choices=list(LOCAL_MODELS))
    parser.add_argument("--models-dir", type=Path, default=config.PROJECT_DIR / "local_models_cache",
                        help="where models are downloaded (on Colab: a Google Drive folder)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="logs, server logs and result CSVs go here (on Colab: a Google Drive folder)")
    parser.add_argument("--fast-disk-dir", type=Path, default=None,
                        help="copy each model here before loading it (on Colab: /content/models); much faster than Drive")
    parser.add_argument("--delete-fast-disk-copy", action="store_true",
                        help="delete the fast-disk copy after each model, to free local disk space")
    parser.add_argument("--attacked-tests", type=int, default=50, help="attacked tests per model")
    parser.add_argument("--suites", nargs="+", default=list(DEFAULT_SUITES))
    parser.add_argument("--attacks", nargs="+", default=list(DEFAULT_ATTACKS),
                        choices=sorted(a for a in ATTACKS if a != "manual"))
    parser.add_argument("--no-baseline", action="store_true", help="skip the no-attack runs")
    parser.add_argument("--seed", type=int, default=0, help="changes which tasks are picked")
    parser.add_argument("--workers", type=int, default=8, help="tests running at the same time")
    parser.add_argument("--max-model-len", type=int, default=32768, help="context window given to vLLM")
    parser.add_argument("--vllm-command", default="vllm", help='e.g. "/content/vllm-env/bin/vllm"')
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--server-url", default=None,
                        help="use an already running OpenAI-compatible server instead of starting vLLM (one model only)")
    parser.add_argument("--force-rerun", action="store_true", help="ignore cached logs and run everything again")
    parser.add_argument("--download-only", action="store_true", help="download the models, then stop")
    parser.add_argument("--plan-only", action="store_true", help="print the test plan, then stop")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = [get_model(key) for key in args.models]
    if args.server_url and len(models) != 1:
        raise SystemExit("--server-url serves one model; pass exactly one --models value")

    if args.download_only:
        for model in models:
            download_model(model, args.models_dir)
        return

    plan = build_test_plan(args.attacked_tests, tuple(args.suites), tuple(args.attacks),
                           include_baseline=not args.no_baseline, seed=args.seed)
    print(describe_plan(plan))
    if args.plan_only:
        for test in plan:
            print("  ", test.label())
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "test_plan.json").write_text(json.dumps([t.to_dict() for t in plan], indent=2))

    all_results: list[TestResult] = []
    for model in models:
        print(f"\n===== {model.key}: {model.hf_repo} =====")
        try:
            all_results.extend(benchmark_model(model, plan, args))
        except Exception as exc:  # e.g. the model failed to load; keep going with the next one
            print(f"[{model.key}] SKIPPED: {type(exc).__name__}: {exc}")
        if all_results:
            save_results(all_results, args.output_dir)  # save after every model, in case Colab disconnects

    if all_results:
        print_summary(summarize(all_results))


if __name__ == "__main__":
    main()
