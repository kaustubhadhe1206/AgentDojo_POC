"""Phase 4: run user tasks with no attack and report utility (did the agent finish the task?).

Done when: you get True/False per task and JSON logs appear under ./logs.
"""

import argparse
import inspect
import time

from agentdojo.agent_pipeline import AgentPipeline
from agentdojo.benchmark import benchmark_suite_without_injections
from agentdojo.logging import OutputLogger
from agentdojo.task_suite.load_suites import get_suite

import config
import tracker
from nim_pipeline import DEFENSES, SYSTEM_PROMPTS, build_pipeline


def run_baseline(
    pipeline: AgentPipeline,
    suite_name: str = "workspace",
    user_tasks: list[str] | None = None,
    force_rerun: bool = False,
    sleep_s: float = config.SLEEP_BETWEEN_TASKS,
) -> dict[str, bool]:
    """Returns {user_task_id: utility}. user_tasks=None runs every task in the suite."""
    suite = get_suite(config.BENCHMARK_VERSION, suite_name)
    task_ids = user_tasks or list(suite.user_tasks)
    utility: dict[str, bool] = {}

    # AgentDojo writes logs to the active logger's folder; `logdir` below is only used to
    # find cached results. Without this context manager no logs would be saved.
    with OutputLogger(str(config.LOGDIR)):
        for n, task_id in enumerate(task_ids):
            if n and sleep_s:
                time.sleep(sleep_s)  # one task at a time, to stay under the rate limit
            results = benchmark_suite_without_injections(
                pipeline,
                suite,
                logdir=config.LOGDIR,
                force_rerun=force_rerun,
                user_tasks=[task_id],
                benchmark_version=config.BENCHMARK_VERSION,
            )
            utility[task_id] = results["utility_results"][(task_id, "")]
    return utility


def print_utility(utility: dict[str, bool], title: str) -> None:
    print(f"\n=== {title} ===")
    for task_id, ok in utility.items():
        print(f"  {task_id:<16} utility={ok}")
    if utility:
        print(f"  Average utility: {100 * sum(utility.values()) / len(utility):.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: baseline run without attacks")
    parser.add_argument("--model", default=config.SMALL_MODEL)
    parser.add_argument("--agent", default="default", choices=list(SYSTEM_PROMPTS))
    parser.add_argument("--defense", default="none", choices=["none", *DEFENSES],
                        help="repeat_prompt = remind the model of the user's request after every tool output")
    parser.add_argument("--suite", default="workspace")
    parser.add_argument("--user-tasks", nargs="+", default=["user_task_0", "user_task_1"])
    parser.add_argument("--force-rerun", action="store_true", help="ignore cached logs")
    parser.add_argument("--show-signature", action="store_true", help="print AgentDojo's function signature and exit")
    args = parser.parse_args()

    if args.show_signature:  # the plan warns this signature changes between versions
        print("benchmark_suite_without_injections" + str(inspect.signature(benchmark_suite_without_injections)))
        return

    config.setup_logging()
    pipeline = build_pipeline(args.model, None if args.defense == "none" else args.defense, args.agent)
    utility = run_baseline(pipeline, args.suite, args.user_tasks, args.force_rerun)
    print_utility(utility, f"Baseline: {pipeline.name} / {args.suite}")
    print(f"Logs: {config.LOGDIR / pipeline.name / args.suite}")
    tracker.rebuild()


if __name__ == "__main__":
    main()
