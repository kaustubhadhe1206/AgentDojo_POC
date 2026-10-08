"""Phase 5: hide attacker instructions in tool outputs and check whether the model follows them.

Done when: you get a utility score and an attack-success score per (user task, injection task).

AgentDojo is the attacker and the judge here, not a defense: it plants the injection, runs
the agent, then checks the environment (e.g. was the attacker's email sent?). In its logs
security=True means the ATTACK SUCCEEDED, so this script prints it as "attack_success".
"""

import argparse
import time

from agentdojo.agent_pipeline import AgentPipeline
from agentdojo.attacks import load_attack  # importing this also registers the built-in attacks
from agentdojo.attacks.attack_registry import ATTACKS
from agentdojo.benchmark import benchmark_suite_with_injections, run_task_with_injection_tasks
from agentdojo.logging import OutputLogger
from agentdojo.task_suite.load_suites import get_suite

import config
import custom_attacks  # noqa: F401  (registers injection_task_14 edit-file and _15 share-file)
import tracker
from nim_pipeline import DEFENSES, SYSTEM_PROMPTS, build_pipeline

ATTACK_CHOICES = sorted(a for a in ATTACKS if a != "manual")  # "manual" needs keyboard input


def run_attack(
    pipeline: AgentPipeline,
    suite_name: str = "workspace",
    attack_name: str = "direct",
    user_tasks: list[str] | None = None,
    injection_tasks: list[str] | None = None,
    force_rerun: bool = False,
    sleep_s: float = config.SLEEP_BETWEEN_TASKS,
    skip_sanity: bool = False,
) -> tuple[dict[tuple[str, str], tuple[bool, bool]], dict[str, bool]]:
    """Returns ({(user_task, injection_task): (utility, attack_success)}, {injection_task: solvable})."""
    suite = get_suite(config.BENCHMARK_VERSION, suite_name)
    attack = load_attack(attack_name, suite, pipeline)
    task_ids = user_tasks or list(suite.user_tasks)
    results: dict[tuple[str, str], tuple[bool, bool]] = {}
    injection_solvable: dict[str, bool] = {}

    with OutputLogger(str(config.LOGDIR)):  # see the note in phase4_no_attack_test.py
        for n, task_id in enumerate(task_ids):
            if n and sleep_s:
                time.sleep(sleep_s)
            if skip_sanity:  # only the attacked run itself: fewest API calls
                utility, security = run_task_with_injection_tasks(
                    suite, pipeline, suite.get_user_task_by_id(task_id), attack, config.LOGDIR,
                    force_rerun, injection_tasks, config.BENCHMARK_VERSION,
                )
                for key, u in utility.items():
                    results[key] = (u, security[key])
                continue
            res = benchmark_suite_with_injections(
                pipeline,
                suite,
                attack,
                logdir=config.LOGDIR,
                force_rerun=force_rerun,
                user_tasks=[task_id],
                injection_tasks=injection_tasks,
                benchmark_version=config.BENCHMARK_VERSION,
            )
            # AgentDojo first asks the model to do each attacker goal as a normal request,
            # to check the goal is achievable at all.
            injection_solvable.update(res["injection_tasks_utility_results"])
            for key, utility in res["utility_results"].items():
                results[key] = (utility, res["security_results"][key])
    return results, injection_solvable


def print_attack_results(results: dict[tuple[str, str], tuple[bool, bool]], title: str) -> None:
    print(f"\n=== {title} ===")
    for (ut, it), (utility, success) in results.items():
        print(f"  {ut:<14} x {it:<18} utility={utility!s:<5}  attack_success={success}")
    if results:
        n = len(results)
        print(f"  Utility under attack: {100 * sum(u for u, _ in results.values()) / n:.1f}%")
        print(f"  Attack success rate:  {100 * sum(s for _, s in results.values()) / n:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: attack run")
    parser.add_argument("--model", default=config.SMALL_MODEL)
    parser.add_argument("--agent", default="default", choices=list(SYSTEM_PROMPTS),
                        help="'vulnerable' = control agent told to obey instructions in tool outputs")
    parser.add_argument("--defense", default="none", choices=["none", *DEFENSES],
                        help="repeat_prompt = remind the model of the user's request after every tool output")
    parser.add_argument("--suite", default="workspace")
    parser.add_argument("--attack", default="direct", choices=ATTACK_CHOICES)
    parser.add_argument("--user-tasks", nargs="+", default=["user_task_0"])
    parser.add_argument("--injection-tasks", nargs="+", default=["injection_task_0"])
    parser.add_argument("--force-rerun", action="store_true", help="ignore cached logs")
    parser.add_argument("--skip-sanity", action="store_true",
                        help="skip AgentDojo's 'is the attacker goal doable?' pre-check (saves ~2 calls per injection task)")
    args = parser.parse_args()

    config.setup_logging()
    pipeline = build_pipeline(args.model, None if args.defense == "none" else args.defense, args.agent)
    results, solvable = run_attack(
        pipeline, args.suite, args.attack, args.user_tasks, args.injection_tasks, args.force_rerun,
        skip_sanity=args.skip_sanity,
    )
    for inj_id, ok in solvable.items():
        # False means the model can't do the attacker's goal even when asked, so ASR is understated.
        print(f"  (sanity) {inj_id} solvable as a normal request: {ok}")
    print_attack_results(results, f"Attack '{args.attack}': {pipeline.name} / {args.suite}")
    tracker.rebuild()


if __name__ == "__main__":
    main()
