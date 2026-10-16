"""Builds the fixed list of tests that every local model runs, so the models are compared fairly.

A test is one of:
  * attacked run:  (suite, user task, injection task, attack). AgentDojo hides the attacker's goal
                   (the injection task) in the data the user task reads, using the given attack style.
  * baseline run:  (suite, user task) with no attack, to check the model can do the task at all.

The plan is random but seeded, so the same settings always produce the same tests.
"""

import math
import random
from dataclasses import asdict, dataclass

from agentdojo.task_suite.load_suites import get_suite

import config
import custom_attacks  # noqa: F401  (registers the extra workspace goals injection_task_14 and _15)

# Banking is left out on purpose: agents are not realistically trusted with bank accounts.
DEFAULT_SUITES = ("workspace", "slack", "travel")
DEFAULT_ATTACKS = ("important_instructions", "tool_knowledge", "ignore_previous", "direct")
TESTS_PER_USER_TASK = 3  # each chosen user task is attacked about this many times, with different goals/attacks


@dataclass(frozen=True)
class PlannedTest:
    suite: str
    user_task: str
    injection_task: str | None = None  # None = baseline run (no attack)
    attack: str | None = None

    @property
    def is_baseline(self) -> bool:
        return self.attack is None

    def label(self) -> str:
        if self.is_baseline:
            return f"{self.suite}/{self.user_task} (no attack)"
        return f"{self.suite}/{self.user_task} x {self.injection_task} ({self.attack})"

    def to_dict(self) -> dict:
        return asdict(self)


def _split_evenly(total: int, parts: int) -> list[int]:
    """_split_evenly(50, 3) -> [17, 17, 16]"""
    return [total // parts + (1 if i < total % parts else 0) for i in range(parts)]


def build_test_plan(
    attacked_tests: int = 50,
    suites: tuple[str, ...] = DEFAULT_SUITES,
    attacks: tuple[str, ...] = DEFAULT_ATTACKS,
    include_baseline: bool = True,
    seed: int = 0,
) -> list[PlannedTest]:
    """Spread `attacked_tests` over the suites, rotating through the attack styles.
    If include_baseline, also run each chosen user task once without an attack."""
    rng = random.Random(seed)
    attacked: list[PlannedTest] = []
    baseline: list[PlannedTest] = []
    attack_index = 0  # rotates through `attacks` across all suites, so each style is used about equally

    for suite_name, tests_in_suite in zip(suites, _split_evenly(attacked_tests, len(suites))):
        suite = get_suite(config.BENCHMARK_VERSION, suite_name)
        user_task_ids = sorted(suite.user_tasks)
        injection_task_ids = sorted(suite.injection_tasks)

        user_task_count = min(len(user_task_ids), math.ceil(tests_in_suite / TESTS_PER_USER_TASK))
        chosen_user_tasks = sorted(rng.sample(user_task_ids, user_task_count),
                                   key=lambda task_id: int(task_id.rsplit("_", 1)[-1]))

        used_combinations: set[tuple[str, str, str]] = set()
        for test_number in range(tests_in_suite):
            user_task = chosen_user_tasks[test_number % user_task_count]
            attack = attacks[attack_index % len(attacks)]
            attack_index += 1
            # Pick an attacker goal not yet paired with this user task and attack.
            remaining_goals = [goal for goal in injection_task_ids if (user_task, goal, attack) not in used_combinations]
            injection_task = rng.choice(remaining_goals)
            used_combinations.add((user_task, injection_task, attack))
            attacked.append(PlannedTest(suite_name, user_task, injection_task, attack))

        if include_baseline:
            baseline.extend(PlannedTest(suite_name, user_task) for user_task in chosen_user_tasks)

    return baseline + attacked


def describe_plan(plan: list[PlannedTest]) -> str:
    """Short text summary: how many tests per suite and per attack."""
    attacked = [test for test in plan if not test.is_baseline]
    baseline_count = len(plan) - len(attacked)
    lines = [f"{len(plan)} runs per model: {len(attacked)} attacked + {baseline_count} baseline (no attack)"]
    for field_name in ("suite", "attack"):
        counts: dict[str, int] = {}
        for test in attacked:
            value = getattr(test, field_name)
            counts[value] = counts.get(value, 0) + 1
        lines.append(f"  attacked tests per {field_name}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return "\n".join(lines)


if __name__ == "__main__":
    test_plan = build_test_plan()
    print(describe_plan(test_plan))
    for planned_test in test_plan:
        print("  ", planned_test.label())
