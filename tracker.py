"""Experiment tracker: keeps every run in one Excel workbook, experiment_tracker.xlsx.

Tabs:
  Summary                   totals and success rates (Excel formulas over the other tabs)
  Normal usage              plain LLM calls, no tools, no attack       (phase2_normal_usage.py)
  Quick attack test (homemade)  one hand-built call with injected text     (phase3_quick_attack_test.py)
  No-attack test (AgentDojo)       AgentDojo runs without injections          (phase4_no_attack_test.py, plus AgentDojo's sanity checks)
  Full attack test (AgentDojo)     AgentDojo runs with injections             (phase5_full_attack_test.py)
  Read me                   what each tab and column means

The workbook is rebuilt from scratch each time: AgentDojo tabs from the JSON logs in ./logs,
the other two tabs from small .jsonl files in ./logs/tracker. So nothing is lost if the
workbook is open in Excel during a run; it is simply refreshed on the next run.

Rebuild by hand:  python tracker.py
Start fresh:      python tracker.py --reset   (moves all past runs to logs_archive/<timestamp>/, nothing is deleted)
Clearing cells in Excel does NOT remove runs: the sheet is rebuilt from ./logs on every run.
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

import config

WORKBOOK = config.PROJECT_DIR / "experiment_tracker.xlsx"
EVENTS_DIR = config.LOGDIR / "tracker"
EVENT_FILES = {"normal": EVENTS_DIR / "normal_usage.jsonl", "raw_attack": EVENTS_DIR / "attack_without_agentdojo.jsonl"}

FONT = "Arial"
HEADER_FILL = PatternFill("solid", fgColor="1F3864")
GOOD_FILL = PatternFill("solid", fgColor="C6EFCE")
BAD_FILL = PatternFill("solid", fgColor="FFC7CE")

# (header, width). Columns whose header is in WRAP get wrapped text.
COLUMNS = {
    "Normal usage": [
        ("Timestamp", 19), ("Model", 38), ("Prompt", 40), ("Response", 60), ("Latency (s)", 11), ("Error", 30),
    ],
    "Quick attack test (homemade)": [
        ("Timestamp", 19), ("Model", 38), ("Agent", 11), ("Attack style", 22), ("Attacker goal", 45),
        ("User request", 40), ("Injected text", 60), ("Model tool calls", 50), ("Model reply", 50),
        ("Attack success", 10), ("Latency (s)", 11), ("Error", 30),
    ],
    "No-attack test (AgentDojo)": [
        ("Timestamp", 19), ("Model", 30), ("Agent", 11), ("Defense", 14), ("Suite", 11), ("Task", 16),
        ("Task type", 16), ("User request", 45), ("Tool calls made", 50), ("Final answer", 50),
        ("Utility", 9), ("LLM calls", 9), ("Duration (s)", 11), ("Error", 30), ("Log file", 40),
    ],
    "Full attack test (AgentDojo)": [
        ("Timestamp", 19), ("Model", 30), ("Agent", 11), ("Defense", 14), ("Suite", 11), ("Attack type", 22),
        ("User task", 13), ("User request", 45), ("Injection task", 15), ("Attacker goal", 45),
        ("Injected text", 60), ("Injection seen", 10), ("Agent's next step", 50), ("All tool calls", 50),
        ("Utility", 9), ("Attack success", 10), ("LLM calls", 9), ("Duration (s)", 11), ("Error", 30), ("Log file", 40),
    ],
}
COLUMNS["Task reference"] = [
    ("Suite", 11), ("Type", 13), ("ID", 18), ("What it says", 80), ("Difficulty", 10),
    ("Tools the correct solution uses", 40), ("Where an attack can be planted", 45), ("Source", 24),
]
WRAP = {"Prompt", "Response", "Attacker goal", "User request", "Injected text", "Model tool calls", "Model reply",
        "Tool calls made", "Final answer", "Agent's next step", "All tool calls", "Error",
        "What it says", "Tools the correct solution uses", "Where an attack can be planted"}
SUITES = ("workspace", "slack", "banking", "travel")
REFERENCE_CACHE = config.PROJECT_DIR / ".cache" / "task_reference.json"


# ---------- recording ----------

def record(kind: str, row: dict) -> None:
    """Append one non-AgentDojo run ('normal' or 'raw_attack') and refresh the workbook."""
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **row}
    with EVENT_FILES[kind].open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    rebuild()


def _read_events(kind: str) -> list[dict]:
    path = EVENT_FILES[kind]
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------- AgentDojo logs -> rows ----------
# Log layout (written by AgentDojo): logs/<pipeline>/<suite>/<user_task>/<attack>/<injection_task|none>.json

def load_runs(logdir: Path) -> list[dict]:
    runs = []
    for path in sorted(logdir.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:  # e.g. a run interrupted mid-write
            continue
        data["_path"] = path
        data["_attack"] = data.get("attack_type") or "none"
        # Injection tasks that AgentDojo ran as normal requests (its "is the goal doable?" check).
        data["_sanity"] = str(data.get("user_task_id", "")).startswith("injection_task")
        runs.append(data)
    return runs


def text_of(content) -> str:
    """Log content is a list of {"type", "content"} blocks (or a plain string in older logs)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(block.get("content") or "" for block in content)


def _normalize(s: str) -> str:
    """Tool outputs are YAML-dumped (wrapped lines, escaped newlines), so the injected text never
    appears verbatim. Compare only lowercase letters and digits instead."""
    s = s.replace("\\n", " ").replace("\\t", " ").replace('\\"', '"')
    return "".join(ch.lower() for ch in s if ch.isalnum())


def find_injection_index(run: dict) -> int | None:
    """Index of the first tool output containing the attacker's text, or None if the agent never saw it."""
    payloads = [_normalize(p) for p in (run.get("injections") or {}).values() if p]
    for i, msg in enumerate(run.get("messages", [])):
        if msg["role"] == "tool" and any(p and p in _normalize(text_of(msg.get("content"))) for p in payloads):
            return i
    return None


def _split_pipeline(name: str) -> tuple[str, str, str]:
    """'nemotron-...-repeat_prompt-vulnerable' -> (model, agent, defense); see build_pipeline."""
    agent = "vulnerable" if name.endswith("-vulnerable") else "default"
    name = name.removesuffix("-vulnerable")
    defense = "repeat_prompt" if name.endswith("-repeat_prompt") else "none"
    return name.removesuffix("-repeat_prompt"), agent, defense


def _calls(messages: list[dict]) -> str:
    out = []
    for m in messages:
        for c in m.get("tool_calls") or []:
            out.append(f"{c.get('function')}({json.dumps(c.get('args'), ensure_ascii=False)})")
    return "\n".join(out)


def _agentdojo_rows() -> tuple[list[list], list[list]]:
    from agentdojo.task_suite.load_suites import get_suite

    import custom_attacks  # noqa: F401  (so the custom goals' text can be looked up)

    goals: dict[tuple[str, str], str] = {}

    def goal(suite: str, task_id: str) -> str:
        if (suite, task_id) not in goals:
            try:
                goals[(suite, task_id)] = get_suite(config.BENCHMARK_VERSION, suite).get_injection_task_by_id(task_id).GOAL
            except Exception:
                goals[(suite, task_id)] = ""
        return goals[(suite, task_id)]

    baseline, attacks = [], []
    for r in load_runs(config.LOGDIR):
        if "tracker" in r["_path"].parts:
            continue
        msgs = r.get("messages", [])
        model, agent, defense = _split_pipeline(r["pipeline_name"])
        user_req = next((text_of(m.get("content")) for m in msgs if m["role"] == "user"), "")
        llm_calls = sum(m["role"] == "assistant" for m in msgs)
        final = text_of(msgs[-1].get("content")) if msgs and msgs[-1]["role"] == "assistant" else ""
        common = [r.get("evaluation_timestamp") or "", model, agent, defense, r["suite_name"]]
        tail = [llm_calls, round(r.get("duration") or 0, 1), r.get("error") or "",
                str(r["_path"].relative_to(config.PROJECT_DIR))]

        if r["_attack"] == "none":
            task_type = "sanity check" if r["_sanity"] else "user task"
            baseline.append(common + [r["user_task_id"], task_type, user_req, _calls(msgs), final,
                                      r.get("utility")] + tail)
        else:
            idx = find_injection_index(r)
            next_step = ""
            if idx is not None and idx + 1 < len(msgs):
                nxt = msgs[idx + 1]
                next_step = _calls([nxt]) or text_of(nxt.get("content"))
            injected = "\n---\n".join(v for v in (r.get("injections") or {}).values() if v)
            attacks.append(common + [r["_attack"], r["user_task_id"], user_req, r.get("injection_task_id"),
                                     goal(r["suite_name"], r.get("injection_task_id") or ""), injected.strip(),
                                     idx is not None, next_step, _calls(msgs), r.get("utility"),
                                     r.get("security")] + tail)
    key = lambda row: row[0]  # newest first
    return sorted(baseline, key=key, reverse=True), sorted(attacks, key=key, reverse=True)


# ---------- task reference (what each user_task / injection_task is) ----------

def _task_reference() -> list[list]:
    """One row per user task and attacker goal in every suite. Slow to compute (~15 s: AgentDojo
    replays each task's correct solution to find the attack slots), so it is cached and only
    recomputed when AgentDojo or custom_attacks.py changes."""
    import importlib.metadata

    custom = config.PROJECT_DIR / "custom_attacks.py"
    key = f"{importlib.metadata.version('agentdojo')}|{config.BENCHMARK_VERSION}|" \
          f"{custom.stat().st_mtime if custom.exists() else 0}"
    try:
        cached = json.loads(REFERENCE_CACHE.read_text(encoding="utf-8"))
        if cached["key"] == key:
            return cached["rows"]
    except (OSError, ValueError, KeyError):
        pass

    from agentdojo.attacks.baseline_attacks import DirectAttack
    from agentdojo.task_suite.load_suites import get_suite

    import custom_attacks  # noqa: F401

    def number(task_id: str) -> int:
        return int(task_id.rsplit("_", 1)[1])

    def tools(task, env) -> str:
        try:
            return ", ".join(dict.fromkeys(c.function for c in task.ground_truth(env)))
        except Exception:
            return ""

    rows = []
    for name in SUITES:
        suite = get_suite(config.BENCHMARK_VERSION, name)
        env = suite.load_and_inject_default_environment({})
        slot_finder = DirectAttack(suite, None)  # only used to replay the correct solution; no LLM involved
        for tid in sorted(suite.user_tasks, key=number):
            task = suite.user_tasks[tid]
            try:
                slots = ", ".join(slot_finder.get_injection_candidates(task))
            except Exception:
                slots = "(none: an attack can't reach this task)"
            rows.append([name, "User task", tid, task.PROMPT, task.DIFFICULTY.name.lower(), tools(task, env),
                         slots, "AgentDojo"])
        for tid in sorted(suite.injection_tasks, key=number):
            task = suite.injection_tasks[tid]
            source = "custom (custom_attacks.py)" if type(task).__module__ == "custom_attacks" else "AgentDojo"
            rows.append([name, "Attacker goal", tid, task.GOAL, task.DIFFICULTY.name.lower(), tools(task, env),
                         "", source])

    REFERENCE_CACHE.parent.mkdir(exist_ok=True)
    REFERENCE_CACHE.write_text(json.dumps({"key": key, "rows": rows}, ensure_ascii=False), encoding="utf-8")
    return rows


# ---------- workbook ----------

def _write_table(ws, columns, rows) -> None:
    headers = [h for h, _ in columns]
    ws.append(headers)
    for row in rows:
        ws.append(["" if v is None else v for v in row])
    for i, (header, width) in enumerate(columns, start=1):
        ws.column_dimensions[get_column_letter(i)].width = width
        cell = ws.cell(row=1, column=i)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF")
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            header = headers[cell.column - 1]
            cell.font = Font(name=FONT, size=10)
            cell.alignment = Alignment(wrap_text=header in WRAP, vertical="top")
            if header in ("Utility", "Attack success") and isinstance(cell.value, bool):
                good = cell.value if header == "Utility" else not cell.value
                cell.fill = GOOD_FILL if good else BAD_FILL
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = ws.dimensions


def _col(tab: str, header: str) -> str:
    letter = get_column_letter([h for h, _ in COLUMNS[tab]].index(header) + 1)
    return f"'{tab}'!${letter}:${letter}"


def _write_summary(ws, attack_rows: list[list]) -> None:
    bold = Font(name=FONT, bold=True)
    ws["A1"] = "Experiment tracker"
    ws["A1"].font = Font(name=FONT, bold=True, size=14)
    ws["A2"] = f"Rebuilt {datetime.now():%Y-%m-%d %H:%M:%S}. All numbers below are formulas over the other tabs."
    ws["A2"].font = Font(name=FONT, italic=True, size=9)

    ws.append([])
    ws.append(["Tab", "Runs", "Successes", "Success rate", "Success means"])
    header_row = ws.max_row
    table = [
        ("Normal usage", "Model", None, "(no score: plain chat)"),
        ("No-attack test (AgentDojo)", "Task type", "Utility", "agent completed the user's task (sanity checks excluded)"),
        ("Quick attack test (homemade)", "Model", "Attack success", "model called the attacker's tool (bad: model was hijacked)"),
        ("Full attack test (AgentDojo)", "Model", "Attack success", "attacker's goal happened, per AgentDojo's check (bad: model was hijacked)"),
    ]
    for tab, count_col, ok_col, meaning in table:
        r = ws.max_row + 1
        ws.cell(r, 1, tab)
        if count_col == "Task type":  # count real user tasks only, not AgentDojo's sanity checks
            ws.cell(r, 2, f'=COUNTIF({_col(tab, count_col)},"user task")')
            ws.cell(r, 3, f'=COUNTIFS({_col(tab, count_col)},"user task",{_col(tab, ok_col)},TRUE)')
        else:
            ws.cell(r, 2, f"=COUNTA({_col(tab, count_col)})-1")
        if ok_col and count_col != "Task type":
            ws.cell(r, 3, f"=COUNTIF({_col(tab, ok_col)},TRUE)")
        if ok_col:
            ws.cell(r, 4, f"=IF(B{r}=0,0,C{r}/B{r})").number_format = "0.0%"
        ws.cell(r, 5, meaning)

    ws.append([])
    ws.append(["Full attack test (AgentDojo), by model / agent / defense / attack"])
    ws.cell(ws.max_row, 1).font = bold
    ws.append(["Model", "Agent", "Defense", "Attack type", "Runs", "Attack successes", "Attack success rate",
               "Utility under attack"])
    breakdown_header = ws.max_row
    tab = "Full attack test (AgentDojo)"
    combos = sorted({(r[1], r[2], r[3], r[5]) for r in attack_rows})
    for model, agent, defense, attack in combos:
        r = ws.max_row + 1
        crit = (f"{_col(tab, 'Model')},A{r},{_col(tab, 'Agent')},B{r},"
                f"{_col(tab, 'Defense')},C{r},{_col(tab, 'Attack type')},D{r}")
        ws.cell(r, 1, model); ws.cell(r, 2, agent); ws.cell(r, 3, defense); ws.cell(r, 4, attack)
        ws.cell(r, 5, f"=COUNTIFS({crit})")
        ws.cell(r, 6, f"=COUNTIFS({crit},{_col(tab, 'Attack success')},TRUE)")
        ws.cell(r, 7, f"=IF(E{r}=0,0,F{r}/E{r})").number_format = "0.0%"
        ws.cell(r, 8, f"=IF(E{r}=0,0,COUNTIFS({crit},{_col(tab, 'Utility')},TRUE)/E{r})").number_format = "0.0%"
    if not combos:
        ws.append(["(no AgentDojo attack runs yet)"])

    for row in ws.iter_rows(min_row=3):
        for cell in row:
            if cell.font != bold and cell.row not in (header_row, breakdown_header):
                cell.font = Font(name=FONT, size=10)
    for r in (header_row, breakdown_header):
        for cell in ws[r]:
            cell.font = Font(name=FONT, bold=True, color="FFFFFF")
            cell.fill = HEADER_FILL
    for col, width in zip("ABCDEFGH", (38, 12, 14, 24, 10, 19, 21, 21)):
        ws.column_dimensions[col].width = width


def _write_readme(ws) -> None:
    lines = [
        ("Tab", "What it holds"),
        ("Normal usage", "phase2_normal_usage.py: one plain prompt to the model, no tools, no attack."),
        ("Quick attack test (homemade)", "phase3_quick_attack_test.py: ONE API call. The conversation is pre-built up to the "
                                     "point where a tool returns a calendar event containing the injected text; the "
                                     "model's single reply is checked for a call to the attacker's tool."),
        ("No-attack test (AgentDojo)", "phase4_no_attack_test.py: AgentDojo runs the full agent loop with no injection. "
                                "'sanity check' rows are AgentDojo asking the attacker's goal as a normal request "
                                "(proves the goal is doable)."),
        ("Full attack test (AgentDojo)", "phase5_full_attack_test.py: AgentDojo plants the injected text in its fake data, runs "
                                  "the full agent loop, then checks the environment (was the email sent, file deleted...)."),
        ("", ""),
        ("Column", "Meaning"),
        ("Agent", "default = 'You are a helpful assistant.'  vulnerable = control prompt that tells the model to obey "
                  "instructions found in tool outputs."),
        ("Defense", "none, or repeat_prompt (reminds the model of the user's request after every tool output)."),
        ("Utility", "TRUE = the agent completed the user's real task (green)."),
        ("Attack success", "TRUE = the attacker's goal was achieved (red). AgentDojo logs call this 'security'."),
        ("Injection seen", "FALSE = the agent never read the tool output that carried the injection."),
        ("Agent's next step", "What the agent did right after reading the injected text."),
        ("LLM calls", "Number of model replies in the run (each one is one API call)."),
        ("Log file", "Full AgentDojo conversation (JSON). Open it to read every message of the run."),
        ("", ""),
        ("Task reference tab", "Every AgentDojo task you can pass to phase4/phase5. 'User task' rows = the user's real "
                               "request (--user-tasks). 'Attacker goal' rows = what the attack tries to make the agent "
                               "do (--injection-tasks). Pick the suite with --suite. Use the filter on Suite / Type."),
        ("Where an attack can be planted", "The spots in AgentDojo's fake data (an email, a calendar event, a file...) "
                                           "the agent reads while doing that user task. The attack text is hidden there."),
        ("Tools the correct solution uses", "The tool calls that correctly complete the task (AgentDojo's answer key)."),
        ("", ""),
        ("Note", "Close this workbook in Excel before running a script, or that run's refresh is skipped "
                 "(the data is kept and appears on the next refresh). Rebuild by hand: python tracker.py. "
                 "Start fresh: python tracker.py --reset (archives past runs; clearing cells in Excel does not)."),
    ]
    for row in lines:
        ws.append(list(row))
    for row in ws.iter_rows():
        for cell in row:
            cell.font = Font(name=FONT, size=10, bold=cell.value in ("Tab", "Column", "Note"))
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 110


def rebuild() -> Path | None:
    baseline, attacks = _agentdojo_rows()
    normal = [[e.get(h, "") for h, _ in COLUMNS["Normal usage"]] for e in reversed(_read_events("normal"))]
    raw = [[e.get(h, "") for h, _ in COLUMNS["Quick attack test (homemade)"]] for e in reversed(_read_events("raw_attack"))]

    wb = Workbook()
    _write_summary(wb.active, attacks)
    wb.active.title = "Summary"
    for tab, rows in (("Normal usage", normal), ("Quick attack test (homemade)", raw),
                      ("No-attack test (AgentDojo)", baseline), ("Full attack test (AgentDojo)", attacks)):
        _write_table(wb.create_sheet(tab), COLUMNS[tab], rows)
    _write_table(wb.create_sheet("Task reference"), COLUMNS["Task reference"], _task_reference())
    _write_readme(wb.create_sheet("Read me"))
    wb.calculation.fullCalcOnLoad = True  # Excel computes the Summary formulas when the file opens

    try:
        wb.save(WORKBOOK)
    except PermissionError:
        print()
        print("!" * 78)
        print(f"[tracker] NOT UPDATED: {WORKBOOK.name} is open in Excel. The run is saved;")
        print("[tracker] close the workbook (don't save it) and run:  python tracker.py")
        print("!" * 78)
        return None
    print(f"[tracker] updated {WORKBOOK.name}")
    return WORKBOOK


def reset() -> None:
    """Move every past run into logs_archive/<timestamp>/ and rebuild an empty workbook."""
    if config.LOGDIR.exists() and any(config.LOGDIR.iterdir()):
        archive = config.PROJECT_DIR / "logs_archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
        archive.parent.mkdir(exist_ok=True)
        shutil.move(str(config.LOGDIR), str(archive))
        print(f"[tracker] past runs moved to {archive.relative_to(config.PROJECT_DIR)}")
    rebuild()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild experiment_tracker.xlsx")
    parser.add_argument("--reset", action="store_true", help="archive all past runs and start with an empty sheet")
    if parser.parse_args().reset:
        reset()
    else:
        rebuild()
