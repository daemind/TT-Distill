# ruff: noqa
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath("."))
from src.orchestration.arc_math_solver import solve_task_math
from src.orchestration.arc_solvers import solve_task


def load_arc_tasks():  # type: ignore[no-untyped-def]
    tasks = {}  # type: ignore[var-annotated]
    d = Path("data/training/arc")
    if not d.exists():
        return tasks
    for f in d.glob("*.json"):
        try:
            with open(f) as fd:
                tasks[f.stem] = json.load(fd)
        except Exception:
            pass
    return tasks


def run_diff():  # type: ignore[no-untyped-def]
    tasks = load_arc_tasks()  # type: ignore[no-untyped-call]
    for task_data in tasks.values():
        res_heur = solve_task(task_data)
        res_math = solve_task_math(task_data)

        heur_solved = res_heur["solved"] and all(res_heur["correct"])
        math_solved = res_math["solved"] and all(res_math["correct"])

        if heur_solved and not math_solved:
            pass


if __name__ == "__main__":
    run_diff()  # type: ignore[no-untyped-call]
