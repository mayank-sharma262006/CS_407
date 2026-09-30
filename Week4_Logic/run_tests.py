"""Task 3: tests A, B and C for the planner. Writes results/test_results.txt.

Plans are validated by check_plan(), which looks up each action's definition
in a table written out by hand below rather than reusing the Action objects
or apply() from planner.py. A mistake in the planner's action model would
therefore show up as a disagreement instead of being repeated.

Run:  python run_tests.py
"""

import re
from pathlib import Path

from planner import GOAL, INITIAL_STATE, find_plan, satisfies, show, warehouse_actions

LINKS = {("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")}


def step(state, name):
    """Apply one named action by the handout's rules. Returns None if illegal."""
    kind, first, second = re.fullmatch(r"(\w+)\((\w+),(\w+)\)", name).groups()
    state = set(state)

    if kind == "Move":
        if (first, second) not in LINKS or f"At(Robot,{first})" not in state:
            return None
        state.remove(f"At(Robot,{first})")
        state.add(f"At(Robot,{second})")
    elif kind == "PickUp":
        if not {f"At(Robot,{second})", f"At(Package,{second})"} <= state:
            return None
        state.remove(f"At(Package,{second})")
        state.add("Holding(Package)")
    elif kind == "Drop":
        if not {f"At(Robot,{second})", "Holding(Package)"} <= state:
            return None
        state.remove("Holding(Package)")
        state.add(f"At(Package,{second})")
    else:
        return None
    return state


def check_plan(initial, goal, names):
    """True when every action is legal in turn and the final state meets the goal."""
    state = set(initial)
    for name in names:
        state = step(state, name)
        if state is None:
            return False
    return set(goal) <= state


def run_case(title, note, actions, expect_plan, lines):
    plan = find_plan(INITIAL_STATE, GOAL, actions)
    names = [action.name for action in plan] if plan else []
    found = plan is not None
    valid = check_plan(INITIAL_STATE, GOAL, names) if found else None
    passed = (found and valid) if expect_plan else not found

    lines += [
        title,
        f"  Setup: {note}",
        f"  Actions available: {', '.join(action.name for action in actions)}",
        f"  Initial state: {show(INITIAL_STATE)}",
        f"  Goal: {show(GOAL)}",
        f"  Plan found: {'Yes' if found else 'No plan found'}",
        f"  Plan: {' -> '.join(names) if found else '-'}",
        f"  Plan valid (independent check): {valid if found else 'n/a'}",
        f"  Expected: {'a valid plan' if expect_plan else 'no plan'}",
        f"  Result: {'PASS' if passed else 'FAIL'}",
        "",
    ]
    return passed


def main():
    lines = ["LOGIC LAB - PLANNER TEST RESULTS", ""]
    results = []

    results.append(
        run_case(
            "Test A - solvable problem",
            "original warehouse problem",
            warehouse_actions(),
            True,
            lines,
        )
    )
    results.append(
        run_case(
            "Test B - impossible problem",
            "every PickUp action removed",
            warehouse_actions(with_pick_up=False),
            False,
            lines,
        )
    )
    results.append(
        run_case(
            "Test C - irrelevant actions",
            "only Move actions: the robot can reach C, the package cannot",
            warehouse_actions(with_pick_up=False, with_drop=False),
            False,
            lines,
        )
    )

    # Test C, second half: the state "robot at C, package still at A" is
    # reachable with Move alone and must not count as the goal.
    robot_only = frozenset({"At(Robot,C)", "At(Package,A)"})
    not_goal = not satisfies(robot_only, GOAL)
    results.append(not_goal)
    lines += [
        "Test C - goal check on the robot-only state",
        f"  State: {show(robot_only)}",
        f"  Treated as goal: {not not_goal}",
        "  Expected: False",
        f"  Result: {'PASS' if not_goal else 'FAIL'}",
        "",
    ]

    # The independent checker itself must reject bad plans.
    bad_plans = {
        "Move(A,B) -> Move(B,C)": ["Move(A,B)", "Move(B,C)"],
        "Move(A,C) -> Drop(Package,C)": ["Move(A,C)", "Drop(Package,C)"],
        "Move(A,B) -> PickUp(Package,B) -> Move(B,C) -> Drop(Package,C)": [
            "Move(A,B)",
            "PickUp(Package,B)",
            "Move(B,C)",
            "Drop(Package,C)",
        ],
    }
    lines.append("Checker sanity - plans that must be rejected")
    for label, names in bad_plans.items():
        rejected = not check_plan(INITIAL_STATE, GOAL, names)
        results.append(rejected)
        lines.append(f"  {label}: {'rejected' if rejected else 'ACCEPTED (wrong)'}")
    lines += ["", f"All checks passed: {all(results)}"]

    text = "\n".join(lines)
    print(text)
    out = Path(__file__).parent / "results" / "test_results.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text + "\n")


if __name__ == "__main__":
    main()
