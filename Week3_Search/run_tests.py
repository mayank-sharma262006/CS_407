"""Task 3: run A* and BFS on the four test maps and check known answers.

Writes results/test_results.csv.

Run:  python run_tests.py
"""

import csv
from pathlib import Path

from astar import astar
from bfs import bfs
from gridworld import (
    NO_SOLUTION,
    ONE_STEP,
    TWO_ROUTES,
    WAREHOUSE,
    GridProblem,
    is_valid_path,
)

# (name, map, expected shortest path length; None means "no solution")
# The expected lengths were worked out by hand from the maps.
TESTS = [
    ("1 Original warehouse", WAREHOUSE, 40),
    ("2 Trivial one-step", ONE_STEP, 1),
    ("3 No solution", NO_SOLUTION, None),
    ("4 Alternative paths", TWO_ROUTES, 10),
]


def check(problem, result, expected):
    if expected is None:
        return not result.found
    return is_valid_path(problem, result) and result.length == expected


def main():
    rows = []
    all_passed = True
    for name, text, expected in TESTS:
        problem = GridProblem(text)
        for algorithm, search in (("A*", astar), ("BFS", bfs)):
            result = search(problem)
            passed = check(problem, result, expected)
            all_passed = all_passed and passed
            rows.append(
                {
                    "Test": name,
                    "Algorithm": algorithm,
                    "Expected length": "none" if expected is None else expected,
                    "Solution found": result.found,
                    "Path length": "" if result.length is None else result.length,
                    "States expanded": result.expanded,
                    "Pass": passed,
                }
            )

    header = list(rows[0])
    print(" | ".join(header))
    for row in rows:
        print(" | ".join(str(row[key]) for key in header))
    print("\nAll tests passed:", all_passed)

    out = Path(__file__).parent / "results" / "test_results.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
