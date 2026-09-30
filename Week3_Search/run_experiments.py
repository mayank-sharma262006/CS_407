"""Tasks 5 and 6: BFS vs A*, and the heuristic investigation.

Writes results/bfs_vs_astar.csv and results/heuristics.csv.

Run:  python run_experiments.py
"""

import csv
from pathlib import Path

from astar import HEURISTICS, astar
from bfs import bfs
from gridworld import (
    OPEN_FLOOR,
    OVERESTIMATE_TRAP,
    TWO_ROUTES,
    WAREHOUSE,
    GridProblem,
)

MAPS = {
    "Lab warehouse": WAREHOUSE,
    "Two routes": TWO_ROUTES,
    "Open floor": OPEN_FLOOR,
    "Overestimate trap": OVERESTIMATE_TRAP,
}
RESULTS = Path(__file__).parent / "results"


def save(filename, rows):
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / filename).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def show(title, rows):
    print(title)
    header = list(rows[0])
    print("  " + " | ".join(header))
    for row in rows:
        print("  " + " | ".join(str(row[key]) for key in header))
    print()


def row(map_name, label, result):
    return {
        "Map": map_name,
        "Search": label,
        "Solution found": result.found,
        "Path length": result.length,
        "States expanded": result.expanded,
    }


def main():
    comparison, heuristics = [], []
    for map_name, text in MAPS.items():
        problem = GridProblem(text)
        comparison.append(row(map_name, "BFS", bfs(problem)))
        comparison.append(row(map_name, "A* (Manhattan)", astar(problem)))
        for label, h in HEURISTICS.items():
            heuristics.append(row(map_name, f"A* with {label}", astar(problem, h)))

    show("BFS vs A*", comparison)
    show("Heuristic investigation", heuristics)
    save("bfs_vs_astar.csv", comparison)
    save("heuristics.csv", heuristics)


if __name__ == "__main__":
    main()
