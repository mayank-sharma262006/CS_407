"""Look for a map where an inadmissible heuristic costs A* its optimality.

On the lab warehouse every heuristic returns the same 40-move path, so the
effect of overestimating cannot be seen there. This script generates random
small maps and keeps the one where A* with 2 x Manhattan is furthest from the
true shortest path (given by BFS). The map it prints is stored in
gridworld.py as OVERESTIMATE_TRAP.

Run:  python find_counterexample.py
"""

import random

from astar import astar, double_manhattan, manhattan
from bfs import bfs
from gridworld import GridProblem

SEED = 407
TRIALS = 200_000
SIZES = [(5, 7), (5, 8), (6, 8), (6, 9)]  # interior rows x columns
SHELF_PROBABILITY = 0.28


def random_map(rng):
    height, width = rng.choice(SIZES)
    cells = [
        ["#" if rng.random() < SHELF_PROBABILITY else "." for _ in range(width)]
        for _ in range(height)
    ]
    free = [(r, c) for r in range(height) for c in range(width) if cells[r][c] == "."]
    if len(free) < 6:
        return None
    (sr, sc), (gr, gc) = rng.sample(free, 2)
    cells[sr][sc] = "S"
    cells[gr][gc] = "G"
    border = "#" * (width + 2)
    return "\n".join([border] + ["#" + "".join(row) + "#" for row in cells] + [border])


def main():
    rng = random.Random(SEED)
    best_gap, best = 0, None
    for _ in range(TRIALS):
        text = random_map(rng)
        if text is None:
            continue
        problem = GridProblem(text)
        shortest = bfs(problem)
        if not shortest.found:
            continue
        # Manhattan is admissible, so it must always match BFS.
        assert astar(problem, manhattan).length == shortest.length
        gap = astar(problem, double_manhattan).length - shortest.length
        if gap > best_gap:
            best_gap, best = gap, (text, shortest.length)

    text, shortest = best
    print(text)
    print(f"Shortest path: {shortest} moves")
    print(f"A* with 2 x Manhattan: {shortest + best_gap} moves")


if __name__ == "__main__":
    main()
