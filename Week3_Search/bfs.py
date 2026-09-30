"""Breadth-first search on the same problem, for the blind-search comparison.

Run:  python bfs.py
"""

from collections import deque

from astar import report
from gridworld import WAREHOUSE, GridProblem, SearchResult, rebuild


def bfs(problem):
    """Expand states in the order they were discovered (FIFO frontier).

    `expanded` is counted the same way as in astar(): states taken off the
    frontier and processed, including the goal.
    """
    frontier = deque([problem.start])
    parents = {problem.start: None}  # also serves as the visited set
    expanded = 0

    while frontier:
        state = frontier.popleft()
        expanded += 1

        if problem.is_goal(state):
            states, actions = rebuild(parents, state)
            return SearchResult(True, expanded, states, actions)

        for action, nxt in problem.successors(state):
            if nxt not in parents:
                parents[nxt] = (state, action)
                frontier.append(nxt)

    return SearchResult(False, expanded)


if __name__ == "__main__":
    report("BFS on the lab warehouse", bfs(GridProblem(WAREHOUSE)))
