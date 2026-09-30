"""A* search for the warehouse robot.

f(n) = g(n) + h(n), with Manhattan distance as the default h.

Run:  python astar.py
"""

import heapq
import itertools
import math

from gridworld import WAREHOUSE, GridProblem, SearchResult, rebuild


def manhattan(state, goal):
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def zero(state, goal):
    return 0


def euclidean(state, goal):
    return math.hypot(state[0] - goal[0], state[1] - goal[1])


def double_manhattan(state, goal):
    return 2 * manhattan(state, goal)


HEURISTICS = {
    "h = 0": zero,
    "Manhattan": manhattan,
    "Euclidean": euclidean,
    "2 x Manhattan": double_manhattan,
}


def astar(problem, h=manhattan):
    """Expand the frontier state with the lowest f. Returns a SearchResult.

    `expanded` counts states taken off the frontier and processed, including
    the goal itself.
    """
    start, goal = problem.start, problem.goal
    order = itertools.count()  # insertion number, so the heap never compares states

    g = {start: 0}  # cheapest known cost from the start
    parents = {start: None}
    closed = set()  # states already expanded
    h_start = h(start, goal)
    # entries are (f, h, insertion number, state); ties on f go to the lower h
    frontier = [(h_start, h_start, next(order), start)]
    expanded = 0

    while frontier:
        _, _, _, state = heapq.heappop(frontier)
        if state in closed:
            continue  # an older, more expensive entry for this state
        closed.add(state)
        expanded += 1

        if problem.is_goal(state):
            states, actions = rebuild(parents, state)
            return SearchResult(True, expanded, states, actions)

        for action, nxt in problem.successors(state):
            if nxt in closed:
                continue
            g_next = g[state] + 1
            if g_next < g.get(nxt, math.inf):
                g[nxt] = g_next
                parents[nxt] = (state, action)
                h_next = h(nxt, goal)
                f_next = g_next + h_next
                heapq.heappush(frontier, (f_next, h_next, next(order), nxt))

    return SearchResult(False, expanded)


def report(name, result):
    print(name)
    print("  Solution found:", result.found)
    if result.found:
        print("  Path length:", result.length)
        print("  Actions:", " ".join(result.actions))
        print("  States:", " ".join(f"({r},{c})" for r, c in result.states))
    print("  States expanded:", result.expanded)


if __name__ == "__main__":
    report("A* (Manhattan) on the lab warehouse", astar(GridProblem(WAREHOUSE)))
