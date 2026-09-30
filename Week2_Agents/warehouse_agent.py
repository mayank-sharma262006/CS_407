"""Goal-based agent for the warehouse navigation problem (Agents lab).

The program is split the same way as the design in REPORT.md:

    Warehouse       -> the environment (grid, obstacles, legal moves)
    GoalBasedAgent  -> the decision-making component (breadth-first search)
    check_route     -> an independent replay of the plan, used for validation

Search algorithm: breadth-first search (BFS).
Every move costs the same (one grid square), so the first time BFS reaches
the goal it has done so with the fewest possible moves. BFS also terminates
on a finite grid because each cell is put on the frontier at most once,
which means an unreachable goal is reported instead of looping forever.

Run:  python warehouse_agent.py
"""

from collections import deque

WAREHOUSE_MAP = [
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################",
]

# Same layout with cells (2,6) and (5,8) turned into shelves. Those are the
# only two ways out of the left-hand section, so G cannot be reached.
BLOCKED_MAP = [
    "#####################",
    "#S....#............G#",
    "#.##..#.##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#.......##..........#",
    "#####################",
]

# action name -> (row change, column change)
ACTIONS = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}


class Warehouse:
    """The environment: a fixed grid that answers 'where does this move lead?'."""

    def __init__(self, rows):
        if len({len(row) for row in rows}) != 1:
            raise ValueError("every row of the map must have the same width")
        self.rows = rows
        self.start = self._locate("S")
        self.goal = self._locate("G")

    def _locate(self, symbol):
        hits = [
            (r, c)
            for r, row in enumerate(self.rows)
            for c, cell in enumerate(row)
            if cell == symbol
        ]
        if len(hits) != 1:
            raise ValueError(f"the map must contain exactly one '{symbol}'")
        return hits[0]

    def is_free(self, cell):
        r, c = cell
        inside = 0 <= r < len(self.rows) and 0 <= c < len(self.rows[0])
        return inside and self.rows[r][c] != "#"

    def move(self, cell, action):
        """Cell reached by taking `action` from `cell`, or None if blocked."""
        dr, dc = ACTIONS[action]
        target = (cell[0] + dr, cell[1] + dc)
        return target if self.is_free(target) else None


class GoalBasedAgent:
    """Knows its position and its goal, and searches for actions linking them."""

    def __init__(self, warehouse):
        self.warehouse = warehouse
        self.cells_discovered = 0

    def plan(self):
        """Return a shortest list of actions from S to G, or None if none exists."""
        env = self.warehouse
        frontier = deque([env.start])
        # reached_by[cell] = (previous cell, action taken); doubles as visited set
        reached_by = {env.start: None}

        while frontier:
            cell = frontier.popleft()
            if cell == env.goal:
                break
            for action in ACTIONS:
                target = env.move(cell, action)
                if target is not None and target not in reached_by:
                    reached_by[target] = (cell, action)
                    frontier.append(target)

        self.cells_discovered = len(reached_by)
        if env.goal not in reached_by:
            return None

        actions = []
        cell = env.goal
        while reached_by[cell] is not None:
            cell, action = reached_by[cell]
            actions.append(action)
        actions.reverse()
        return actions


def check_route(warehouse, actions):
    """Replay the actions from S. Returns the cells visited.

    Raises if any step enters an obstacle or the route does not end on G.
    This does not reuse the search code, so it is an independent check.
    """
    cell = warehouse.start
    visited = [cell]
    for step, action in enumerate(actions, start=1):
        cell = warehouse.move(cell, action)
        if cell is None:
            raise AssertionError(f"step {step} ({action}) hits an obstacle")
        visited.append(cell)
    if cell != warehouse.goal:
        raise AssertionError("route does not finish on G")
    return visited


def draw(warehouse, visited):
    """The map with the route marked by '*'."""
    canvas = [list(row) for row in warehouse.rows]
    for r, c in visited[1:-1]:
        canvas[r][c] = "*"
    return "\n".join("".join(row) for row in canvas)


def run(title, rows):
    print(f"=== {title} ===")
    warehouse = Warehouse(rows)
    agent = GoalBasedAgent(warehouse)
    actions = agent.plan()

    print(f"Start {warehouse.start}, goal {warehouse.goal}")
    print(f"Cells discovered by BFS: {agent.cells_discovered}")
    if actions is None:
        print("No collision-free path exists from S to G.")
        print()
        return

    visited = check_route(warehouse, actions)
    print(f"Path found with {len(actions)} moves (replay check passed).")
    print("Actions: " + " ".join(actions))
    print("Cells:   " + " ".join(f"({r},{c})" for r, c in visited))
    print(draw(warehouse, visited))
    print()


if __name__ == "__main__":
    run("Lab warehouse", WAREHOUSE_MAP)
    run("Blocked warehouse (goal unreachable)", BLOCKED_MAP)
