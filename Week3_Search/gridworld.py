"""Search-problem formulation shared by astar.py and bfs.py.

    State       (row, col) of the robot
    Actions     Up, Down, Left, Right
    Transition  move one cell if the target is inside the map and not '#'
    Initial     the cell marked S
    Goal        the cell marked G
    Cost        1 per move
"""

from dataclasses import dataclass, field

MOVES = (("Up", -1, 0), ("Down", 1, 0), ("Left", 0, -1), ("Right", 0, 1))

# Map supplied in the lab handout.
WAREHOUSE = """
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################
"""

# Test 2: goal directly next to the start (from the handout).
ONE_STEP = """
#####
#SG##
#####
"""

# Test 3: goal sealed off (from the handout).
NO_SOLUTION = """
#######
#S....#
###.###
#...#G#
#######
"""

# Test 4: two routes, 10 moves down-then-right and 14 moves through the top.
TWO_ROUTES = """
#########
#S..#...#
#.#.#.#.#
#.#...#.#
#.#####.#
#......G#
#########
"""

# Extra map for the heuristic investigation: an open floor with one shelf.
# There is no corridor here, so the heuristic has real choices to make.
OPEN_FLOOR = """
###############
#.............#
#.............#
#......#......#
#..S...#...G..#
#......#......#
#.............#
#.............#
###############
"""

# Extra map for the heuristic investigation: found by find_counterexample.py,
# which tries random small maps until 2 x Manhattan returns a longer path
# than BFS. Shortest path here is 12 moves.
OVERESTIMATE_TRAP = """
###########
#..#.....S#
##.#...#..#
#....#.#..#
##.#....#.#
#..G#...#.#
#.#..#....#
###########
"""


class GridProblem:
    """A grid map turned into (S, A, T, s0, G, c)."""

    def __init__(self, text):
        self.rows = text.strip().splitlines()
        self.start = self._find("S")
        self.goal = self._find("G")

    def _find(self, symbol):
        for r, row in enumerate(self.rows):
            c = row.find(symbol)
            if c >= 0:
                return (r, c)
        raise ValueError(f"map has no '{symbol}'")

    def is_goal(self, state):
        return state == self.goal

    def successors(self, state):
        """Yield (action, next_state) for every valid move. Each costs 1."""
        r, c = state
        for name, dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(self.rows) and 0 <= nc < len(self.rows[nr]):
                if self.rows[nr][nc] != "#":
                    yield name, (nr, nc)


@dataclass
class SearchResult:
    found: bool
    expanded: int
    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)

    @property
    def length(self):
        """Number of moves, or None when there is no solution."""
        return len(self.actions) if self.found else None


def rebuild(parents, goal):
    """Walk the parent links back from the goal. Returns (states, actions)."""
    states, actions = [goal], []
    while parents[states[-1]] is not None:
        previous, action = parents[states[-1]]
        states.append(previous)
        actions.append(action)
    states.reverse()
    actions.reverse()
    return states, actions


def is_valid_path(problem, result):
    """Replay a result against the map: legal moves only, S to G."""
    if not result.found:
        return False
    state = problem.start
    for action in result.actions:
        options = dict(problem.successors(state))
        if action not in options:
            return False
        state = options[action]
    return problem.is_goal(state)
