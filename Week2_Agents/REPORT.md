# Agents lab: goal-based warehouse agent

| File | Contents |
|---|---|
| `agents_lab.pdf` | Lab handout |
| `warehouse_agent.py` | The agent |
| `PROMPT.md` | How the LLM was used and the specification it worked to |
| `results/output.txt` | Output of `python warehouse_agent.py` |

Run with `python warehouse_agent.py` (Python 3, standard library only).

## Task 1: understanding the problem

**1. Environment.** A 7 x 21 grid of cells. Each cell is a shelf (`#`), free
space (`.`), the start `S` or the goal `G`. The map does not change while the
agent acts, the agent knows all of it in advance, and a move always has the
intended result.

**2. Goal.** Be in the cell marked `G`, having got there from `S` without
entering a shelf cell.

**3. Actions.** Up, Down, Left, Right. Each moves the vehicle one cell. A move
into a shelf or off the map is not allowed.

**4. Information the agent must maintain.**

- its current position, as `(row, column)`;
- the map, to know which moves are legal;
- the position of the goal;
- the cells already discovered, so the search never goes round in circles;
- for each discovered cell, the cell and action that led to it, so the route
  can be read back once the goal is found.

**5. Why goal-based and not simple reflex.** A simple reflex agent picks an
action from the current percept alone, using fixed rules such as "if the cell
to the right is free, move right". On this map that stops at (1,5), where the
shelf at (1,6) blocks the way. The route continues by dropping into row 2 and
coming back up at column 7, and nothing in the percept at (1,5) says that this
detour leads back to the goal while others lead to dead ends. The agent here
holds an explicit goal and a model of what each action
does, and it chooses actions by looking ahead for a sequence that ends at the
goal.

**Think About It: a warehouse twice as large.** BFS would still be correct,
and still return a shortest path. The cost changes: doubling each side gives
four times as many cells, and BFS may store and visit nearly all of them, so
time and memory grow with the area. Longer routes also mean more of the map is
explored before the goal turns up. At some size an informed search such as A*
becomes worth it, because a heuristic can steer the search toward the goal.
Other difficulties are outside the search itself: a large real warehouse is
unlikely to stay static or be fully known, which breaks the assumption that a
plan made once stays valid.

## Task 2: agent design

| Component | Design |
|---|---|
| Environment | `Warehouse`: the grid as a list of strings, plus `move(cell, action)`, which returns the new cell or `None` if blocked |
| Current state | `(row, column)` of the vehicle |
| Goal | the cell containing `G`; goal test is `cell == goal` |
| Actions | Up `(-1,0)`, Down `(+1,0)`, Left `(0,-1)`, Right `(0,+1)` |
| Decision-making | `GoalBasedAgent.plan()`: breadth-first search from `S` |

```
                 +------------------------------+
                 |  Environment (Warehouse)     |
                 |  grid, S, G, move()          |
                 +------------------------------+
                      |                   ^
       start, goal,   |                   |  "what happens if I take
       legal moves    v                   |   this action here?"
  +---------------------------------------------------+
  |  Goal-based agent                                 |
  |                                                   |
  |   state (row, col) ---> goal test: state == G ?   |
  |          |                      | no              |
  |          |                      v                 |
  |          |        try Up / Down / Left / Right    |
  |          |                      |                 |
  |          |                      v                 |
  |          +<---- BFS frontier (FIFO) + visited     |
  |                                                   |
  +---------------------------------------------------+
                      |
                      v
        list of actions  or  "no path exists"
```

## Task 3: prompt and generated program

The specification used is in `PROMPT.md`. The program generated from it is
`warehouse_agent.py`.

### Result on the lab map

```
Start (1, 1), goal (1, 19)
Cells discovered by BFS: 63
Path found with 20 moves (replay check passed).
Actions: Right Right Right Down Right Right Right Up Right Right Right Right
         Right Right Right Right Right Right Right Right

#####################
#S***.#************G#
#.##****##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
```

### Validation

- **Replay check.** `check_route()` re-executes the returned actions from `S`
  using only the map. It raises an error if a step enters a shelf or the route
  does not end on `G`. It shares no code with the search. It passed.
- **Shortest-path check by hand.** `S` and `G` are 18 columns apart in the
  same row, so 18 moves is a lower bound. Row 1 is blocked at column 6, so the
  vehicle must leave row 1 and return, which costs at least 2 extra moves. 20
  is therefore the minimum, and the agent found 20.
- **No-path case.** On `BLOCKED_MAP` (the lab map with cells (2,6) and (5,8)
  turned into shelves) the program prints "No collision-free path exists from
  S to G." after discovering 20 cells, and terminates.

### Questions

**1. Did the LLM generate a working program on the first attempt?**
Yes. The first execution produced the 20-move route above and the replay check
passed. One change was made afterwards, and it was not to the logic: a comment
describing which cells `BLOCKED_MAP` closes off was wrong and was corrected.

**2. If not, how can you improve your prompt?**
Not needed here, but the parts of the specification that did the most work
were the ones that pinned down what would otherwise be guessed: the exact map,
the four moves, "fewest moves", what to print, and what to do when no path
exists. Asking for an independent replay check in the prompt meant the program
arrived with its own test.

**3. What search algorithm did the LLM choose?**
Breadth-first search.

**4. Why was this algorithm selected?**
Every move has the same cost. BFS explores cells in order of distance from
`S`, so the first time it reaches `G` it has used the fewest moves possible.
It is also complete on a finite grid: each cell enters the frontier at most
once, so the search ends either at `G` or when no cells are left, which is how
the no-path case is detected. The map is small (BFS had discovered 63
cells when it reached `G`), so the memory cost of BFS does not matter. Depth-first search would also find a path
but with no guarantee that it is short.

## LLM use and what was verified

The specification, the code and this report were produced with Claude (Claude
Code) working from the lab handout. What was checked independently of the
model's say-so: the program was executed, the route was replayed by separate
code, the length was checked against a hand-derived lower bound, and the
no-path behaviour was tested on a map built for that purpose.
