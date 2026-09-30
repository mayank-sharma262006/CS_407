# Search lab: A* and BFS on the warehouse map

| File | Contents |
|---|---|
| `search_lab_ex.pdf` | Lab handout |
| `gridworld.py` | Problem formulation, the maps, path reconstruction, path replay check |
| `astar.py` | A* and the four heuristics |
| `bfs.py` | Breadth-first search |
| `run_tests.py` | Task 3 tests |
| `run_experiments.py` | Task 5 comparison and Task 6 heuristic investigation |
| `find_counterexample.py` | Random search for a map where `2 x Manhattan` is not optimal |
| `PROMPT.md` | How the LLM was used and the specifications it worked to |
| `results/` | Output of every script, as text and CSV |

To reproduce everything (Python 3, standard library only):

```
python astar.py
python bfs.py
python run_tests.py
python run_experiments.py
python find_counterexample.py
```

## Task 0: the search problem

| Component | Specification |
|---|---|
| State `S` | The robot's position `(row, col)`. The state space is the set of free cells: 64 on the lab map. |
| Actions `A` | Up, Down, Left, Right |
| Transition `T` | `T((r, c), a) = (r + dr, c + dc)` when that cell is inside the map and is not `#`. Otherwise the action is not available in that state. |
| Initial state `s0` | The cell marked `S`: `(1, 1)` |
| Goal `G` | The cell marked `G`: `{(7, 15)}` |
| Cost `c` | 1 per move, so path cost is the number of moves |

**(a) What is needed to specify a state?** Only the robot's position. The map
never changes, so it belongs to the problem, not the state.

**(b) What makes an action invalid?** The target cell is an obstacle or lies
outside the map.

**(c) Is the problem deterministic?** Yes. A valid action in a given state has
exactly one outcome, and nothing else changes the world.

**(d) What is a solution?** A sequence of valid actions that takes the robot
from `S` to `G`. An optimal solution is one with the fewest moves.

## Task 1: agent design

1. **State in Python:** a tuple `(row, col)`. Tuples are hashable, so states
   can be dictionary keys and set members.
2. **Warehouse:** a list of strings inside `GridProblem`; `rows[r][c]` is the
   cell.
3. **Valid actions:** `GridProblem.successors(state)` tries the four moves and
   yields only those that stay on the map and do not land on `#`.
4. **Goal test:** `GridProblem.is_goal(state)`, which is `state == goal`. It
   is applied when a state is taken off the frontier, not when it is added.
   A* needs this to be optimal.
5. **Frontier contents:** `(f, h, insertion number, state)`. `f` orders the
   queue, `h` breaks ties toward states nearer the goal, and the insertion
   number makes every entry unique so the heap never has to compare states.
6. **Path reconstruction:** a dictionary maps each state to the
   `(parent, action)` that reached it most cheaply. `rebuild()` follows it
   back from the goal and reverses the result.

Reported on termination: solution found, actions, states on the path, path
length, states expanded. "States expanded" means states taken off the frontier
and processed, with the goal counted. Both algorithms count the same way.

## Task 2: prompt

See `PROMPT.md`.

## Task 3: tests

`run_tests.py` runs both algorithms on each map. A row passes when the result
matches the expected length and, where a path is returned,
`is_valid_path()` replays it against the map without hitting an obstacle.
The expected lengths were worked out by hand before running.

| Test | Algorithm | Expected length | Found | Path length | Expanded | Pass |
|---|---|---|---|---|---|---|
| 1 Original warehouse | A* | 40 | yes | 40 | 64 | yes |
| 1 Original warehouse | BFS | 40 | yes | 40 | 64 | yes |
| 2 Trivial one-step | A* | 1 | yes | 1 | 2 | yes |
| 2 Trivial one-step | BFS | 1 | yes | 1 | 2 | yes |
| 3 No solution | A* | none | no | - | 9 | yes |
| 3 No solution | BFS | none | no | - | 9 | yes |
| 4 Alternative paths | A* | 10 | yes | 10 | 11 | yes |
| 4 Alternative paths | BFS | 10 | yes | 10 | 20 | yes |

**Test 1 path (40 moves):** Right x4, Down x4, Right x8, Up x2, Left x6,
Up x2, Right x8, Down x6. The full list of states is in
`results/astar_output.txt`.

**Test 3** has 9 cells reachable from `S`. Both algorithms expanded exactly
those 9 and then reported failure, so neither loops.

**Test 4 map.** Two routes exist: 10 moves (down the left side, then along the
bottom) and 14 moves (through the top). Both algorithms returned 10.

```
#########
#S..#...#
#.#.#.#.#
#.#...#.#
#.#####.#
#......G#
#########
```

## Task 4: where each concept is in the code

| Concept | Where |
|---|---|
| State | `(row, col)` tuples produced by `GridProblem` in `gridworld.py` |
| Action | the names in `MOVES` in `gridworld.py` |
| Transition | `GridProblem.successors()` |
| Goal test | `problem.is_goal(state)` in `astar()`, right after the pop |
| `g(n)` | the dictionary `g`; `g_next = g[state] + 1` |
| `h(n)` | `h(nxt, goal)`, by default `manhattan()` |
| `f(n)` | `f_next = g_next + h_next` |
| Frontier | the list `frontier`, used as a heap through `heapq` |
| Visited states | the set `closed` |
| Path reconstruction | `parents` dictionary and `rebuild()` in `gridworld.py` |

**(a) Data structure for the frontier:** a binary min-heap (`heapq`) of
`(f, h, insertion number, state)` tuples.

**(b) How the next state is chosen:** `heapq.heappop` returns the entry with
the smallest `f`; ties go to the smaller `h`, then to the earlier insertion.

**(c) Where the heuristic is calculated:** once for the start state, and then
for each successor as it is pushed (`h_next = h(nxt, goal)`).

**(d) Is `f = g + h` computed explicitly?** Yes: `f_next = g_next + h_next`.

**(e) How repeated exploration is prevented:** a state is pushed again only if
a strictly cheaper `g` has been found for it. When a state is popped it goes
into `closed`; later heap entries for the same state are discarded on pop, and
successors already in `closed` are skipped.

## Task 5: BFS and A* compared

On the lab warehouse:

| Measure | BFS | A* (Manhattan) |
|---|---|---|
| Solution found | yes | yes |
| Path length | 40 | 40 |
| States expanded | 64 | 64 |

**(a) Did both find a solution?** Yes.

**(b) Same path length?** Yes, 40 moves.

**(c) Which expanded fewer states?** Neither. Both expanded 64, which is every
free cell on the map.

**(d) Why might A\* expand fewer states?** A* orders the frontier by
`g + h`, so states that the heuristic says are far from the goal wait, and
many are never expanded. BFS orders by `g` alone and expands everything closer
to the start than the goal is.

On this particular map the heuristic has nothing to work with. The map is one
winding route with dead-end branches. The route is 40 moves long and
repeatedly heads away from the goal, while the Manhattan distance from the
start is only 20. A* expands every state whose `f` is below the optimal cost
of 40, and may expand those equal to it; on this map that turned out to be
every cell.
The same comparison on maps with real choices shows the expected gap:

| Map | Path length | BFS expanded | A* expanded |
|---|---|---|---|
| Lab warehouse | 40 | 64 | 64 |
| Two routes (Test 4) | 10 | 20 | 11 |
| Open floor | 12 | 77 | 21 |
| Overestimate trap | 12 | 38 | 27 |

## Task 6: heuristic investigation

**Why Manhattan distance suits this warehouse (LLM explanation, checked
against the results below).** With only horizontal and vertical unit moves,
the robot needs at least `|row - row_G|` vertical moves and `|col - col_G|`
horizontal moves even if there were no obstacles. Obstacles can only add
moves. So Manhattan distance never overestimates the true remaining cost: it
is admissible. It is also consistent, because one move changes it by exactly
1, which equals the step cost. With a consistent heuristic, A* with a closed
set returns an optimal path. Among admissible heuristics for this movement
rule it is also the most informed simple choice, since it is exactly the cost
on an empty grid.

Results for all four heuristics (`results/heuristics.csv`):

| Map | Heuristic | Found | Path length | Expanded |
|---|---|---|---|---|
| Lab warehouse | `h = 0` | yes | 40 | 64 |
| Lab warehouse | Manhattan | yes | 40 | 64 |
| Lab warehouse | Euclidean | yes | 40 | 64 |
| Lab warehouse | 2 x Manhattan | yes | 40 | 64 |
| Two routes | `h = 0` | yes | 10 | 20 |
| Two routes | Manhattan | yes | 10 | 11 |
| Two routes | Euclidean | yes | 10 | 17 |
| Two routes | 2 x Manhattan | yes | 10 | 11 |
| Open floor | `h = 0` | yes | 12 | 77 |
| Open floor | Manhattan | yes | 12 | 21 |
| Open floor | Euclidean | yes | 12 | 47 |
| Open floor | 2 x Manhattan | yes | 12 | 18 |
| Overestimate trap | `h = 0` | yes | 12 | 38 |
| Overestimate trap | Manhattan | yes | 12 | 27 |
| Overestimate trap | Euclidean | yes | 12 | 32 |
| Overestimate trap | 2 x Manhattan | yes | **18** | 33 |

**1. `h(n) = 0`.** A* becomes uniform-cost search, which with unit costs
behaves like BFS. The path stays optimal and the number of expanded states
equals the BFS number on every map (64, 20, 77, 38).

**2. Euclidean distance.** Still admissible, because a straight line is never
longer than a path made of horizontal and vertical steps. The path stays
optimal on every map. It is a weaker estimate than Manhattan distance, so A*
expands more states with it (17 against 11, 47 against 21, 32 against 27),
though fewer than with `h = 0`.

**3. `2 x Manhattan`.** No longer admissible: next to the goal it says 2 when
the true cost is 1. On three of the maps the path is still optimal, and on the
open floor it expands slightly fewer states than Manhattan (18 against 21).
On the trap map it returns an 18-move path when the shortest is 12, and it
expands more states than plain Manhattan while doing so (33 against 27).

**Think About It: too optimistic or too aggressive.** A heuristic that is too
optimistic (`h = 0` is the extreme) keeps A* correct but throws away guidance,
and the search widens toward BFS. A heuristic that is too aggressive
overstates the remaining cost, so the search commits to states that look
close to the goal and may take a state off the frontier through a worse route
before the better one is found. Optimality is then no longer guaranteed. The
trap map shows that it is not even guaranteed to be faster.

The trap map was found by `find_counterexample.py`, which generates random
small maps (seed 407) and keeps the one with the largest gap between the
`2 x Manhattan` path and the BFS path.

```
###########
#..#.....S#
##.#...#..#
#....#.#..#
##.#....#.#
#..G#...#.#
#.#..#....#
###########
```

## Task 7: evaluating the LLM-generated agent

**1. What was correct immediately?** All of the search code. On the first
execution `astar.py`, `bfs.py` and all eight rows of `run_tests.py` gave the
expected results.

**2. Bugs or design problems?** No bugs were found by the tests. There was one
design problem in the experiment, not the code: the lab warehouse cannot
distinguish BFS from A* or one heuristic from another, because every search
expands all 64 cells.

**3. How was it discovered?** From the first comparison table: the same number
on every row. Counting the free cells on the map gave 64, which explained it.

**4. Unfamiliar terminology or data structures?** The parts of the generated
code that are not obvious from the lecture description of A* are: the
insertion counter in each heap entry (without it, two entries with equal `f`
and `h` would be ordered by comparing states); discarding stale heap entries
on pop instead of updating them in place; and the difference between an
admissible heuristic and a consistent one, which matters once a closed set is
used.

**5. Was the generated code modified?** The search functions were not. After
the first run, the open-floor and trap maps, `find_counterexample.py`, and the
extra rows in `run_experiments.py` were added (Prompt 5 in `PROMPT.md`).

**6. Most useful tests.** The no-solution map, because it is the only one that
tests termination. The alternative-paths map, because the right answer (10,
not 14) is known in advance. The trap map, because it is the only one that
separates an admissible heuristic from an inadmissible one.

**7. Could the program have been trusted without testing?** No. The results
give a concrete reason: a version of the program using `2 x Manhattan` passes
all four tests the handout asks for, with optimal paths on each, and is still
not an optimal search. Only a map built to expose it shows the fault.

**8. What implementing A\* makes clear.** That A* is not automatically cheaper
than BFS: the saving comes entirely from the heuristic, and on a map where the
heuristic is uninformative (the lab warehouse) there is none. And that
"admissible" is a property with a visible consequence, not a formality: the
same code with the heuristic doubled returns a path 50% longer on the trap
map.

### What was designed, suggested, accepted, changed and tested

The formulation, code and this report were produced with Claude (Claude Code)
working from the lab handout. The distinction the handout asks for maps onto
that work as follows.

1. **Fixed before any code was written:** the problem formulation in Task 0
   and the design in Task 1, both taken from the handout's requirements.
2. **Choices the LLM made beyond the handout:** tie-breaking on `h`, the
   insertion counter, the separate `GridProblem` class, and the random search
   for a counter-example.
3. **Accepted unchanged:** the search code as first generated.
4. **Changed after seeing results:** the set of experiment maps.
5. **Tested by execution:** the four handout tests on both algorithms, replay
   of every returned path, and the heuristic table on four maps.

## Final reflection

**1. Why formulate the problem before writing the algorithm?** The algorithm
only manipulates states, actions and costs, so those have to be fixed first.
Here the formulation settled that a state is just a position (the map is not
part of it), that the goal test is an equality, and that every step costs 1.
Each of those decisions shows up directly in the code, and the step cost is
what makes Manhattan distance admissible. A search written before these were
decided would have had nothing to be checked against.

**2. In what sense is A\* informed?** It uses knowledge about the problem that
is not in the search tree built so far: an estimate `h(n)` of the cost still
to go. BFS knows only how far each state is from the start. A* combines both,
and expands the state whose estimated total cost through it is lowest.

**3. Why does the choice of heuristic matter?** It decides both whether the
answer is optimal and how much work is done. In these experiments an
admissible but weak heuristic (Euclidean) kept the path optimal and expanded
more than twice as many states as Manhattan on the open floor (47 against
21); no heuristic at all matched BFS; and
an inadmissible one returned a path of 18 where 12 was possible.

**4. What did the LLM contribute?** Speed on the mechanical parts: the heap
handling, path reconstruction, CSV output and the test harness. It also
supplied the explanation of why Manhattan distance is admissible and
consistent, which the experiments then agreed with.

**5. What could go wrong if LLM code were accepted untested?** The code could
be plausible and wrong in a way that ordinary use does not reveal. Examples
for this problem: testing for the goal when a state is pushed instead of
popped, a heuristic that overestimates, or a search that never terminates when
the goal is unreachable. Each of these returns a sensible-looking path on the
lab map. They are caught only by tests with known answers.
