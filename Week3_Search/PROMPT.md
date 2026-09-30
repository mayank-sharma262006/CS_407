# Search lab: how the LLM was used

**LLM:** Claude (Anthropic), used through Claude Code.

**What it was given:** the lab handout (`search_lab_ex.pdf`) and a request to
complete the tasks in it. The model wrote the specifications below from the
handout, wrote the code to them, and ran it.

Each specification also works as a standalone prompt.

## Prompt 1: A* (Task 2)

I am implementing a goal-based search agent in Python for this warehouse:

```
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################
```

The robot starts at `S` and must reach `G`. `#` is an obstacle and `.` is a
free cell. It can move Up, Down, Left or Right, and every move costs 1.

Implement A* search with the Manhattan distance heuristic
`h(n) = |row - row_G| + |col - col_G|`.

Design decisions already made:

- a state is a `(row, col)` tuple;
- the map is a list of strings and lives in a problem class that exposes the
  start, the goal, a goal test, and a successor function returning
  `(action, next_state)` pairs for valid moves only;
- the frontier is a priority queue ordered by `f(n) = g(n) + h(n)`, with ties
  on `f` broken in favour of the lower `h`;
- `g` values are kept in a dictionary, and a closed set stops any state from
  being expanded twice;
- the path is rebuilt from parent links once the goal is taken off the
  frontier;
- the heuristic is a function argument, so it can be swapped later.

The program must report whether a solution was found, the actions, the states
on the path, the path length, and the number of states expanded. It must
return a failure result, not loop forever, when the goal is unreachable. Use
only the standard library and keep the code short.

## Prompt 2: BFS for comparison (Task 5)

Add a breadth-first search for the same problem class, returning the same
result type. Count expanded states the same way as in A* (each state taken off
the frontier and processed, the goal included), so the two numbers can be
compared directly. Do not change the warehouse.

## Prompt 3: tests (Task 3)

Write a test script that runs both algorithms on four maps: the lab warehouse,
a one-step map, a map where the goal is sealed off, and a map with two routes
of different length. For each, compare against an expected shortest length
that I state in the script, replay the returned actions against the map to
confirm they are legal, and write the results to a CSV file.

## Prompt 4: heuristic investigation (Task 6)

Run A* with four heuristics: `h = 0`, Manhattan, Euclidean, and
`2 x Manhattan`. Record solution found, path length and states expanded for
each, on each map, and write a CSV file.

## Prompt 5: follow-up after seeing the results

On the lab warehouse all four heuristics give the same path length and the
same number of expanded states, so the experiment shows nothing about
admissibility. Add an open map where the heuristics can differ, and write a
script that searches random small maps for one where `2 x Manhattan` returns a
path longer than the BFS path. Add the map it finds to the experiments.

## Prompt 6: heuristic explanation (Task 6)

Explain why Manhattan distance is an appropriate heuristic for this warehouse
when the robot can only move horizontally and vertically. (The answer is
recorded in `REPORT.md`, Task 6.)
