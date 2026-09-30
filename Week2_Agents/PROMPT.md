# Agents lab: how the LLM was used

**LLM:** Claude (Anthropic), used through Claude Code.

**What it was given:** the lab handout (`agents_lab.pdf`) and a request to
complete the tasks in it. The model wrote the specification below from the
handout, then wrote `warehouse_agent.py` to that specification and ran it.

The specification also works as a standalone prompt: pasting it into an LLM
should reproduce an equivalent program.

## Specification

Write a well-documented Python program implementing a goal-based agent for
this warehouse navigation problem.

```
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
```

`S` is the start, `G` is the goal, `#` is a shelf that cannot be entered and
`.` is free space. The vehicle moves Up, Down, Left or Right, one grid square
per move.

The program should:

- represent the warehouse as a two-dimensional grid, with the environment
  (map and legal moves) kept separate from the agent (the search);
- use a state that is only the vehicle's `(row, column)` position;
- find a collision-free path from `S` to `G` with the fewest moves;
- never enter a `#` cell or leave the grid;
- print the actions, the cells visited, the number of moves, and the map with
  the route drawn on it;
- print a clear message, and terminate, if no path exists;
- replay the returned actions with a function that does not share code with
  the search, and fail loudly if any step hits an obstacle or the route does
  not end on `G`;
- demonstrate the no-path case on a second map where `G` is walled off;
- use only the Python standard library;
- explain in the module docstring which search algorithm was chosen and why.
