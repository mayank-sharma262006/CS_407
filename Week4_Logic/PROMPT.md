# Logic lab: how the LLM was used

**LLM:** Claude (Anthropic), used through Claude Code.

**What it was given:** the lab handout (`logic_lab_ex.pdf`) and a request to
complete the tasks in it. The model wrote the specifications below from the
handout, wrote the code to them, and ran it.

Each specification also works as a standalone prompt.

## Prompt 1: the planner (Task 2)

I want to implement a simple planning agent in Python.

Represent a state as a set of logical propositions written as strings, for
example `At(Robot,A)`. A proposition that is not in the set is false.

Each action contains:

- a name;
- positive preconditions;
- negative preconditions;
- positive effects;
- negative effects.

An action is applicable if every positive precondition is in the current state
and no negative precondition is. When an action is applied:

1. remove its negative effects from the state;
2. add its positive effects to the state.

Use breadth-first search to find a sequence of actions that achieves a
specified goal. The goal is a set of propositions that must all be in the
state.

The problem is a warehouse with locations A, B and C, connected A-B and B-C in
both directions.

- Initial state: `At(Robot,A)`, `At(Package,A)`.
- Goal: `At(Package,C)`.
- `Move(X,Y)` for each connected pair: needs `At(Robot,X)`; removes it; adds
  `At(Robot,Y)`.
- `PickUp(Package,X)` for each location: needs `At(Robot,X)` and
  `At(Package,X)`; removes `At(Package,X)`; adds `Holding(Package)`.
- `Drop(Package,X)` for each location: needs `At(Robot,X)` and
  `Holding(Package)`; removes `Holding(Package)`; adds `At(Package,X)`.

The program should also:

- detect when no plan exists and print `No plan found`;
- print the resulting sequence of actions;
- print the state reached after each action.

Make it possible to build the action list without `PickUp`, or with `Move`
only, so the same planner can be run on modified problems. Explain the
implementation and state any assumptions.

## Prompt 2: tests (Task 3)

Write a test script for tests A, B and C of the lab: the original problem, the
problem with `PickUp` removed, and a problem where only `Move` is available.
Validate any plan with a checker that does not reuse the planner's action
objects or its apply function: write the action rules out again by hand.
Confirm that the checker rejects plans that are known to be invalid. Save the
results to a text file.

## Prompt 3: plan explanation (Task 5, optional)

For every action in the plan, identify its preconditions and show that those
preconditions are satisfied in the state in which the action is executed.
(The answer is recorded in `REPORT.md`, Task 5, next to the states computed by
the program.)
