# Logic lab: logical reasoning for planning

| File | Contents |
|---|---|
| `logic_lab_ex.pdf` | Lab handout |
| `planner.py` | The planning agent (breadth-first search over logical states) |
| `run_tests.py` | Tests A, B, C and an independent plan checker |
| `planner.pl` | Optional extension: Prolog facts and rules |
| `PROMPT.md` | How the LLM was used and the specifications it worked to |
| `results/` | Output of `planner.py` and `run_tests.py` |

Run with `python planner.py` and `python run_tests.py` (Python 3, standard
library only).

## Task 0: the planning problem

**(a) Initial state.** `I = { At(Robot,A), At(Package,A) }`

**(b) Goal.** `G = { At(Package,C) }`

**(c) Actions.** Ten ground actions:

- `Move(A,B)`, `Move(B,A)`, `Move(B,C)`, `Move(C,B)`
- `PickUp(Package,A)`, `PickUp(Package,B)`, `PickUp(Package,C)`
- `Drop(Package,A)`, `Drop(Package,B)`, `Drop(Package,C)`

**(d) Preconditions and effects.**

| Action | Preconditions | Effects |
|---|---|---|
| `Move(X,Y)`, X and Y connected | `At(Robot,X)` | not `At(Robot,X)`; `At(Robot,Y)` |
| `PickUp(Package,X)` | `At(Robot,X)`, `At(Package,X)` | not `At(Package,X)`; `Holding(Package)` |
| `Drop(Package,X)` | `At(Robot,X)`, `Holding(Package)` | not `Holding(Package)`; `At(Package,X)` |

**Initially applicable actions.** `Move(A,B)` and `PickUp(Package,A)`. No
other action has all its preconditions in `I`.

**Question.** `PickUp(Package,A)` is applicable in `I`: its preconditions are
`At(Robot,A)` and `At(Package,A)`, and both are in `I`. `Drop(Package,C)` is
not: it needs `At(Robot,C)` and `Holding(Package)`, and neither is in `I`.

**Think About It.** Being in the list of actions says only that the action
exists. Whether it can be used now is a separate question, answered by
checking each precondition against the current state. That check,
`S |= Preconditions(a)`, is the logical reasoning step.

## Task 1: plan constructed by hand

The sequence the handout suggests reasoning about starts `Move(A,B)`,
`PickUp(Package,B)`. Checking preconditions shows it does not work: after
`Move(A,B)` the package is still at A, so `At(Package,B)` is false and
`PickUp(Package,B)` is not applicable. The package has to be picked up first.

Plan: `PickUp(Package,A)`, `Move(A,B)`, `Move(B,C)`, `Drop(Package,C)`.

| State | After | Facts |
|---|---|---|
| S0 | (initial) | `At(Robot,A)`, `At(Package,A)` |
| S1 | `PickUp(Package,A)` | `At(Robot,A)`, `Holding(Package)` |
| S2 | `Move(A,B)` | `At(Robot,B)`, `Holding(Package)` |
| S3 | `Move(B,C)` | `At(Robot,C)`, `Holding(Package)` |
| S4 | `Drop(Package,C)` | `At(Robot,C)`, `At(Package,C)` |

S4 contains `At(Package,C)`, so `S4 |= G`.

## Task 2: prompt and generated planner

The specification used is in `PROMPT.md`. The generated program is
`planner.py`. Its output on the warehouse problem:

```
Initial state: {At(Package,A), At(Robot,A)}
Goal: {At(Package,C)}
Plan: PickUp(Package,A) -> Move(A,B) -> Move(B,C) -> Drop(Package,C)
S0: {At(Package,A), At(Robot,A)}
S1: {At(Robot,A), Holding(Package)}   after PickUp(Package,A)
S2: {At(Robot,B), Holding(Package)}   after Move(A,B)
S3: {At(Robot,C), Holding(Package)}   after Move(B,C)
S4: {At(Package,C), At(Robot,C)}   after Drop(Package,C)
Goal satisfied in final state: True
```

The plan and every intermediate state match the hand-constructed plan.

**Think About It: where the specification appears in the program.**

| Idea | In `planner.py` |
|---|---|
| Preconditions: when is an action applicable? | `is_applicable()`: positive preconditions are a subset of the state, and no negative precondition is in it |
| Effects: how does the state change? | `apply()`: the state minus `removes`, then plus `adds` |
| Goal: when does planning terminate? | `satisfies()`, called on each state as `find_plan()` takes it off the queue; also when the queue empties (no plan) |
| BFS: how are alternatives explored? | the FIFO `queue` in `find_plan()`, with `reached_by` recording visited states |

**Assumptions made by the implementation.**

- Closed world: a proposition not in the state is false.
- `PickUp` has the negative precondition `Holding(Package)`. The handout does
  not list it. It is redundant in this problem (a held package is not `At`
  any location, so `PickUp` already fails) and is there to exercise negative
  preconditions.
- There is one robot and one package, and all ground actions are listed in
  advance.
- BFS returns a plan with the fewest actions.

## Task 3: tests

Plans are validated by `check_plan()` in `run_tests.py`. It applies the
handout's rules from a separate, hand-written table and does not use the
planner's `Action` objects or `apply()`.

| Test | Setup | Plan found | Plan | Valid | Expected | Result |
|---|---|---|---|---|---|---|
| A Solvable | original problem | yes | `PickUp(Package,A)`, `Move(A,B)`, `Move(B,C)`, `Drop(Package,C)` | yes | a valid plan | pass |
| B Impossible | all `PickUp` actions removed | No plan found | - | n/a | no plan | pass |
| C Irrelevant actions | only `Move` actions available | No plan found | - | n/a | no plan | pass |

In every test the initial state is `{At(Package,A), At(Robot,A)}` and the goal
is `{At(Package,C)}`.

**Test C, second check.** The state `{At(Package,A), At(Robot,C)}` is
reachable using `Move` alone. `satisfies()` returns False for it, so the
planner does not treat the robot reaching C as the package reaching C.

**Checker sanity.** The checker rejects each of these invalid plans:

- `Move(A,B)`, `Move(B,C)` (the robot arrives, the package does not);
- `Move(A,C)`, `Drop(Package,C)` (A and C are not connected);
- `Move(A,B)`, `PickUp(Package,B)`, `Move(B,C)`, `Drop(Package,C)` (the
  package is not at B).

Full output is in `results/test_results.txt`. All checks passed.

## Task 4: logic and search

Completed description:

```
Current state
      |
Check action preconditions
      |
Keep the actions that are applicable
      |
Generate successor state
      |
Search over alternatives
      |
Goal?
```

**How they work together.** Logical reasoning is used in three places: to
decide whether an action may be taken in a state (`S |= Preconditions(a)`), to
compute the state it leads to (`S' = Apply(S, a)`), and to decide whether a
state meets the goal (`S |= G`). None of these says which action to take.
Search does that. From each state there may be several applicable actions, and
most lead nowhere useful, so BFS keeps a queue of states still to explore,
expands them in order of distance from the initial state, remembers which it
has seen, and stops at the first one that meets the goal. Logic fixes the
edges of the graph; search walks it.

**Think About It.** "Logic determines what is possible; search determines what
to try" describes `find_plan()` line by line: the `is_applicable()` test
filters the actions, and the queue decides the order in which the survivors
are followed up.

## Task 5 (optional): can the LLM verify its own plan?

LLM explanation of why the plan is valid, next to the states computed by
`planner.py`:

| Step | Action | Preconditions | State before (from program) | Satisfied |
|---|---|---|---|---|
| 1 | `PickUp(Package,A)` | `At(Robot,A)`, `At(Package,A)` | S0 `{At(Package,A), At(Robot,A)}` | yes |
| 2 | `Move(A,B)` | `At(Robot,A)` | S1 `{At(Robot,A), Holding(Package)}` | yes |
| 3 | `Move(B,C)` | `At(Robot,B)` | S2 `{At(Robot,B), Holding(Package)}` | yes |
| 4 | `Drop(Package,C)` | `At(Robot,C)`, `Holding(Package)` | S3 `{At(Robot,C), Holding(Package)}` | yes |

The explanation and the executed transitions agree.

**Which to trust more?** (b), the independently executed state transitions.
The explanation is text produced by the same kind of process that produced the
plan. It can be fluent and wrong, and nothing forces it to match what the code
does. The executed transitions are the result of applying the stated rules
mechanically, and `check_plan()` repeats them with separately written rules.
An explanation is useful for understanding a plan; it is not evidence that the
plan is valid.

## Reflection questions

**1. Why specify preconditions and effects before asking for the planner?**
They are the definition of the problem. With them written down, the planner is
a small mechanical program and there is something exact to test it against.
Without them the LLM would have to invent the rules, and there would be no way
to tell a wrong plan from a right one.

**2. An error that could occur if preconditions were not checked.** The planner
could return `Move(A,B)`, `Move(B,C)`, `Drop(Package,C)`: three actions, and
the final state contains `At(Package,C)`. The goal test passes, but the robot
never held the package.

**3. Why is a plan that "looks reasonable" not necessarily valid?** Validity
depends on the state at each step, not on the list as a whole. The handout's
example sequence, `Move(A,B)`, `PickUp(Package,B)`, `Move(B,C)`,
`Drop(Package,C)`, reads naturally and fails at its second action.

**4. What did the LLM contribute?** The Python implementation: the action
representation, the applicability and apply functions, the BFS, the state
trace, and the test script.

**5. What had to be verified independently?** That each action's preconditions
and effects in the code match the handout; that the returned plan is valid
when replayed by rules written separately from the planner; that "No plan
found" is reported when `PickUp` is missing; and that a state with the robot
at C and the package at A is not accepted as the goal.

**6. Where is logical reasoning used?** In `is_applicable()` (does the state
entail the preconditions?), in `apply()` (what is true afterwards?), and in
`satisfies()` (does the state entail the goal?).

**7. How is planning related to search?** Planning is search in a different
state space. In the search lab a state was a grid position and the successors
came from the map. Here a state is a set of facts and the successors come from
the applicable actions. The BFS is the same algorithm, and a plan is the path
it returns.

## Optional extension: Prolog as a logical verifier

`planner.pl` contains the facts and rules from the handout.

> **Not executed.** SWI-Prolog was not installed on the machine used for this
> lab, so the answers below are the results that follow from the facts and
> rules by hand, not recorded output. They can be checked by loading
> `planner.pl` in SWI-Prolog or at swish.swi-prolog.org.

### Task 6

| Query | Expected answer |
|---|---|
| `?- can_move(a,b).` | `true` |
| `?- can_move(a,c).` | `false` |

**(a)** `can_move(a,b)` reduces to `connected(a,b)`, which is a fact.

**(b)** `can_move(a,c)` reduces to `connected(a,c)`. No fact or rule produces
it, so the query fails. Prolog reports `false`, meaning "cannot be proved from
this knowledge base".

**(c)** The rule `can_move(X,Y) :- connected(X,Y).` is the implication
`Connected(X,Y) -> CanMove(X,Y)` written with the conclusion first. `:-` reads
"if", and X and Y are universally quantified.

### Task 7

| Query | Expected answer |
|---|---|
| `?- valid_move(a,b).` | `true` |
| `?- valid_move(b,c).` | `true` |
| `?- valid_move(a,c).` | `false` |

**Challenge.** A proposed `Move(a,c)` is checked with `?- valid_move(a,c).`
It fails, so the action is not supported by the warehouse knowledge and a plan
containing it should be rejected. The Python checker rejects the same move for
the same reason (see "Checker sanity" above).

### Task 8

`?- reduce_speed.` succeeds. Prolog needs `slippery`, which needs `wet_road`,
which is a fact.

```
wet_road  =>  (wet_road -> slippery)  =>  (slippery -> reduce_speed)  =>  reduce_speed
  Fact               Rule                         Rule                    Conclusion
```

### Reflection

**1. Fact versus rule.** A fact is stated to be true unconditionally
(`connected(a,b).`). A rule is true when its body can be proved
(`can_move(X,Y) :- connected(X,Y).`).

**2. Query and knowledge base.** A query asks whether a statement can be
derived from the facts and rules. `true` means a proof was found; `false`
means none exists in this knowledge base.

**3. Why verify a Python plan with Prolog?** The Prolog program states the
warehouse knowledge separately and in a different form. If the Python planner
has a mistake in its action model, a second program that reuses the same code
would repeat it; one written independently would disagree.

**4. Advantage of an independent verifier for LLM-assisted work.** The check
does not depend on the LLM being right, either in the code it wrote or in its
explanation of that code. The LLM generates a candidate; a separate logical
system decides whether to accept it.

## LLM use

`planner.py`, `run_tests.py`, `planner.pl` and this report were produced with
Claude (Claude Code) working from the lab handout. Both Python programs gave
the results above on their first execution and were not modified afterwards.
