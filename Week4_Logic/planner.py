"""Breadth-first planner for the warehouse delivery problem (Logic lab).

A state is a frozenset of propositions written as strings, for example
"At(Robot,A)". A proposition that is not in the set is false.

    Logic   is_applicable()  decides whether  S |= Preconditions(a)
            apply()          computes         S' = Apply(S, a)
    Search  find_plan()      breadth-first search over the states reached

Run:  python planner.py
"""

from collections import deque
from typing import FrozenSet, NamedTuple

CONNECTIONS = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]
LOCATIONS = ["A", "B", "C"]

INITIAL_STATE = frozenset({"At(Robot,A)", "At(Package,A)"})
GOAL = frozenset({"At(Package,C)"})


class Action(NamedTuple):
    name: str
    needs: FrozenSet[str]  # positive preconditions: must be in the state
    forbids: FrozenSet[str]  # negative preconditions: must not be in the state
    adds: FrozenSet[str]  # positive effects
    removes: FrozenSet[str]  # negative effects


def make_action(name, needs=(), forbids=(), adds=(), removes=()):
    return Action(
        name, frozenset(needs), frozenset(forbids), frozenset(adds), frozenset(removes)
    )


def move(origin, destination):
    return make_action(
        f"Move({origin},{destination})",
        needs=[f"At(Robot,{origin})"],
        adds=[f"At(Robot,{destination})"],
        removes=[f"At(Robot,{origin})"],
    )


def pick_up(location):
    # Assumption: the robot carries one package, so it must not be holding it.
    return make_action(
        f"PickUp(Package,{location})",
        needs=[f"At(Robot,{location})", f"At(Package,{location})"],
        forbids=["Holding(Package)"],
        adds=["Holding(Package)"],
        removes=[f"At(Package,{location})"],
    )


def drop(location):
    return make_action(
        f"Drop(Package,{location})",
        needs=[f"At(Robot,{location})", "Holding(Package)"],
        adds=[f"At(Package,{location})"],
        removes=["Holding(Package)"],
    )


def warehouse_actions(with_pick_up=True, with_drop=True):
    """All ground actions of the warehouse. The flags build the test variants."""
    actions = [move(origin, destination) for origin, destination in CONNECTIONS]
    if with_pick_up:
        actions += [pick_up(location) for location in LOCATIONS]
    if with_drop:
        actions += [drop(location) for location in LOCATIONS]
    return actions


def is_applicable(state, action):
    """S |= Preconditions(a)."""
    return action.needs <= state and not (action.forbids & state)


def apply(state, action):
    """S' = Apply(S, a): remove the negative effects, then add the positive ones."""
    return frozenset((state - action.removes) | action.adds)


def satisfies(state, goal):
    """S |= G."""
    return goal <= state


def find_plan(initial, goal, actions):
    """Breadth-first search. Returns a shortest list of actions, or None.

    The set of reachable states is finite and each state is queued once, so
    the search always ends: with a plan, or with None once nothing is left.
    """
    initial = frozenset(initial)
    queue = deque([initial])
    reached_by = {initial: None}  # state -> (previous state, action)

    while queue:
        state = queue.popleft()
        if satisfies(state, goal):
            plan = []
            while reached_by[state] is not None:
                state, action = reached_by[state]
                plan.append(action)
            plan.reverse()
            return plan

        for action in actions:
            if is_applicable(state, action):
                successor = apply(state, action)
                if successor not in reached_by:
                    reached_by[successor] = (state, action)
                    queue.append(successor)

    return None


def trace(initial, plan):
    """States S0, S1, ..., Sn produced by running the plan from `initial`."""
    states = [frozenset(initial)]
    for action in plan:
        if not is_applicable(states[-1], action):
            raise ValueError(f"{action.name} is not applicable in {show(states[-1])}")
        states.append(apply(states[-1], action))
    return states


def show(state):
    return "{" + ", ".join(sorted(state)) + "}"


def print_solution(initial, goal, actions):
    plan = find_plan(initial, goal, actions)
    if plan is None:
        print("No plan found")
        return None

    states = trace(initial, plan)
    print("Plan: " + " -> ".join(action.name for action in plan))
    print(f"S0: {show(states[0])}")
    for index, (action, state) in enumerate(zip(plan, states[1:]), start=1):
        print(f"S{index}: {show(state)}   after {action.name}")
    print("Goal satisfied in final state:", satisfies(states[-1], goal))
    return plan


if __name__ == "__main__":
    print("Initial state:", show(INITIAL_STATE))
    print("Goal:", show(GOAL))
    print_solution(INITIAL_STATE, GOAL, warehouse_actions())
