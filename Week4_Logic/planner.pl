% Optional extension (Tasks 6-8): Prolog as a logical verifier.
% Facts and rules are the ones given in the lab handout.
%
% Load with:  swipl planner.pl
% The queries and their answers are listed in REPORT.md.

% Task 6: which locations are directly connected.
connected(a, b).
connected(b, a).
connected(b, c).
connected(c, b).

% Connected(X, Y) -> CanMove(X, Y)
can_move(X, Y) :-
    connected(X, Y).

% Task 7: a proposed move is valid only if the warehouse supports it.
valid_move(X, Y) :-
    connected(X, Y).

% Task 8: one fact and two rules.
wet_road.

slippery :-
    wet_road.

reduce_speed :-
    slippery.
