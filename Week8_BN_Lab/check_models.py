"""Part VII: test that the code implements the intended probability model.

Checks, for both models:

1. normalisation: every conditional distribution sums to 1;
2. the counts behind P(. | the), P(. | cat), ... equal values counted by hand;
3. sampling follows the table: empirical frequencies from many draws match it;
4. greedy decoding picks the argmax;
5. an unseen context gives an empty distribution and generation stops cleanly;
6. the probabilities of all complete sentences add up to 1, so the model is
   a proper distribution over sentences (computed exactly, not by sampling).

Writes results/checks.txt.

Run:  python check_models.py
"""

import random
from collections import Counter, defaultdict
from pathlib import Path

from corpus import END, START, training_data
from first_order_model import FirstOrderModel
from second_order_model import SecondOrderModel

TOLERANCE = 1e-12

# Counted by hand from the six training sentences (Question 3).
HAND_COUNTS = {
    START: {"the": 6},
    "the": {"cat": 3, "dog": 3, "mat": 2, "rug": 2, "park": 2},
    "cat": {"sat": 2, "ran": 1},
    "dog": {"sat": 2, "ran": 1},
    "sat": {"on": 4},
    "ran": {"to": 2},
}

lines = []
failures = []


def say(text=""):
    lines.append(text)


def check(name, passed):
    say(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if not passed:
        failures.append(name)


def normalisation(model, label):
    say(f"Normalisation, {label}: sum over v of P(v | context)")
    for context, distribution in model.cpt.items():
        total = sum(distribution.values())
        shown = context if isinstance(context, str) else "(" + ", ".join(context) + ")"
        say(f"  {shown:<22} {total:.12f}")
        if abs(total - 1.0) > TOLERANCE:
            failures.append(f"normalisation {label} {shown}")
    check(f"all {len(model.cpt)} distributions sum to 1 within {TOLERANCE}",
          all(abs(sum(d.values()) - 1.0) <= TOLERANCE for d in model.cpt.values()))


def sentence_mass(model, max_words=60):
    """Total probability of all sentences of up to `max_words` words.

    Pushes probability mass forward one token at a time over contexts
    (a forward pass over the chain), adding whatever reaches <END>.
    """
    mass = defaultdict(float)
    mass[tuple([START] * model.order)] = 1.0
    finished = 0.0
    for _ in range(max_words + 1):
        moved = defaultdict(float)
        for history, p in mass.items():
            for word, q in model.distribution(model.context_of(list(history))).items():
                if word == END:
                    finished += p * q
                else:
                    moved[(history + (word,))[-model.order:]] += p * q
        mass = moved
    return finished


def main():
    first = FirstOrderModel(training_data(order=1))
    second = SecondOrderModel(training_data(order=2))

    normalisation(first, "first-order")
    say()
    normalisation(second, "second-order")
    say()

    say("Counts against hand-counted values (first-order)")
    for context, expected in HAND_COUNTS.items():
        check(f"C({context}, .) = {expected}", dict(first.counts[context]) == expected)
    say()

    say("Sampling follows the table: 60000 draws from P(. | the), seed 1")
    rng = random.Random(1)
    draws = Counter(first.sample("the", rng) for _ in range(60000))
    worst = 0.0
    for word, p in sorted(first.distribution("the").items()):
        observed = draws[word] / 60000
        worst = max(worst, abs(observed - p))
        say(f"  {word:<6} model {p:.4f}   observed {observed:.4f}")
    check(f"largest gap {worst:.4f} is below 0.01", worst < 0.01)
    say()

    say("Greedy decoding picks the argmax")
    for context in ["the", "cat", "dog", "sat", "ran", "on", "to"]:
        options = first.distribution(context)
        pick = first.most_probable(context)
        check(f"most_probable({context}) = {pick}", options[pick] == max(options.values()))
    say()

    say("Unseen context")
    check("distribution('bird') is empty", first.distribution("bird") == {})
    check("most_probable('bird') is None", first.most_probable("bird") is None)
    check("second-order distribution(('cat', 'park')) is empty",
          second.distribution(("cat", "park")) == {})
    say()

    say("First word is predicted from the padded START context")
    check("first-order P(the | <START>) = 1", first.distribution(START) == {"the": 1.0})
    check("second-order P(the | <START>, <START>) = 1",
          second.distribution((START, START)) == {"the": 1.0})
    say()

    say("Total probability of all complete sentences (exact forward pass)")
    for words in (5, 10, 20, 60):
        say(f"  first-order, sentences up to {words:>2} words: {sentence_mass(first, words):.10f}")
    for words in (5, 6, 10):
        say(f"  second-order, sentences up to {words:>2} words: {sentence_mass(second, words):.10f}")
    check("first-order mass tends to 1", abs(sentence_mass(first, 200) - 1.0) < 1e-9)
    check("second-order mass is exactly 1 by length 6",
          abs(sentence_mass(second, 6) - 1.0) < TOLERANCE)
    say()

    say(f"All checks passed: {not failures}")
    text = "\n".join(lines)
    print(text)
    out = Path(__file__).parent / "results" / "checks.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text + "\n")


if __name__ == "__main__":
    main()
