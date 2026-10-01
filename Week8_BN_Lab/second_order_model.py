"""Second-order autoregressive language model: each X_t has two parents,

    X_t-2 -> X_t <- X_t-1

so the conditional probability table is P(X_t | X_t-2, X_t-1), estimated from
counts of observed triples:

    P(w_k | w_i, w_j) = C(w_i, w_j, w_k) / sum_m C(w_i, w_j, w_m)

What changes from the first-order model is only the context: a pair of
tokens instead of one token. Sentences are padded with two <START> tokens so
that the first word is predicted from (<START>, <START>) and the second from
(<START>, x1). Counting, normalising, prediction and generation are
inherited unchanged.

Run:  python second_order_model.py
"""

import random
from collections import Counter, defaultdict

from corpus import training_data
from first_order_model import FirstOrderModel, show_cpt


class SecondOrderModel(FirstOrderModel):
    order = 2

    def __init__(self, sentences):
        # counts[(previous_2, previous_1)][next] = C(previous_2, previous_1, next)
        self.counts = defaultdict(Counter)
        for tokens in sentences:
            for a, b, nxt in zip(tokens, tokens[1:], tokens[2:]):
                self.counts[(a, b)][nxt] += 1

        self.cpt = {}
        for context, followers in self.counts.items():
            total = sum(followers.values())
            self.cpt[context] = {nxt: n / total for nxt, n in followers.items()}

    def context_of(self, history):
        return tuple(history[-2:])


if __name__ == "__main__":
    model = SecondOrderModel(training_data(order=2))
    show_cpt(model, [("on", "the"), ("to", "the"), ("the", "cat"), ("cat", "sat")])
    rng = random.Random(8)
    print("\nFive sampled sentences:")
    for _ in range(5):
        words, stop = model.generate("sample", rng)
        print("   ", " ".join(words), f"[{stop}]")
