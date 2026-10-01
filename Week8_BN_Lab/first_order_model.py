"""First-order autoregressive language model: the Bayesian network

    <START> -> X1 -> X2 -> ... -> XT -> <END>

Each word depends only on the word before it, so the model is one
conditional probability table, P(X_t | X_t-1), estimated by counting:

    P(w_j | w_i) = C(w_i, w_j) / sum_k C(w_i, w_k)

Run:  python first_order_model.py
"""

import math
import random
from collections import Counter, defaultdict

from corpus import END, START, training_data


class FirstOrderModel:
    order = 1

    def __init__(self, sentences):
        # counts[previous][next] = C(previous, next)
        self.counts = defaultdict(Counter)
        for tokens in sentences:
            for previous, nxt in zip(tokens, tokens[1:]):
                self.counts[previous][nxt] += 1

        # cpt[previous][next] = P(next | previous)
        self.cpt = {}
        for previous, followers in self.counts.items():
            total = sum(followers.values())
            self.cpt[previous] = {nxt: n / total for nxt, n in followers.items()}

    def context_of(self, history):
        """The part of the history this model conditions on."""
        return history[-1]

    def distribution(self, context):
        """P(. | context) as a dict. Empty if the context was never seen."""
        return self.cpt.get(context, {})

    def most_probable(self, context):
        """argmax_w P(w | context); ties broken alphabetically. None if unseen."""
        options = self.distribution(context)
        if not options:
            return None
        best = max(options.values())
        return min(word for word, p in options.items() if p == best)

    def sample(self, context, rng):
        """Draw one token from P(. | context). None if unseen."""
        options = self.distribution(context)
        if not options:
            return None
        words = sorted(options)
        return rng.choices(words, weights=[options[w] for w in words])[0]

    def generate(self, mode="sample", rng=None, max_words=25):
        """Generate one sentence. Returns (words, how it stopped).

        Stops on <END>, on a context with no observed transitions, or after
        `max_words` words (greedy decoding can cycle forever).
        """
        history = [START] * self.order
        words = []
        while len(words) < max_words:
            context = self.context_of(history)
            if mode == "greedy":
                nxt = self.most_probable(context)
            else:
                nxt = self.sample(context, rng)
            if nxt is None:
                return words, "unseen context"
            if nxt == END:
                return words, "end"
            words.append(nxt)
            history.append(nxt)
        return words, "length limit"

    def sentence_probability(self, sentence):
        """P(sentence) by the chain rule, including the final <END>."""
        history = [START] * self.order
        log_p = 0.0
        for word in sentence.split() + [END]:
            p = self.distribution(self.context_of(history)).get(word, 0.0)
            if p == 0.0:
                return 0.0
            log_p += math.log(p)
            history.append(word)
        return math.exp(log_p)


def show_cpt(model, contexts):
    for context in contexts:
        counts = model.counts.get(context, {})
        total = sum(counts.values())
        print(f"P(next | {context})   [{total} observed transitions]")
        for word, p in sorted(model.distribution(context).items(), key=lambda kv: -kv[1]):
            print(f"    {word:<8} {counts[word]}/{total} = {p:.3f}")


if __name__ == "__main__":
    model = FirstOrderModel(training_data(order=1))
    show_cpt(model, ["the", "cat", "dog", "sat", "ran"])
    print("\nMost probable word after 'the':", model.most_probable("the"))
    rng = random.Random(8)
    print("\nFive sampled sentences:")
    for _ in range(5):
        words, stop = model.generate("sample", rng)
        print("   ", " ".join(words), f"[{stop}]")
