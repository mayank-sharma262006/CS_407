# BN lab: how the LLM was used

**LLM:** Claude (Anthropic), used through Claude Code.

**What it was given:** the lab handout (`BN_Lab.pdf`) and a request to
complete the tasks in it. The model wrote the specifications below from the
handout, wrote the code to them, and ran it.

Each specification also works as a standalone prompt.

## Prompt 1: first-order model (Part V)

Write a simple Python implementation of a first-order autoregressive language
model, `P(X_t | X_t-1)`, estimated from transition counts, with sampling-based
generation.

Training data: the six sentences below, lower-cased, split on spaces, each
wrapped as `<START> ... <END>`.

```
the cat sat on the mat
the cat sat on the rug
the dog sat on the mat
the dog ran to the park
the cat ran to the park
the dog sat on the rug
```

The model should:

1. take a list of tokenised sentences as training data;
2. count transitions between consecutive tokens in a nested
   `counts[previous][next]` structure;
3. construct `P(next | previous) = C(previous, next) / sum_k C(previous, k)`;
4. return the distribution for a given previous token, and an empty
   distribution for a token never seen as a previous token;
5. predict the most probable next token, breaking ties alphabetically so the
   result is reproducible;
6. generate a sentence starting from `<START>`, either by sampling from the
   distribution or greedily, using a seeded random generator;
7. stop when `<END>` is generated, when the context has no observed
   transitions, or after 25 words, and report which of these happened;
8. compute the probability of a whole sentence by the chain rule.

Do not use a machine-learning library or a pretrained language model. Use
ordinary Python data structures and the `random` module.

## Prompt 2: second-order model (Part XII)

What should change in the probabilistic model: the context becomes the pair
`(X_t-2, X_t-1)`, so the table becomes `P(X_t | X_t-2, X_t-1)`, estimated from
counts of observed triples. Sentences are padded with two `<START>` tokens so
that the first word is predicted from `(<START>, <START>)` and the second from
`(<START>, x1)`. Nothing else changes.

Modify the existing first-order model into a second-order model that does
this. Reuse the prediction and generation code; change only how counts are
collected and what the context is. Do not replace the model with a neural
network or a pretrained language model.

## Prompt 3: tests (Part VII)

Write a script that checks the implementation against the probability model:
that every conditional distribution sums to 1; that the counts for `the`,
`cat`, `dog`, `sat`, `ran` and `<START>` equal values I state by hand; that
sampling 60,000 times from `P(. | the)` reproduces the table to within 0.01;
that greedy decoding picks the argmax; that unseen contexts give an empty
distribution; and that the probabilities of all complete sentences add up to
1, computed exactly with a forward pass over contexts. Save the results.

## Prompt 4: experiments (Parts VIII-XIII)

Write a script that saves: the full CPTs of both models; next-word
distributions and argmax for at least five contexts; 20 sampled first-order
sentences; five greedy and five sampled sentences from each model; and a
comparison of the two models covering number of parameters, contexts with no
data, diversity over 1,000 samples, and chain-rule probabilities of example
sentences.
