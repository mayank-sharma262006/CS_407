# BN lab: Bayesian networks and autoregressive language models

| File | Contents |
|---|---|
| `BN_Lab.pdf` | Lab handout |
| `corpus.py` | The six training sentences and tokenisation |
| `first_order_model.py` | Deliverable 1: first-order model, `P(X_t \| X_t-1)` |
| `second_order_model.py` | Deliverable 2: second-order model, `P(X_t \| X_t-2, X_t-1)` |
| `check_models.py` | Deliverable 5: normalisation and other probability checks |
| `run_lab.py` | CPTs, predictions, generated text, model comparison |
| `PROMPT.md` | How the LLM was used and the specifications it worked to |
| `results/cpts.txt` | Deliverable 3: conditional probability tables |
| `results/generated_text.txt` | Deliverable 4: generated sentences |
| `results/checks.txt` | Deliverable 5: test results |
| `results/predictions.txt` | Next-word distributions (Part VIII) |
| `results/comparison.txt` | First-order against second-order (Part XIII) |

Run with `python check_models.py` and `python run_lab.py` (Python 3, standard
library only). Answers to Questions 1-14 (deliverable 6) and the reflection
(deliverable 7) are below.

## Part I: from probability to language

**Question 1. Why is the autoregressive decomposition useful for generating
text?** It turns one impossible-looking object, a probability for every
possible sentence, into a sequence of small choices made one word at a time.
Each factor `P(X_t | X_1, ..., X_t-1)` is a distribution over the next word
only, which is something that can be estimated and sampled. To generate, draw
`X_1`, then draw `X_2` given `X_1`, and so on until `<END>`. The chain rule
guarantees that a sentence produced this way has exactly the probability the
joint distribution assigns it. Nothing is approximated by the decomposition
itself; approximations only enter when the factors are simplified.

## Part II: a Bayesian network for text

**Question 2. What independence assumption does the chain network make?**
Each word is conditionally independent of all earlier words given the word
immediately before it:

```
P(X_t | X_1, ..., X_t-1) = P(X_t | X_t-1)
```

equivalently, `X_t ⊥ {X_1, ..., X_t-2} | X_t-1`. So the joint distribution
factorises as `P(X_1) P(X_2 | X_1) ... P(X_T | X_T-1)`.

## Parts III and IV: dataset and conditional probability tables

The six handout sentences, lower-cased, each wrapped in `<START> ... <END>`.
The vocabulary is 10 words (`cat dog mat on park ran rug sat the to`); the next
token can be any of these or `<END>`, 11 outcomes in all.

**Question 3. `P(next word | current word)`** (counts in `results/cpts.txt`):

| Current word | Next-word distribution | Zero-probability next tokens |
|---|---|---|
| `the` | cat 3/12 = 0.250, dog 3/12 = 0.250, mat 2/12 = 0.167, rug 2/12 = 0.167, park 2/12 = 0.167 | the, sat, ran, on, to, `<END>` |
| `cat` | sat 2/3 = 0.667, ran 1/3 = 0.333 | all 9 other outcomes |
| `dog` | sat 2/3 = 0.667, ran 1/3 = 0.333 | all 9 other outcomes |
| `sat` | on 4/4 = 1.000 | all 10 other outcomes |
| `ran` | to 2/2 = 1.000 | all 10 other outcomes |

The remaining rows: `<START>` -> the (1.0); `on` -> the (1.0); `to` -> the
(1.0); `mat`, `rug`, `park` -> `<END>` (1.0).

Of the 121 cells in the full first-order table (11 contexts x 11 outcomes),
only 17 are nonzero, so 104 transitions have probability zero. Some are
impossible in English (`the the`); others, like `cat -> mat` or `the ->
<END>`, are merely unseen in six sentences.

Note: the handout's worked example (`the cat` three times, `the dog` twice,
giving 3/5 and 2/5) is illustrative. In the actual dataset `the` is followed
12 times, by five different words, so `P(cat | the) = 3/12`.

## Part V: implementation with the LLM

See `PROMPT.md`. The first-order model is `FirstOrderModel` in
`first_order_model.py`.

## Part VI: inspecting the generated code

**Question 4. Where are the transition counts stored?** In
`FirstOrderModel.counts`, a `defaultdict(Counter)` built in `__init__`:
`counts[previous][next]` is `C(previous, next)`.

**Question 5. Where is `P(X_t | X_t-1)` computed?** Also in `__init__`, in the
second loop: each row of counts is divided by its total and stored in
`self.cpt[previous][next]`. `distribution(context)` returns that row.

**Question 6. How is the next word chosen?** The program supports both, chosen
by the `mode` argument of `generate()`. `most_probable()` always returns the
highest-probability word (greedy). `sample()` draws a word at random with
probability equal to its table entry (`random.choices` with the table values
as weights). The difference: greedy is deterministic, so the same context
always gives the same word and every greedy sentence is identical. Sampling
reproduces the model's uncertainty: after `the`, it picks `cat` a quarter of
the time and `park` a sixth of the time, as the table says.

**Question 7. What happens with a word that has no observed transitions?**
`distribution()` returns an empty dictionary, `most_probable()` and `sample()`
return `None`, and `generate()` stops and reports "unseen context" instead of
inventing a word or crashing. `check_models.py` confirms this for `bird`. In
this dataset it cannot happen during generation, because every word the model
can produce has been seen followed by something. A related case is an unseen
transition from a known word, such as `cat -> mat`: it simply has probability
0, so it is never sampled, and any sentence containing it gets probability 0.

## Part VII: testing the probability model

Results from `results/checks.txt` (deliverable 5):

| Check | Result |
|---|---|
| Every first-order row sums to 1 (11 rows) | pass, each 1.000000000000 |
| Every second-order row sums to 1 (15 rows) | pass, each 1.000000000000 |
| Counts for `<START>`, `the`, `cat`, `dog`, `sat`, `ran` equal hand counts | pass |
| 60,000 samples from `P(. \| the)` match the table | pass, largest gap 0.0033 |
| Greedy decoding returns the argmax | pass |
| Unseen contexts return an empty distribution | pass |
| `P(the \| <START>) = 1` (first-order) and `P(the \| <START>, <START>) = 1` (second-order) | pass |
| Probabilities of all complete sentences sum to 1 | pass (details below) |

The last check sums the probability of every sentence up to a given length,
exactly, by pushing probability mass forward through the contexts. For the
second-order model every sentence has six words, and the total is exactly 1
by length 6. The first-order model can loop (`the cat sat on the cat sat on
...`), so it has sentences of every length. Its total is 0.5 for sentences up
to 5 words, 0.875 up to 10, 0.96875 up to 20 and 0.99997 up to 60, approaching
1. Each pass through `the` ends the sentence with probability 1/2 (by choosing
`mat`, `rug` or `park`), which is where the powers of 1/2 come from.

**Question 8. What would a total of 0.87 mean?** That row is not a probability
distribution, so the implementation does not implement the model. Some
probability mass is missing: for example, the code divides by the wrong total
(counting something in the denominator that is not in the numerator), drops
some successors, or normalises before all counts are collected. The bug would
not show up in generated text, because `random.choices` normalises its
weights silently, so sampling would still look reasonable. But every
probability the model reports, and every sentence probability computed from
them, would be wrong. This is why the invariant needs its own test.

## Part VIII: predicting the next word

From `results/predictions.txt`:

| Context | `P(X_t+1 \| X_t = context)` | argmax |
|---|---|---|
| `<START>` | the 1.000 | the |
| `the` | cat 0.250, dog 0.250, mat 0.167, park 0.167, rug 0.167 | cat (tie with dog) |
| `cat` | sat 0.667, ran 0.333 | sat |
| `dog` | sat 0.667, ran 0.333 | sat |
| `sat` | on 1.000 | on |
| `ran` | to 1.000 | to |
| `on` | the 1.000 | the |
| `to` | the 1.000 | the |
| `mat` | `<END>` 1.000 | `<END>` |

**Question 9. Are the most probable predictions what a person would expect?**
Partly. `sat -> on` and `ran -> to` look natural. Others do not. After `the`
the model rates `cat` and `dog` exactly equal, and returns `cat` only because
ties are broken alphabetically; that is a property of the code, not of
English. After `on` it is certain the next word is `the`, where a reader would
accept "a", "my", or many others. After `mat` it is certain the sentence ends.
And it gives probability zero to perfectly ordinary sentences such as "the
mat is red", because it has never seen them. A probability model like this
has no knowledge of language beyond relative frequencies in its training data.
It is calibrated to six sentences, not to how people use words. Human
expectations come from far more experience and from meaning, which the model
does not represent at all.

## Part IX: generated text

20 sampled first-order sentences (`results/generated_text.txt`, seed 2026,
at most 25 words):

```
 1. the mat
 2. the cat ran to the park
 3. the park
 4. the cat sat on the dog sat on the cat ran to the park
 5. the cat sat on the dog sat on the cat sat on the dog sat on the mat
 6. the dog sat on the rug
 7. the cat ran to the park
 8. the mat
 9. the cat sat on the dog sat on the mat
10. the park
11. the rug
12. the dog ran to the dog sat on the dog sat on the park
13. the park
14. the cat sat on the dog sat on the dog ran to the cat sat on the cat ran to
    the cat sat on the   [stopped: length limit]
15. the rug
16. the cat ran to the rug
17. the cat ran to the rug
18. the park
19. the rug
20. the park
```

## Part X: greedy against sampling

| Mode | Five sentences |
|---|---|
| First-order greedy | all five: `the cat sat on the cat sat on the cat sat on ...` until the 25-word limit |
| First-order sampling | the cat ran to the dog sat on the mat / the dog ran to the park / the dog sat on the cat sat on the cat sat on the park / the rug / the park |

**Question 10. Which mode varies more, and why?** Sampling. Greedy decoding is
deterministic: from a given context it always takes the same word, so all
five greedy sentences are identical. Worse, in a first-order model the
context after `on` is just `the` again, so greedy decoding returns to a state
it has already been in and repeats `the cat sat on` forever. It never reaches
`<END>`, because from `the` the argmax is `cat`, never `mat`, `rug` or `park`.
Sampling follows the whole distribution, so it sometimes picks lower-probability
words, including the ones that end the sentence.

## Part XI: the second-order network

**Question 11. How does the second-order model differ?**

1. **Graph structure.** Each `X_t` has two parents, `X_t-2` and `X_t-1`,
   instead of one. The network is no longer a simple chain; every node is also
   linked to the node two steps back.
2. **Conditional probability table.** Rows are indexed by pairs of tokens. In
   this vocabulary there are 111 possible contexts instead of 11, and 1,221
   table cells instead of 121.
3. **Context available.** Two previous tokens. The model can tell `on the`
   from `to the`, which the first-order model sees only as `the`.
4. **Data needed.** Much more. Each row needs its own counts, and there are
   about ten times as many rows. With six sentences only 15 of the 111
   contexts are ever observed.

## Part XII: the second-order model with the LLM

See `PROMPT.md`, Prompt 2. What should change, stated before the code was
accepted: only the context (a pair instead of a single token) and the padding
(two `<START>` tokens). In `second_order_model.py`, `SecondOrderModel` reuses
all of `FirstOrderModel`'s prediction and generation code and overrides only
`__init__` (counting triples) and `context_of()` (returning the last two
tokens).

Second-order CPT, every observed context (`results/cpts.txt`):

| Context | Distribution |
|---|---|
| (`<START>`, `<START>`) | the 1.000 |
| (`<START>`, the) | cat 0.500, dog 0.500 |
| (the, cat), (the, dog) | sat 0.667, ran 0.333 |
| (cat, sat), (dog, sat) | on 1.000 |
| (cat, ran), (dog, ran) | to 1.000 |
| (sat, on), (ran, to) | the 1.000 |
| (on, the) | mat 0.500, rug 0.500 |
| (to, the) | park 1.000 |
| (the, mat), (the, rug), (the, park) | `<END>` 1.000 |

Generated: five greedy sentences, all `the cat sat on the mat`; five sampled,
`the cat sat on the mat`, `the cat ran to the park` (twice), `the dog sat on
the rug` (twice).

## Part XIII: comparing the two models

From `results/comparison.txt`:

| Measure | First-order | Second-order |
|---|---|---|
| Possible contexts | 11 | 111 |
| Full CPT cells | 121 | 1,221 |
| Free parameters (cells minus one per row) | 110 | 1,110 |
| Contexts observed in the data | 11 | 15 |
| Nonzero probabilities | 17 | 19 |
| Contexts with no data (zero-probability contexts) | 0 | 96 |
| Distinct sentences in 1,000 samples | 194 | 6 |
| Samples identical to a training sentence | 154 / 1,000 | 1,000 / 1,000 |
| Sentence length (words) | 2 to 25, mean 6.43 | always 6 |
| Samples hitting the 25-word limit | 25 | 0 |

Chain-rule probabilities of example sentences:

| Sentence | First-order | Second-order |
|---|---|---|
| the cat sat on the mat | 0.027778 | 0.166667 |
| the dog ran to the park | 0.013889 | 0.166667 |
| the cat ran to the mat | 0.013889 | 0 |
| the dog sat on the cat sat on the rug | 0.004630 | 0 |
| the park | 0.166667 | 0 |
| the mat sat on the cat | 0 | 0 |

**Diversity and coherence, with examples.** The first-order model is diverse
and often incoherent. Its three most frequent outputs are two-word fragments
(`the rug` 160 times, `the park` 154, `the mat` 142 out of 1,000), because after
`the` it ends the sentence half the time. It also produces run-ons such as
`the cat sat on the dog sat on the cat ran to the park`. The second-order model
is perfectly coherent and has no diversity at all: its only possible outputs
are the six training sentences, each with probability exactly 1/6. Neither is
good. The first has too little context to stay grammatical; the second has
enough context to memorise its data, and not enough data for that context to
generalise. It assigns probability zero to `the cat ran to the mat`, which the
first-order model allows.

**Question 12. Why can more context help, and why is it harder to estimate?**
More context separates situations that a shorter context merges. Given only
`the`, the first-order model must use one distribution for "the start of a
sentence", "after on" and "after to", so it predicts `cat` after `on the`,
where it never occurs. With two tokens, `on the` gives mat/rug and `to the`
gives park, which are exactly right. The cost is the size of the table: each
extra token of context multiplies the number of rows by the vocabulary size
(11 to 111 here; with a 50,000-word vocabulary, a trigram table has 2.5
billion contexts). The data does not grow with it, so most rows are never
seen (96 of 111 here) and many of those that are seen rest on one or two
counts (`(dog, ran)` and `(cat, ran)` are each estimated from a single
occurrence). The estimates become exact copies of the training data, and
anything unseen gets probability zero.

## Part XIV: connection to modern language models

The objective is the same chain rule, `P(x_1, ..., x_T) = prod_t P(x_t | x_1,
..., x_t-1)` with `x_1` conditioned on `<START>`. A neural language model
avoids the table-size problem from Question 12 by computing the row it needs
from the context with shared weights, instead of storing every row. That is
why it can use thousands of tokens of context and still assign sensible
probabilities to contexts it has never seen.

## Part XV: the role of the LLM

**Question 13. Why is Approach B preferable?**

- **Specifying the intended behaviour.** "Write a language model" leaves every
  important decision to the LLM: the order, the estimator, how generation
  works, what happens at sentence boundaries. "`P(X_t | X_t-1)` from transition
  counts, with sampling" fixes the model, so there is a definite thing the code
  should do.
- **Understanding the representation.** With the model stated, it is clear
  that the code should contain a table indexed by the previous token, and
  questions such as Questions 4 to 7 have definite answers that can be checked
  in the code.
- **Validating the implementation.** A stated model gives expected values that
  can be worked out by hand (`P(cat | the) = 3/12`) and compared with the
  program.
- **Testing probabilistic invariants.** Only once the code is known to be a
  set of conditional distributions does it make sense to test that each sums
  to 1, that sampling follows the table, and that sentence probabilities sum
  to 1. With Approach A it may not even be clear what to test.
- **Distinguishing implementation from model.** The same model can be coded in
  many ways. Tie-breaking, the 25-word limit and the use of `random.choices`
  are implementation choices; the CPT is the model. Approach B keeps the two
  apart, so a bug in one is not mistaken for a property of the other.

## Final question

**Question 14. What did thinking of the model as a Bayesian network add?**

- **A representation of dependencies.** The graph states which earlier words a
  word depends on. Moving from first to second order is one edge per node,
  and it is immediately clear what the new table must be indexed by.
- **A factorisation of the joint distribution.** The probability of a whole
  sentence is a product of table entries. That is how `sentence_probability()`
  works, and how the exact total-probability check in `check_models.py` works.
- **A principled method for generation.** Ancestral sampling, sampling each
  node given its parents in order, is the generation procedure, and it
  produces sentences with exactly the probabilities the model assigns.
- **A way to reason about independence assumptions.** The first-order
  assumption `X_t ⊥ X_t-2 | X_t-1` is what makes greedy decoding loop: after
  `on`, the model has forgotten everything except `the`.
- **A way to understand the effect of increasing context.** More parents means
  a larger CPT for the same data. Questions 11 and 12 follow directly.
- **A way to test the implementation against its specification.** Every
  node's CPT must be a distribution for each setting of its parents. That
  gives the normalisation test and the sampling test.

## Reflection: how the LLM was used and how its output was validated

The code and this report were produced with Claude (Claude Code) working from
the lab handout, to the specifications in `PROMPT.md`. Validation did not rely
on the model's description of its own code. The counts were compared with
values counted by hand; every distribution was checked to sum to 1; sampling
was checked against the table over 60,000 draws; and the total probability of
all sentences was computed exactly and shown to be 1. All checks passed on the
first run, and no code was changed after it.

Examples of generated code that was inspected:

- **Second-order start padding.** A second-order model needs two tokens of
  context before the first word. If sentences are padded with only one
  `<START>`, the first context available is `(<START>, the)`, and the model
  can never generate the first word: its output would begin at `cat` or `dog`.
  The code pads with two, and two checks confirm it: `P(the | <START>,
  <START>) = 1`, and every second-order sentence begins with `the`.
- **Tie-breaking in `most_probable()`.** `P(cat | the)` and `P(dog | the)` are
  both 0.25. A plain `max()` over a dictionary would return whichever word was
  inserted first, which depends on the order of the training sentences. The
  code breaks ties alphabetically, so the greedy result is reproducible, and
  the report says so (Question 9) rather than presenting `cat` as the model's
  preference.
- **Greedy decoding needs a length limit.** The first-order greedy sentence
  never reaches `<END>` (Question 10). Without the 25-word limit in
  `generate()`, greedy generation would never stop.
