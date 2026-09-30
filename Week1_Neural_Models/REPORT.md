# Neural Models lab: learning, depth, activations and output layers

| File | Contents |
|---|---|
| `neur_models_lab_ex.pdf` | Lab handout |
| `xor_lab.py` | All experiments: linear baseline, binary XOR, gradient checks, symmetry, activations, three-class extension |
| `PROMPT.md` | How the LLM was used, the specifications it worked to, and corrections |
| `results/output.txt` | Full output of `python xor_lab.py` |
| `results/*.csv` | Activation tables: seed 0, solved seeds, all 60 runs, and the per-activation summary |

Run with `python xor_lab.py` (needs PyTorch; see `requirements.txt` in the
repository root). Settings used throughout: Adam, learning rate 0.05, 2000
full-batch steps, seed 0 unless stated, torch 2.14.0 on CPU.

## Task 1: problem specification

- **Input space:** `X = {0,1} x {0,1}`, the readings of the two sensors.
- **Output space:** `Y = {0,1}`, where 1 means "raise the disagreement
  warning".
- **Examples:** (0,0) -> 0, (0,1) -> 1, (1,0) -> 1, (1,1) -> 0.

```
 x2
  1 |  (0,1) y=1          (1,1) y=0
    |
    |
  0 |  (0,0) y=0          (1,0) y=1
    +--------------------------------- x1
          0                  1
```

**Why one straight line cannot separate the classes.** The two class-1 points
sit on one diagonal of the square and the two class-0 points on the other. A
line `w1*x1 + w2*x2 + b = 0` that puts (0,1) and (1,0) on the positive side
needs `w2 + b > 0` and `w1 + b > 0`. Adding them gives `w1 + w2 + 2b > 0`.
Putting (0,0) on the negative side needs `b < 0`, so `w1 + w2 + b > -b > 0`,
which puts (1,1) on the positive side as well. All four constraints cannot
hold together.

**Prediction for a single affine map plus sigmoid.** It cannot classify all
four points. Because the data are symmetric, the best it can do under
cross-entropy is to output 0.5 for every input, giving a loss of ln 2 = 0.6931.

**Measured.** Initial loss 0.7168, final loss 0.6931, probabilities
0.5000 for all four inputs, 2/4 correct. The prediction held.

**Think About It.** XOR tests the claim that what a model can represent
depends on the kind of function it computes, not on how many parameters it
has. No setting of an affine model's parameters solves XOR, and adding more
affine layers does not help because their composition is still affine. One
nonlinear hidden layer with two units is enough.

## Task 2: model design and validation criteria

Model: 2 inputs -> 2 hidden units (tanh) -> 1 output logit, sigmoid output,
binary cross-entropy, Adam.

**1. Why the hidden nonlinearity is necessary.** Without it the network is
`W2(W1 x + b1) + b2`, a single affine map, and Task 1 showed that no affine
map separates XOR. The nonlinearity lets the hidden layer move the four points
to new positions where a single line does separate them.

**2. Why sigmoid with binary cross-entropy.** The target is one yes/no answer,
so the output should be one probability, which is what a sigmoid gives. Binary
cross-entropy is the negative log-likelihood of that probability. Together
they give a logit gradient of `p - y`, which stays large when the prediction
is confidently wrong. In PyTorch the two are combined in `BCEWithLogitsLoss`,
which is applied to the logit and is numerically stable.

**3. Evidence that counts as successful learning.**

1. The final loss is far below the linear baseline's 0.6931 (threshold used:
   below 0.01).
2. All four thresholded labels are correct.
3. The first-layer gradient is nonzero and agrees with a finite-difference
   estimate.
4. The two hidden units end up computing different features.
5. The result is checked over 20 seeds, not claimed from one run.

**Think About It.** Nothing tells a hidden unit what to compute. The only
target is at the output. Backpropagation passes the output error back through
the second-layer weights, so each hidden unit's weights move in whatever
direction lowers the output loss. The features the hidden units end up with
are whatever made the output layer's job possible.

## Task 3: prompt and generated code

The specifications are in `PROMPT.md`; the code is `xor_lab.py`. In
`train()`: the forward pass and the scalar loss are
`loss = loss_fn(model(X), targets)`, reverse-mode AD is `loss.backward()`, and
the parameters change at `optimiser.step()`.

Changes before execution: none. Changes after the first run: the activation
experiment was extended (details in `PROMPT.md`).

**Think About It.** Verifiable by reading the code: the architecture is 2-2-1,
the labels are XOR, the loss is applied to logits without a second sigmoid,
and the training loop zeroes gradients before each backward pass. Only
measurable by running it: whether training converges, what the gradients are,
whether symmetric units stay symmetric, and how the activations differ.

## Task 4: results

### Part A: basic learning check

| | |
|---|---|
| Initial loss | 0.7152 |
| Final loss | 0.000189 |
| Probabilities for (0,0), (0,1), (1,0), (1,1) | 0.0001, 0.9998, 0.9997, 0.0001 |
| Thresholded labels | 0, 1, 1, 0 |
| Correct | 4/4 |

Hidden activations after training:

| Input | Hidden unit 1 | Hidden unit 2 |
|---|---|---|
| (0,0) | -0.98 | -0.97 |
| (0,1) | +0.98 | -1.00 |
| (1,0) | -1.00 | +0.98 |
| (1,1) | -0.99 | -0.98 |

Unit 1 is on only for (0,1) and unit 2 only for (1,0). The two class-0 inputs
land on the same point, and the output layer separates that point from the
other two with one line.

### Part B: backpropagation check

After the first `backward()`:

```
dL/dW1 = [[ 0.0005,  0.0006],
          [-0.0426, -0.0448]]
```

`layer1.weight.grad[i][j]` is the partial derivative of the scalar loss with
respect to the weight from input `j` to hidden unit `i`, evaluated at the
current parameters. It is the matrix `dL/dW1`, with the same shape as `W1`.

**Why it is the average of the example-wise gradients.** The loss is
`L = (L1 + L2 + L3 + L4) / 4`. Differentiation is linear, so
`dL/dW1 = (dL1/dW1 + dL2/dW1 + dL3/dW1 + dL4/dW1) / 4`.

| Check | Largest absolute difference |
|---|---|
| Batch gradient vs mean of four per-example gradients | 0.00e+00 |
| Batch gradient vs central finite differences (eps = 1e-6, float64) | 5.37e-11 |

### Part C: symmetry experiment

| Initialisation | Rows of W1 identical at steps 0, 1, 2, 5, 50, 500, 2000 | Final loss | Correct |
|---|---|---|---|
| All parameters 0, tanh | yes (both rows stay exactly [0, 0]) | 0.6931 | 2/4 |
| All parameters 0, sigmoid | yes (both rows stay exactly [0, 0]) | 0.6931 | 2/4 |
| All parameters 0.5, tanh | yes (rows move, always equal) | 0.4775 | 3/4 |

**Explanation.** Two hidden units that start with the same weights compute the
same output for every input. If their outgoing weights are also equal, they
receive the same error signal, so their gradients are equal and one update
leaves them equal again. By induction they are equal forever, and the network
behaves as if it had one hidden unit. One hidden unit cannot represent XOR,
which is why the 0.5 run stops at 3/4.

The all-zero case is stronger than that: the weights do not move at all. With
`W2 = 0` no error reaches the first layer, so `dL/dW1 = 0`. The hidden outputs
are the same constant for every input, and the output errors `0.5 - y` are
+0.5, -0.5, -0.5, +0.5, which sum to zero, so the gradients of `W2` and `b2`
are zero too. The all-zero point is an exact stationary point for this
dataset, and the loss stays at ln 2.

### Part D: activation experiment

Seed 0, the seed used in the rest of the lab:

| Hidden activation | Final loss | 4/4 correct? | Early gradient norm (step 0) | Gradient norm (step 10) |
|---|---|---|---|---|
| Sigmoid | 0.477470 | no (3/4) | 0.0009 | 0.0012 |
| Tanh | 0.000189 | yes | 0.0618 | 0.0094 |
| ReLU | 0.693147 | no (2/4) | 0.0017 | 0.0000 |

Repeated over seeds 0 to 19:

| Hidden activation | Runs with 4/4 correct | Mean gradient norm at step 0 | Seeds solved |
|---|---|---|---|
| Sigmoid | 8/20 | 0.0072 | 1 2 3 4 11 12 14 16 |
| Tanh | 9/20 | 0.0329 | 0 2 3 4 10 11 13 14 19 |
| ReLU | 5/20 | 0.0373 | 2 5 11 15 18 |

The two seeds where all three activations solve the task:

| Seed | Hidden activation | Final loss | 4/4 correct? | Gradient norm (step 0) | Gradient norm (step 10) |
|---|---|---|---|---|---|
| 2 | Sigmoid | 0.000588 | yes | 0.0055 | 0.0011 |
| 2 | Tanh | 0.000204 | yes | 0.0051 | 0.0217 |
| 2 | ReLU | 0.000077 | yes | 0.0049 | 0.0068 |
| 11 | Sigmoid | 0.000614 | yes | 0.0030 | 0.0037 |
| 11 | Tanh | 0.000214 | yes | 0.0165 | 0.0545 |
| 11 | ReLU | 0.000114 | yes | 0.0647 | 0.0507 |

**Interpretation.** With only two hidden units, this network fails more often
than it succeeds under these settings, for every activation: a single seed
says little. Across 20 seeds, tanh and sigmoid solved XOR about as often (9
and 8 runs) and ReLU less often (5). The early gradient was on average about
five times smaller with sigmoid (0.0072) than with tanh (0.0329) or ReLU
(0.0373). When all three do learn (seeds 2 and 11), the final loss after the
same 2000 steps is lowest for ReLU, then tanh, then sigmoid. So in this
experiment ReLU was the fastest when it worked and the least reliable, and
sigmoid was the slowest. That is a statement about a 2-unit network on four
points, not about the activations in general.

The two failures at seed 0 have different causes, visible in the hidden layer:

- **Sigmoid, stuck at loss 0.4775.** For inputs (0,1), (1,0) and (1,1) the
  hidden pre-activations are around -10 to -21 for unit 1 and +11 to +24 for
  unit 2, so the activations are 0.0000 and 1.0000 and the derivative
  `a(1 - a)` is 0.0000. The three inputs have become the same hidden point, so
  the output must give them one probability. The best single value is 2/3
  (two of the three are class 1), and the run's probabilities for those
  inputs are 0.6666, 0.6666 and 0.6667. That accounts for the loss:
  `-(2 ln(2/3) + ln(1/3)) / 4 = 0.4774`. The units are saturated.
- **ReLU, stuck at loss 0.6931.** Both hidden pre-activations are negative for
  all four inputs (between -0.21 and -1.50), so both units output exactly 0
  and their derivative is exactly 0. By step 10 the first-layer gradient norm
  is 0.0000. The network outputs a constant and the loss is ln 2. The units
  are dead.

**Think About It.** A saturated sigmoid and a dead ReLU both give a near-zero
gradient, and the table above shows how to tell them apart. Saturated sigmoid:
pre-activations are large in magnitude, of either sign, and activations sit at
0 or 1; the derivative is tiny but not zero. Dead ReLU: pre-activations are
negative, possibly only slightly, activations are exactly 0 for every input,
and the derivative is exactly zero.

## Task 5: three-class extension

Only the output layer and the loss change: three logits, `CrossEntropyLoss`.

**Predictions before running.**

1. The final weight matrix has shape (3, 2): three logits from two hidden
   units.
2. There are three logits per example.
3. Softmax probabilities sum to one because each is `exp(z_k)` divided by the
   sum of `exp(z_j)` over all classes, so the numerators add up to the
   denominator.
4. For one example `L = -log p_c`, where `c` is the true class and `p` is the
   softmax of the logits `z`. Differentiating, `dL/dz_k = p_k - 1` when
   `k = c` and `p_k` otherwise, which is `p - y` with `y` one-hot. Raising a
   logit is penalised in proportion to the probability it already has, except
   for the correct class, which is rewarded by the probability it is missing.

**Measured.**

| | |
|---|---|
| Final weight matrix shape | (3, 2) |
| Logits per example | 3 |
| Initial loss | 1.0591 |
| Final loss | 0.000089 |
| Predicted classes | 0, 1, 1, 2 (4/4 correct) |
| Largest difference between autograd's logit gradient and `(p - y) / 4` | 7.45e-09 |

The division by 4 is the mean over the four examples.

| Input | P(class 0) | P(class 1) | P(class 2) |
|---|---|---|---|
| (0,0) | 0.9999 | 0.0001 | 0.0000 |
| (0,1) | 0.0001 | 0.9999 | 0.0000 |
| (1,0) | 0.0001 | 0.9999 | 0.0000 |
| (1,1) | 0.0000 | 0.0001 | 0.9999 |

Probability vector for (0,1): `[0.0001, 0.9999, 0.0000]`, sum 1.00000000.

**Optional diagnostic.** Adding 100 to all three logits changed the
probabilities by at most 2.51e-10. Softmax depends only on differences between
logits, because a common factor `exp(100)` cancels between numerator and
denominator. That is also why stable implementations subtract the maximum
logit first: the result is unchanged, and the largest exponent becomes
`exp(0) = 1`, so nothing overflows. With 1000 added to the logits, the naive
`exp(z) / sum(exp(z))` returned `[nan, nan, nan]`, and the version that
subtracts the maximum returned `[0.0001, 0.9999, 0.0000]`.

**Think About It.** With tens of thousands of classes the output mathematics
is unchanged: one logit per class, softmax, cross-entropy, a logit gradient of
`p - y`, probabilities that sum to one, and the same stability trick. What
changes is everything that produces the logits. The two hidden units become a
deep network that has to encode a whole context, the inputs become learned
embeddings of tokens, the final weight matrix has one row per vocabulary
entry, and the training set goes from four examples to a corpus.

## Reflection questions

**1. Depth versus nonlinearity.** The linear baseline stayed at loss 0.6931
with 2/4 correct, and stacking more affine layers would not change that,
because the composition is affine. The same data were fitted to a loss of
0.000189 by adding one hidden layer with a nonlinearity. What XOR needs is the
nonlinearity; depth only helps when there is a nonlinearity between the
layers.

**2. Evidence of a useful learning signal, not just a nonzero gradient.** The
gradient matched finite differences to 5e-11, so it is the true slope of the
loss. Following it took the loss from 0.7152 to 0.000189 and made all four
labels correct, and the hidden units ended up as two different detectors, one
for (0,1) and one for (1,0). A nonzero gradient alone would not show this: the
sigmoid run at seed 0 also had nonzero gradients at steps 0 and 10 and still
stopped at 3/4.

**3. Why identical or zero initialisation prevents distinct features.**
Identical units compute identical outputs and receive identical gradients, so
every update changes them by the same amount and they never separate. The
network keeps two copies of one feature. With all zeros it is worse here: the
gradient is exactly zero and nothing moves.

**4. Effect of the hidden activation on the gradient.** *Engineering
observation:* the mean early gradient norm over 20 seeds was 0.0072 for
sigmoid, 0.0329 for tanh and 0.0373 for ReLU, and in runs where all three
learned, sigmoid finished with the highest loss. Failed runs showed saturated
sigmoid units and dead ReLU units. *Scientific explanation:* backpropagation
multiplies the error by the activation's derivative at each hidden unit. The
sigmoid's derivative is at most 0.25 and falls toward zero when the unit
saturates. Tanh's is at most 1. ReLU's is exactly 1 when the unit is active
and exactly 0 when it is not, which passes the gradient undiminished or blocks
it completely.

**5. Why the output layer and loss are chosen together.** Together they define
what the output means and what error signal comes back. Sigmoid with binary
cross-entropy treats the output as one probability; softmax with cross-entropy
treats it as a distribution over exclusive classes. In both pairings the logit
gradient is `p - y`. A mismatched pair, such as a sigmoid output with squared
error, multiplies the error by the sigmoid's derivative and learns slowly when
the output is confidently wrong.

**6. LLM productivity and human verification.** *Productivity:* the model
produced the training loop, the per-example and finite-difference gradient
checks, and the CSV output in one pass, and the code ran on the first
execution. *Verification was essential:* the first activation table, from one
seed, showed tanh succeeding and the other two failing. Taken at face value
that reads as "tanh is best". Running 20 seeds and inspecting the hidden
layers of the failed runs showed the real picture: every activation fails on
most seeds with two hidden units, for different reasons.

**7. Tests to keep when scaling up, and tests that become too expensive.**
*Keep:* loss before and after training; accuracy on held-out data; gradient
norms per layer; the fraction of dead or saturated units per layer; a
finite-difference check on a handful of randomly chosen parameters when new
layer code is written; a few seeds. *Too expensive:* finite differences over
every parameter (two forward passes per parameter); comparing the batch
gradient with every per-example gradient; 20-seed sweeps of full training
runs; reading every prediction individually.

## LLM use

`xor_lab.py` and this report were produced with Claude (Claude Code) working
from the lab handout. Every number in this report comes from
`results/output.txt` and the CSV files, which are the saved output of running
the code.
