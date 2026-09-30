# Neural Models lab: how the LLM was used

**LLM:** Claude (Anthropic), used through Claude Code.

**What it was given:** the lab handout (`neur_models_lab_ex.pdf`) and a
request to complete the tasks in it. The model wrote the specifications below
from the handout, wrote `xor_lab.py` to them, and ran it.

Each specification also works as a standalone prompt.

## Prompt 1: binary XOR model (Task 3)

Generate minimal PyTorch code for the following model and dataset. Do not
change the architecture or the task.

Dataset (all four examples, used as one full batch):

| x1 | x2 | y |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Model: 2 inputs, 2 hidden units with tanh, 1 output logit. Loss:
`BCEWithLogitsLoss` on the logit (sigmoid output). Default PyTorch random
initialisation with a fixed seed. Optimiser: Adam, learning rate 0.05, 2000
full-batch steps on CPU.

After training, report the initial loss, the final loss, all four
probabilities and the thresholded labels. Print the gradient of the
first-layer weight matrix after the first `backward()`. Make the hidden
activation a parameter (sigmoid, tanh or ReLU). Explain each test in one
sentence.

Also train a linear baseline (one affine layer, sigmoid output) on the same
data with the same settings and report the same numbers.

## Prompt 2: checks (Task 4)

Add the following experiments without changing the model.

- **Gradient check.** Show that the first-layer gradient of the mean loss
  equals the mean of the four per-example gradients, and compare it with a
  central finite-difference estimate. Use float64 for this check.
- **Symmetry.** Set every weight and bias to the same constant before training
  and print both rows of the first-layer weight matrix at several steps,
  saying whether they are identical. Run it for the constant 0 (as the lab
  asks) with tanh and with sigmoid, and for the constant 0.5 with tanh.
- **Activations.** Train with sigmoid, tanh and ReLU, changing nothing else.
  Record final loss, whether all four labels are correct, and the Euclidean
  norm of the first-layer gradient at step 0 and step 10. Repeat for seeds 0
  to 19 and save the results as CSV.

## Prompt 3: three-class extension (Task 5)

Modify only the output and the loss: three logits and `CrossEntropyLoss`, with
classes 0 = (0,0), 1 = (0,1) or (1,0), 2 = (1,1). Print the shape of the final
weight matrix, the class probabilities for all four inputs, and the sum of one
probability vector. Check numerically that the gradient of the loss with
respect to the logits is `(p - y) / 4`. Add 100 to all logits of one example
and show the probabilities are unchanged. Show what a naive
`exp(z) / sum(exp(z))` does when the logits are large, and what subtracting
the maximum first does.

## Prompt 4: follow-up after seeing the results

At seed 0 only the tanh network learns XOR. Report this as it is. Add a table
for the seeds where all three activations succeed, and for the failed sigmoid
and ReLU runs at seed 0 print the hidden pre-activations, the activations and
the activation derivative, so the reason each stopped learning can be read
off.

## Corrections made to the generated code

- **Before the first execution:** none. The code was read to confirm the
  2-2-1 architecture, the loss on logits (no second sigmoid), the labels, and
  the order `zero_grad()`, forward, `backward()`, `step()`.
- **After the first execution:** the activation experiment was extended as in
  Prompt 4. No result from the first run was wrong; the single-seed table was
  not enough evidence on its own.
