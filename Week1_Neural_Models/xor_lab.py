"""Neural Models lab: sensor-disagreement (XOR) network and its checks.

Sections, in the order of the handout:

    Task 1   linear baseline (one affine map + sigmoid)
    Task 4A  2-2-1 network: losses, probabilities, labels
    Task 4B  backpropagation checks on dL/dW1
    Task 4C  symmetry experiment (identical initial weights)
    Task 4D  activation experiment (sigmoid / tanh / ReLU)
    Task 5   three-class extension with softmax + cross-entropy

Run:  python xor_lab.py
"""

import csv
from pathlib import Path

import torch
from torch import nn

SEED = 0
STEPS = 2000
LEARNING_RATE = 0.05
SWEEP_SEEDS = range(20)
RESULTS = Path(__file__).parent / "results"

X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y_XOR = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
Y_CLASS = torch.tensor([0, 1, 1, 2])  # 0 both inactive, 1 disagree, 2 both active

ACTIVATIONS = {"sigmoid": torch.sigmoid, "tanh": torch.tanh, "relu": torch.relu}

torch.set_printoptions(precision=4, sci_mode=False)


class SensorNet(nn.Module):
    """2 inputs -> 2 hidden units -> `outputs` logits."""

    def __init__(self, activation="tanh", outputs=1):
        super().__init__()
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, outputs)
        self.activation = ACTIVATIONS[activation]

    def forward(self, x):
        return self.layer2(self.activation(self.layer1(x)))


def train(model, targets, loss_fn, steps=STEPS, watch=None):
    """Full-batch Adam. Returns the initial loss and whatever `watch` recorded.

    `watch(step, model)` is called after backward() and before the optimiser
    update, so it sees the gradients of the current parameters.
    """
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    initial_loss = None
    for step in range(steps):
        optimiser.zero_grad()
        loss = loss_fn(model(X), targets)  # forward pass and scalar loss
        loss.backward()  # reverse-mode AD fills parameter.grad
        if step == 0:
            initial_loss = loss.item()
        if watch is not None:
            watch(step, model)
        optimiser.step()  # parameters change here
    return initial_loss


def evaluate_binary(model):
    with torch.no_grad():
        logits = model(X)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, Y_XOR).item()
        probabilities = torch.sigmoid(logits).flatten()
    labels = (probabilities >= 0.5).long()
    correct = int((labels == Y_XOR.flatten().long()).sum())
    return loss, probabilities, labels, correct


def run_binary(activation, seed):
    """Train one randomly initialised 2-2-1 network and collect the evidence."""
    torch.manual_seed(seed)
    model = SensorNet(activation)
    gradients = {}

    def watch(step, net):
        if step in (0, 10):
            gradients[step] = net.layer1.weight.grad.clone()

    initial_loss = train(model, Y_XOR, nn.BCEWithLogitsLoss(), watch=watch)
    final_loss, probabilities, labels, correct = evaluate_binary(model)
    return {
        "model": model,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "probabilities": probabilities,
        "labels": labels,
        "correct": correct,
        "first_gradient": gradients[0],
        "norm_step_0": gradients[0].norm().item(),
        "norm_step_10": gradients[10].norm().item(),
    }


def heading(text):
    print(f"\n{'=' * 8} {text} {'=' * 8}")


# ---------------------------------------------------------------- Task 1
def linear_baseline():
    heading("Task 1: linear baseline (2 -> 1, sigmoid output)")
    torch.manual_seed(SEED)
    model = nn.Linear(2, 1)
    initial_loss = train(model, Y_XOR, nn.BCEWithLogitsLoss())
    final_loss, probabilities, labels, correct = evaluate_binary(model)
    print(f"Initial loss {initial_loss:.4f}, final loss {final_loss:.4f}")
    print("Probabilities:", probabilities)
    print("Labels:", labels.tolist(), f"-> {correct}/4 correct")
    print(f"ln 2 = {torch.log(torch.tensor(2.0)).item():.4f}")


# ---------------------------------------------------------------- Task 4A
def basic_learning():
    heading(f"Task 4A: 2-2-1 network, tanh hidden units, seed {SEED}")
    result = run_binary("tanh", SEED)
    print(f"Initial loss {result['initial_loss']:.4f}")
    print(f"Final loss   {result['final_loss']:.6f}")
    print("Probabilities:", result["probabilities"])
    print("Labels:", result["labels"].tolist(), f"-> {result['correct']}/4 correct")
    print("dL/dW1 after the first backward():")
    print(result["first_gradient"])
    with torch.no_grad():
        print("Hidden activations for the four inputs after training:")
        print(torch.tanh(result["model"].layer1(X)))


# ---------------------------------------------------------------- Task 4B
def gradient_checks():
    heading("Task 4B: backpropagation checks on dL/dW1")
    torch.manual_seed(SEED)
    model = SensorNet("tanh").double()  # float64 keeps finite differences clean
    x, y = X.double(), Y_XOR.double()
    loss_fn = nn.BCEWithLogitsLoss()

    model.zero_grad()
    loss_fn(model(x), y).backward()
    batch_gradient = model.layer1.weight.grad.clone()

    # 1. The mean loss gives the mean of the four per-example gradients.
    per_example = []
    for i in range(4):
        model.zero_grad()
        loss_fn(model(x[i : i + 1]), y[i : i + 1]).backward()
        per_example.append(model.layer1.weight.grad.clone())
    averaged = torch.stack(per_example).mean(dim=0)
    print("Batch gradient:")
    print(batch_gradient)
    print("Mean of the four per-example gradients:")
    print(averaged)
    print(f"Largest difference: {(batch_gradient - averaged).abs().max().item():.2e}")

    # 2. Central finite differences on every entry of W1.
    epsilon = 1e-6
    numeric = torch.zeros_like(batch_gradient)
    with torch.no_grad():
        weights = model.layer1.weight
        for i in range(2):
            for j in range(2):
                original = weights[i, j].item()
                weights[i, j] = original + epsilon
                up = loss_fn(model(x), y).item()
                weights[i, j] = original - epsilon
                down = loss_fn(model(x), y).item()
                weights[i, j] = original
                numeric[i, j] = (up - down) / (2 * epsilon)
    print("Finite-difference gradient:")
    print(numeric)
    print(f"Largest difference: {(batch_gradient - numeric).abs().max().item():.2e}")


# ---------------------------------------------------------------- Task 4C
def symmetry_case(label, activation, value):
    print(f"\n-- {label} --")
    torch.manual_seed(SEED)
    model = SensorNet(activation)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)

    def watch(step, net):
        if step in (0, 1, 2, 5, 50, 500):
            row_a, row_b = net.layer1.weight.detach()
            same = torch.equal(row_a, row_b)
            print(
                f"step {step:4d}  row 1 {row_a.tolist()}  row 2 {row_b.tolist()}"
                f"  identical: {same}"
            )

    train(model, Y_XOR, nn.BCEWithLogitsLoss(), watch=watch)
    final_loss, probabilities, labels, correct = evaluate_binary(model)
    row_a, row_b = model.layer1.weight.detach()
    print(f"after {STEPS} steps: rows identical: {torch.equal(row_a, row_b)}")
    print(f"final loss {final_loss:.4f}, labels {labels.tolist()}, {correct}/4 correct")


def symmetry_experiment():
    heading("Task 4C: symmetry experiment")
    symmetry_case("all parameters 0, tanh hidden units", "tanh", 0.0)
    symmetry_case("all parameters 0, sigmoid hidden units", "sigmoid", 0.0)
    symmetry_case("all parameters 0.5, tanh hidden units", "tanh", 0.5)


# ---------------------------------------------------------------- Task 4D
def result_row(seed, activation, run):
    return {
        "Seed": str(seed),
        "Hidden activation": activation,
        "Final loss": f"{run['final_loss']:.6f}",
        "4/4 correct": "yes" if run["correct"] == 4 else f"no ({run['correct']}/4)",
        "Grad norm step 0": f"{run['norm_step_0']:.4f}",
        "Grad norm step 10": f"{run['norm_step_10']:.4f}",
    }


def activation_experiment():
    heading("Task 4D: activation experiment")
    RESULTS.mkdir(exist_ok=True)

    # Only the hidden activation changes between the three runs of one seed:
    # the seed fixes the initial weights, so they start from the same point.
    runs = {
        (seed, activation): run_binary(activation, seed)
        for seed in SWEEP_SEEDS
        for activation in ACTIVATIONS
    }
    all_rows = [result_row(seed, activation, run) for (seed, activation), run in runs.items()]
    write_table(RESULTS / "activation_all_seeds.csv", all_rows, echo=False)

    print(f"\nSeed {SEED} (the seed used in the rest of the lab):")
    write_table(
        RESULTS / "activation_results.csv",
        [row for row in all_rows if row["Seed"] == str(SEED)],
    )

    print(f"\nRepeated runs, seeds {SWEEP_SEEDS.start} to {SWEEP_SEEDS.stop - 1}:")
    summary = []
    for activation in ACTIVATIONS:
        solved = [s for s in SWEEP_SEEDS if runs[s, activation]["correct"] == 4]
        norms = torch.tensor([runs[s, activation]["norm_step_0"] for s in SWEEP_SEEDS])
        summary.append(
            {
                "Hidden activation": activation,
                "Runs with 4/4 correct": f"{len(solved)}/{len(SWEEP_SEEDS)}",
                "Mean grad norm step 0": f"{norms.mean().item():.4f}",
                "Seeds solved": " ".join(str(s) for s in solved) or "none",
            }
        )
    write_table(RESULTS / "seed_sweep.csv", summary)

    shared = [
        seed
        for seed in SWEEP_SEEDS
        if all(runs[seed, activation]["correct"] == 4 for activation in ACTIVATIONS)
    ]
    print(f"\nSeeds where all three activations reach 4/4: {shared}")
    write_table(
        RESULTS / "activation_results_solved_seeds.csv",
        [row for row in all_rows if int(row["Seed"]) in shared],
    )

    print("\nWhy the failed runs at seed", SEED, "stopped learning:")
    for activation in ("sigmoid", "relu"):
        model = runs[SEED, activation]["model"]
        with torch.no_grad():
            pre = model.layer1(X)
            post = model.activation(pre)
        if activation == "sigmoid":
            slope = post * (1 - post)
        else:
            slope = (pre > 0).float()
        print(f"{activation}: hidden pre-activations (rows = inputs 00, 01, 10, 11)")
        print(pre)
        print(f"{activation}: hidden activations")
        print(post)
        print(f"{activation}: activation derivative at those points")
        print(slope)


def write_table(path, rows, echo=True):
    header = list(rows[0])
    if echo:
        print(" | ".join(header))
        for row in rows:
            print(" | ".join(row[key] for key in header))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------- Task 5
def three_class_extension():
    heading(f"Task 5: three-class extension, tanh hidden units, seed {SEED}")
    torch.manual_seed(SEED)
    model = SensorNet("tanh", outputs=3)
    loss_fn = nn.CrossEntropyLoss()  # takes logits; applies log-softmax itself
    print("Final weight matrix shape:", tuple(model.layer2.weight.shape))
    print("Logits per example:", model(X).shape[1])

    # Logit gradient before training: should equal (p - y) / 4 for a mean loss.
    logits = model(X)
    logits.retain_grad()
    loss_fn(logits, Y_CLASS).backward()
    with torch.no_grad():
        p_minus_y = torch.softmax(logits, dim=1) - nn.functional.one_hot(Y_CLASS, 3)
    difference = (logits.grad - p_minus_y / 4).abs().max().item()
    print("dL/dlogits from autograd:")
    print(logits.grad)
    print("(p - y) / 4:")
    print(p_minus_y / 4)
    print(f"Largest difference: {difference:.2e}")

    initial_loss = train(model, Y_CLASS, loss_fn)
    with torch.no_grad():
        logits = model(X)
        probabilities = torch.softmax(logits, dim=1)
        final_loss = loss_fn(logits, Y_CLASS).item()
    predicted = probabilities.argmax(dim=1)
    correct = int((predicted == Y_CLASS).sum())
    print(f"Initial loss {initial_loss:.4f}, final loss {final_loss:.6f}")
    print("Class probabilities (rows = inputs 00, 01, 10, 11):")
    print(probabilities)
    print("Predicted classes:", predicted.tolist(), f"-> {correct}/4 correct")
    print("Probability vector for input (0,1):", probabilities[1])
    print(f"Its sum: {probabilities[1].sum().item():.8f}")

    # Optional diagnostic: softmax ignores a constant added to every logit.
    shifted = torch.softmax(logits[1] + 100.0, dim=0)
    print(
        "Largest change after adding 100 to the logits: "
        f"{(shifted - probabilities[1]).abs().max().item():.2e}"
    )
    big = logits[1] + 1000.0
    naive = torch.exp(big) / torch.exp(big).sum()
    stable = torch.exp(big - big.max()) / torch.exp(big - big.max()).sum()
    print("Naive exp/sum with logits + 1000:     ", naive)
    print("Max subtracted first, logits + 1000:  ", stable)


if __name__ == "__main__":
    print(f"torch {torch.__version__}, Adam lr={LEARNING_RATE}, {STEPS} full-batch steps")
    linear_baseline()
    basic_learning()
    gradient_checks()
    symmetry_experiment()
    activation_experiment()
    three_class_extension()
