# CS F407 Artificial Intelligence: lab work

Lab exercises for CS F407 (Artificial Intelligence), AY 2026-27 Semester 1.

Each week's folder holds the lab handout, the code, a report answering the
handout's tasks and questions, and the saved output the report's numbers come
from.

| Folder | Lab | Main program | Report |
|---|---|---|---|
| [Week1_Neural_Models](Week1_Neural_Models/) | Neural models: XOR, backpropagation checks, symmetry, activations, softmax | `xor_lab.py` | [REPORT.md](Week1_Neural_Models/REPORT.md) |
| [Week2_Agents](Week2_Agents/) | Goal-based agent for warehouse navigation | `warehouse_agent.py` | [REPORT.md](Week2_Agents/REPORT.md) |
| [Week3_Search](Week3_Search/) | A* and BFS, tests, heuristic investigation | `astar.py`, `bfs.py` | [REPORT.md](Week3_Search/REPORT.md) |
| [Week4_Logic](Week4_Logic/) | Logical planning with BFS, optional Prolog verifier | `planner.py` | [REPORT.md](Week4_Logic/REPORT.md) |
| [Week8_BN_Lab](Week8_BN_Lab/) | Bayesian networks and autoregressive language models (first- and second-order) | `first_order_model.py`, `second_order_model.py` | [REPORT.md](Week8_BN_Lab/REPORT.md) |
## Layout of a week folder

| File | Purpose |
|---|---|
| `*.pdf` | The lab handout |
| `*.py` | Code |
| `REPORT.md` | Answers to the handout's tasks, results and reflection |
| `PROMPT.md` | How the LLM was used and the specifications it worked to |
| `results/` | Saved program output (text and CSV) |

## Running the code

Weeks 2, 3, 4 and 8 need only Python 3 and its standard library. Week 1 needs
PyTorch:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run each script from inside its own folder, for example:

```
cd Week3_Search
python run_tests.py
```

Each report lists the commands that reproduce its results.

## Use of an LLM

These labs are designed around using an LLM as an engineering assistant. The
code and reports here were produced with Claude (Anthropic), through Claude
Code, working from the lab handouts. Each `PROMPT.md` records how. The results
in each report are the output of actually running the code, and each lab
includes checks that do not rely on the model's own account of its work:
replaying returned paths and plans with separately written rules, comparing
against answers worked out by hand, and finite-difference gradient checks.
