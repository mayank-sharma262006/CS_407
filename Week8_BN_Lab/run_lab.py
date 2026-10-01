"""Parts IV, VIII-XIII: CPTs, next-word prediction, generation, comparison.

Writes results/cpts.txt, results/predictions.txt, results/generated_text.txt
and results/comparison.txt.

Run:  python run_lab.py
"""

import random
from collections import Counter
from pathlib import Path

from corpus import END, SENTENCES, START, training_data
from first_order_model import FirstOrderModel
from second_order_model import SecondOrderModel

RESULTS = Path(__file__).parent / "results"
SAMPLE_SEED = 2026
DIVERSITY_SEED = 7
DIVERSITY_RUNS = 1000


def name(context):
    return context if isinstance(context, str) else "(" + ", ".join(context) + ")"


def cpt_lines(model):
    out = []
    for context in model.cpt:
        counts = model.counts[context]
        total = sum(counts.values())
        cells = ", ".join(
            f"{word} {counts[word]}/{total} = {p:.3f}"
            for word, p in sorted(model.cpt[context].items(), key=lambda kv: (-kv[1], kv[0]))
        )
        out.append(f"P(. | {name(context)}): {cells}")
    return out


def write(filename, lines):
    RESULTS.mkdir(exist_ok=True)
    text = "\n".join(lines)
    (RESULTS / filename).write_text(text + "\n")
    print(text)
    print()


def generations(model, mode, count, rng):
    out = []
    for i in range(1, count + 1):
        words, stop = model.generate(mode, rng)
        note = "" if stop == "end" else f"   [stopped: {stop}]"
        out.append(f"{i:>2}. {' '.join(words)}{note}")
    return out


def vocabulary():
    words = sorted({w for s in SENTENCES for w in s.split()})
    return words, words + [END]


def comparison(first, second):
    words, outcomes = vocabulary()
    lines = ["MODEL COMPARISON", ""]
    lines.append(f"Vocabulary: {len(words)} words, plus <START> and <END>")
    lines.append(f"Possible next tokens: {len(outcomes)} (the words and <END>)")
    lines.append("")

    # Contexts the model could in principle be asked about.
    possible = {
        "first-order": [START] + words,
        "second-order": [(START, START)]
        + [(START, w) for w in words]
        + [(a, b) for a in words for b in words],
    }
    lines.append("Parameters and contexts")
    lines.append(
        "  model         | possible contexts | full CPT entries | free parameters "
        "| contexts seen | nonzero probabilities | contexts with no data"
    )
    for label, model in (("first-order", first), ("second-order", second)):
        contexts = possible[label]
        seen = [c for c in contexts if c in model.cpt]
        nonzero = sum(len(model.cpt[c]) for c in seen)
        lines.append(
            f"  {label:<13} | {len(contexts):>17} | {len(contexts) * len(outcomes):>16} "
            f"| {len(contexts) * (len(outcomes) - 1):>15} | {len(seen):>13} "
            f"| {nonzero:>21} | {len(contexts) - len(seen):>21}"
        )
    lines.append("  (free parameters = contexts x (outcomes - 1), since each row sums to 1)")
    lines.append("")

    lines.append(f"Diversity over {DIVERSITY_RUNS} sampled sentences per model (seed {DIVERSITY_SEED})")
    training = set(SENTENCES)
    for label, model in (("first-order", first), ("second-order", second)):
        rng = random.Random(DIVERSITY_SEED)
        sampled = [model.generate("sample", rng) for _ in range(DIVERSITY_RUNS)]
        texts = [" ".join(words) for words, _ in sampled]
        stops = Counter(stop for _, stop in sampled)
        distinct = set(texts)
        copies = sum(t in training for t in texts)
        lengths = [len(words) for words, _ in sampled]
        lines.append(f"  {label}:")
        lines.append(f"    distinct sentences: {len(distinct)}")
        lines.append(f"    samples identical to a training sentence: {copies}/{DIVERSITY_RUNS}")
        lines.append(f"    distinct sentences not in the training data: {len(distinct - training)}")
        lines.append(f"    length in words: min {min(lengths)}, max {max(lengths)}, "
                     f"mean {sum(lengths) / len(lengths):.2f}")
        lines.append(f"    stopped at <END>: {stops['end']}, hit length limit: {stops['length limit']}")
        lines.append("    most frequent:")
        for text, n in Counter(texts).most_common(3):
            lines.append(f"      {n:>4} x  {text}")
    lines.append("")

    lines.append("Sentence probabilities (chain rule, including <END>)")
    examples = [
        "the cat sat on the mat",
        "the dog ran to the park",
        "the cat ran to the mat",
        "the dog sat on the cat sat on the rug",
        "the park",
        "the mat sat on the cat",
    ]
    lines.append(f"  {'sentence':<40} {'first-order':>12} {'second-order':>13}")
    for sentence in examples:
        lines.append(
            f"  {sentence:<40} {first.sentence_probability(sentence):>12.6f} "
            f"{second.sentence_probability(sentence):>13.6f}"
        )
    return lines


def main():
    first = FirstOrderModel(training_data(order=1))
    second = SecondOrderModel(training_data(order=2))

    write(
        "cpts.txt",
        ["CONDITIONAL PROBABILITY TABLES", "", "First-order: P(X_t | X_t-1)"]
        + cpt_lines(first)
        + ["", "Second-order: P(X_t | X_t-2, X_t-1)"]
        + cpt_lines(second),
    )

    lines = ["NEXT-WORD PREDICTION (first-order)", ""]
    for context in [START, "the", "cat", "dog", "sat", "ran", "on", "to", "mat"]:
        options = first.distribution(context)
        shown = ", ".join(
            f"{w} {p:.3f}" for w, p in sorted(options.items(), key=lambda kv: (-kv[1], kv[0]))
        )
        lines.append(f"P(X_t+1 | X_t = {context}): {shown}")
        lines.append(f"    argmax: {first.most_probable(context)}")
    lines += ["", "NEXT-WORD PREDICTION (second-order), same last word, different word before it", ""]
    for context in [("on", "the"), ("to", "the"), (START, "the"), ("the", "cat"), ("the", "dog")]:
        options = second.distribution(context)
        shown = ", ".join(
            f"{w} {p:.3f}" for w, p in sorted(options.items(), key=lambda kv: (-kv[1], kv[0]))
        )
        lines.append(f"P(X_t+1 | {name(context)}): {shown}")
        lines.append(f"    argmax: {second.most_probable(context)}")
    write("predictions.txt", lines)

    rng = random.Random(SAMPLE_SEED)
    write(
        "generated_text.txt",
        [f"GENERATED TEXT (seed {SAMPLE_SEED}, at most 25 words per sentence)", ""]
        + ["First-order, sampling, 20 sentences:"]
        + generations(first, "sample", 20, rng)
        + ["", "First-order, greedy, 5 sentences:"]
        + generations(first, "greedy", 5, rng)
        + ["", "First-order, sampling, 5 sentences:"]
        + generations(first, "sample", 5, rng)
        + ["", "Second-order, greedy, 5 sentences:"]
        + generations(second, "greedy", 5, rng)
        + ["", "Second-order, sampling, 5 sentences:"]
        + generations(second, "sample", 5, rng),
    )

    write("comparison.txt", comparison(first, second))


if __name__ == "__main__":
    main()
