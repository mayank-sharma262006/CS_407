"""Training data for the BN lab (the starting dataset from the handout)."""

START = "<START>"
END = "<END>"

SENTENCES = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]


def tokenize(sentence, order=1):
    """Lower-case, split on spaces, and add boundary tokens.

    `order` START tokens are added so that the first word also has a full
    context: P(x1 | <START>) for order 1, P(x1 | <START>, <START>) for order 2.
    """
    return [START] * order + sentence.lower().split() + [END]


def training_data(order=1):
    return [tokenize(sentence, order) for sentence in SENTENCES]
