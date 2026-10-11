"""Task 01: n-gram counting."""
from typing import Iterable
from collections import defaultdict, Counter
from tqdm import tqdm

EOS = "_EOS_"
UNK = "_UNK_"


def count_ngrams(lines: Iterable[str], n: int):
    """
    Count how many times each word occured after (n - 1) previous words
    :param lines: an iterable of strings with space-separated tokens
    :returns: a dictionary { tuple(prefix_tokens): {next_token_1: count_1, next_token_2: count_2}}

    When building counts, please consider the following two edge cases:
    - every prefix must contain exactly (n - 1) tokens. If a prefix is shorter than that,
      pad it with UNK on the left. The number of padding tokens therefore depends on n:
      derive it from n, do not hard-code it. For n=3,
      empty prefix: "" -> (UNK, UNK)
      short prefix: "the" -> (UNK, the)
      long prefix: "the new approach" -> (new, approach)
      For n=2 the very same prefixes give (UNK,), (the,) and (approach,), and for n=1 all
      of them give the empty tuple (). Equivalently: prepend (n - 1) UNK tokens to a line,
      then slide a window of size n over it.
    - you should add a special token, EOS, at the end of each sequence
      "... with deep neural networks ." -> (..., with, deep, neural, networks, ., EOS)
      count the probability of this token just like all others.

    Useful invariant: a line with L tokens always contributes exactly L + 1 counted events
    (its L tokens plus EOS), for every n. Note also that NGramLanguageModel below already
    implements this padding rule in get_possible_next_tokens - keep the two consistent, or
    the model will look up prefixes that you never counted.
    """
    counts = defaultdict(Counter)
    # counts[(word1, word2)][word3] = how many times word3 occured after (word1, word2)

    for text in tqdm(lines, "Calculating n-grams", unit="tok"):
        tokens = [UNK] * (n-1) + text.split() + [EOS]
        for *prefix, next_token in zip(*(tokens[i:] for i in range(n))):
            counts[tuple(prefix)][next_token] += 1

    return counts
