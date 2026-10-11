"""Task 04: corpus perplexity."""
from .task_01_ngram_counts import *
from .task_02_ngram_lm import *

import numpy as np

def perplexity(lm: NGramLanguageModel, lines: list[str], min_logprob=np.log(10 ** -50.)):
    """
    :param lines: a list of strings with space-separated tokens
    :param min_logprob: if log(P(w | ...)) is smaller than min_logprop, set it equal to min_logrob
    :returns: corpora-level perplexity - a single scalar number from the formula above

    Note: scoring a line "w_1 w_2 ... w_L" means scoring L + 1 events:
        P(w_1 | ""), P(w_2 | "w_1"), ..., P(w_L | "w_1 ... w_{L-1}"), P(EOS | "w_1 ... w_L")
    The prefix of the very first token is the empty string "", i.e. an empty prefix and not
    a string of UNKs: the model pads it to (n - 1) UNK tokens on its own inside
    get_possible_next_tokens, so do not add any padding here. Likewise, w_1 is simply the
    first real token of the line. The trailing EOS is scored like any other token and is
    counted in N too, so a line with L tokens adds L + 1 to N.

    PLEASE USE lm.get_next_token_prob and NOT lm.get_possible_next_tokens
    """
    log_sum = 0.0
    N = 0
    for text in lines:
        tokens = text.split() + [EOS]
        N += len(tokens)
        for i in range(len(tokens)):
            prob = lm.get_next_token_prob(" ".join(tokens[:i]), tokens[i])
            log_sum += max(np.log(prob), min_logprob)
    log_sum *= (-1 / N)

    return np.exp(log_sum)
