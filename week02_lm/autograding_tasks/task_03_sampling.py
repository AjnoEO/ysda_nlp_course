"""Task 03: temperature and nucleus sampling."""
import numpy as np
import random

def sample_with_temperature(dist: dict[str, float], temperature = 1.0, rng = None):
    """
    return next token after prefix;
    :param temperature: samples proportionally to lm probabilities ^ (1 / temperature)
        if temperature == 0, always takes most likely token. Break ties arbitrarily.
    """
    if temperature < 0.0:
        raise ValueError()
    possible_tokens = list(dist.keys())
    power = 1 if temperature == 0.0 else (1 / temperature)
    probs = np.array([dist[next_token] ** power for next_token in possible_tokens])
    # probs /= np.sum(probs)
    if temperature == 0.0:
        max_prob = np.max(probs)
        # count = np.sum(probs == max_prob)
        probs[probs != max_prob] = 0.0
        # probs[probs != 0.0] = 1.0 / count
    return random.choices(possible_tokens, weights=probs)


def nucleus_filter(dist: dict[str, float], nucleus=0.9):
    """Keep the smallest high-probability token set with cumulative mass >= nucleus."""
    if not (0.0 < nucleus <= 1.0):
        raise ValueError()
    dist_list = sorted(dist.items(), key=lambda i: (-i[1], i[0]))
    result: dict[str, float] = {dist_list[0][0]: dist_list[0][1]}
    # if nucleus == 0.0:
    #     return result
    total_p = dist_list[0][1]
    for token, p in dist_list[1:]:
        if total_p + p > nucleus: break
        result[token] = p
        total_p += p
        # if total_p >= nucleus: break
    return result


def sample_nucleus(dist: dict[str, float], nucleus=0.9, rng=None):
    """Sample one token after applying nucleus filtering."""
    dist = nucleus_filter(dist, nucleus=nucleus)
    possible_tokens = list(dist.keys())
    probs = np.array(list(dist.values()))
    # probs /= np.sum(probs)
    return random.choices(possible_tokens, weights=probs)
