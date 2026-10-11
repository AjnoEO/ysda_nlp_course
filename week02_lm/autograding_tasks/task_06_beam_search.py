"""Task 06: beam search generation."""


def generate_beamsearch(next_dist, prefix, beam_size=5, length=10):
    """Generate top hypotheses with beam search.

    Args:
        next_dist: callable that accepts a prefix string and returns
            ``{token: probability}``.
        prefix: initial text prefix.
        beam_size: number of hypotheses to keep.
        length: number of tokens to generate.

    Returns:
        A list of ``(text, log_probability)`` pairs sorted by score descending.
    """
    raise NotImplementedError
