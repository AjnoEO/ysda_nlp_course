"""Task 03: temperature and nucleus sampling."""


def sample_with_temperature(dist, temperature=1.0, rng=None):
    """Sample one token from a probability distribution with temperature."""
    raise NotImplementedError


def nucleus_filter(dist, nucleus=0.9):
    """Keep the smallest high-probability token set with cumulative mass >= nucleus."""
    raise NotImplementedError


def sample_nucleus(dist, nucleus=0.9, rng=None):
    """Sample one token after applying nucleus filtering."""
    raise NotImplementedError
