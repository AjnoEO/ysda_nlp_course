"""Task 05: masked language-model loss."""


def compute_mask(input_ix, eos_ix):
    """Return a boolean mask that keeps tokens up to and including first EOS."""
    raise NotImplementedError


def compute_lm_loss(logits, targets, eos_ix):
    """Compute cross-entropy language-model loss, ignoring tokens after EOS."""
    raise NotImplementedError
