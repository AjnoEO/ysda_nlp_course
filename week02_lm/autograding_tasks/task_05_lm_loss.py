"""Task 05: masked language-model loss."""
import numpy as np
import torch
from torch.functional import F

def compute_mask(input_ix, eos_ix):
    """Return a boolean mask that keeps tokens up to and including first EOS."""
    return F.pad(torch.cumsum(input_ix == eos_ix, dim=-1)[..., :-1] < 1, pad=(1, 0, 0, 0), value=True)


def compute_lm_loss(logits, targets, eos_ix):
    """Compute cross-entropy language-model loss, ignoring tokens after EOS.
    :param model: language model that can compute next token logits given token indices
    :param input ix: int32 matrix of tokens, shape: [batch_size, length]; padded with eos_ix
    :returns: scalar loss function, mean crossentropy over non-eos tokens
    """
    # input_ix = torch.as_tensor(input_ix, dtype=torch.int64)

    # Your task: implement loss function as per formula above
    # your loss should only be computed on actual tokens, excluding padding
    # predicting actual tokens and first EOS do count. Subsequent EOS-es don't
    # you may or may not want to use the compute_mask function from above.

    logits = torch.as_tensor(logits, dtype=torch.float32)
    targets = torch.as_tensor(targets, dtype=torch.int64)

    target_pred_proba = torch.gather(torch.log_softmax(logits, dim=2), dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    mask = compute_mask(targets, eos_ix)

    batch_size = len(targets)

    return -1 / batch_size * torch.sum(mask * target_pred_proba).item()
