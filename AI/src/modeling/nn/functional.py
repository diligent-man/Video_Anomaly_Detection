from typing import List

import torch
from torch import Tensor


__all__ = ["info_nce"]


def info_nce(query,
             positive_key,
             negative_keys=None,
             temperature=0.1,
             reduction="mean",
             negative_mode="unpaired"
             ):
    def _normalize(*xs) -> List[Tensor]:
        """
        :param xs: multiple x to normalize
        :return:
        """
        return [None if x is None else torch.nn.functional.normalize(x, dim=-1) for x in xs]

    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')

    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')

    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")

        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')

    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')

    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    query, positive_key, negative_keys = _normalize(query, positive_key, negative_keys)

    if negative_keys is not None:
        # Explicit negative keys
        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ negative_keys.T

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ negative_keys.T
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.
        # Cosine between all combinations
        logits = query @ positive_key.T

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)
    return torch.nn.functional.cross_entropy(logits / temperature, labels, reduction=reduction)
