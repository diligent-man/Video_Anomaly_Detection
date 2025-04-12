import torch
from torch import Tensor
from torch.nn.functional import normalize, cross_entropy

__all__ = ["info_nce"]


def info_nce(query: Tensor,
             positive_key: Tensor,
             negative_keys: Tensor = None,
             temperature: float = 0.1,
             reduction: str = "mean",
             negative_mode: str = "unpaired"
             ) -> Tensor:
    """
    :param query: (N, D) Tensor with query samples (e.g. embeddings of the input).
    :param positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
    :param negative_keys: (optional) Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.
    :param temperature: Logits are divided by temperature before calculating the cross entropy.
    :param reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
    :param negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.
    :return: computed info_nce loss
    """
    query, positive_key, negative_keys = [
        None if x is None else normalize(x, dim=-1) for x in (query, positive_key, negative_keys)
    ]

    if negative_keys is not None:
        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=-1, keepdim=True)

        if negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ negative_keys.transpose(-2, -1)
            negative_logits = negative_logits.squeeze(1)
        else:
            # Cosine between all query-negative combinations
            negative_logits = query @ negative_keys.transpose(-2, -1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=-1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.
        logits = query @ positive_key.T  # Cosine between all combinations
        labels = torch.arange(len(query), device=query.device)  # Positive keys are the entries on the diagonal
    return cross_entropy(logits / temperature, labels, reduction=reduction)
