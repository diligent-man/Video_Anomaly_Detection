"""
Author: https://github.com/RElbers/info-nce-pytorch
Ref: https://lilianweng.github.io/posts/2021-05-31-contrastive/#infonce
     https://leimao.github.io/article/Noise-Contrastive-Estimation/
"""
from torch import Tensor
from torch.nn import Module

from ..modeling.nn.functional import info_nce


__all__ = ['InfoNCE']


class InfoNCE(Module):
    """

    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:


    Returns:
         Value of the InfoNCE Loss.
    """
    def __init__(self,
                 temperature: float = 0.1,
                 reduction: str = "mean",
                 negative_mode: str = "unpaired"
                 ) -> None:
        super(InfoNCE, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    @staticmethod
    def _check_input_dim(query: Tensor, positive_key: Tensor, negative_keys: Tensor, negative_mode: str) -> None:
        if query.dim() != 2:
            raise ValueError('<query> must have 2 dimensions.')

        if positive_key.dim() != 2:
            raise ValueError('<positive_key> must have 2 dimensions.')

        if negative_keys is not None:
            if negative_mode == 'unpaired' and negative_keys.dim() != 2:
                raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")

            if negative_mode == 'paired' and negative_keys.dim() != 3:
                raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    @staticmethod
    def _check_embed_dim(query: Tensor, positive_key: Tensor, negative_keys: Tensor) -> None:
        if query.shape[-1] != positive_key.shape[-1]:
            raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')

        if negative_keys is not None:
            if query.shape[-1] != negative_keys.shape[-1]:
                raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    @staticmethod
    def _check_num_samples(query: Tensor, positive_key: Tensor, negative_keys: Tensor, negative_mode: str) -> None:
        if len(query) != len(positive_key):
            raise ValueError('<query> and <positive_key> must must have the same number of samples.')

        if negative_keys is not None:
            if negative_mode == 'paired' and len(query) != len(negative_keys):
                raise ValueError(
                    "If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    def forward(self, query: Tensor, positive_key: Tensor, negative_keys: Tensor = None) -> Tensor:
        """
        :param query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        :param positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        :param negative_keys: (optional) Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.
        :return: computed loss
        """
        self._check_input_dim(query, positive_key, negative_keys, self.negative_mode)
        self._check_embed_dim(query, positive_key, negative_keys)
        self._check_num_samples(query, positive_key, negative_keys, self.negative_mode)
        return info_nce(query, positive_key, negative_keys, self.temperature, self.reduction, self.negative_mode)
