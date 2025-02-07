"""
Adopted from huggingface src code. Nothing changed !
"""
from typing import Optional, Tuple, Union

import torch

from transformers.models.clip import CLIPTextConfig, CLIPVisionConfig


from ..MLP import MLP
from ..Attn import CLIP_ATTENTION_CLASSES


__all__ = ["EncoderLayer"]


class EncoderLayer(torch.nn.Module):
    """
    Single encoder Layer from Transformers (2017)
    """
    def __init__(self, config: Union[CLIPTextConfig, CLIPVisionConfig]):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIP_ATTENTION_CLASSES[config._attn_implementation](config)
        self.layer_norm1 = torch.nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = torch.nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        :param: hidden_states: (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        :param: attention_mask: (`torch.FloatTensor`): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            `(config.encoder_attention_heads,)`.
        :param: output_attentions: (`bool`, *optional*)
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
