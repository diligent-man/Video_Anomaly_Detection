"""
Adopted from huggingface src code. Nothing changed !
"""
from typing import Optional, Union, Tuple

import torch

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip import (
    CLIPVisionConfig
)


from ...Encoder import Encoder
from .Embeddings import VisionEmbeddings


__all__ = ["VisionTransformer"]


class VisionTransformer(torch.nn.Module):
    is_leaf_module = True

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = VisionEmbeddings(config)
        self.pre_layrnorm = torch.nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = Encoder(config)
        self.post_layernorm = torch.nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        :param pixel_values: (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`)
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.

        :param output_attentions: (`bool`, *optional*)
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.

        :param output_hidden_states: (`bool`, *optional*)
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.

        :param interpolate_pos_encoding: (`bool`, *optional*, defaults `False`)
            Whether to interpolate the pre-trained position encodings.

        :param return_dict: (`bool`, *optional*)
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
