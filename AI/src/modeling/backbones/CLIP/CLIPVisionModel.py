"""
Adopted from huggingface src code. Nothing changed !
"""
from typing import Optional, Union, Tuple


import torch

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip import (
    CLIPPreTrainedModel,
    CLIPVisionConfig
)


from ....utils import load_weights
from .Transformer import VisionTransformer


__all__ = ["CLIPVisionModel", "clip_vision"]


class CLIPVisionModel(CLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.vision_model = VisionTransformer(config)
        self.post_init()  # Initialize weights and apply final processing

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values,
            output_attentions,
            output_hidden_states,
            return_dict,
            interpolate_pos_encoding,
        )


def clip_vision(weights: str) -> torch.nn.Module:
    return CLIPVisionModel.from_pretrained(load_weights(weights, "hugging_face"))
