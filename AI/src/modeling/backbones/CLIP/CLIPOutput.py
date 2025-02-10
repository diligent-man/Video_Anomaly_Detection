"""
Adopted from huggingface src code. Nothing changed !
"""
from typing import Optional, Tuple, Any
from dataclasses import dataclass

import torch

from transformers.utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPooling


@dataclass
class CLIPOutput(ModelOutput):
    """
    :param: loss: (`torch.FloatTensor` of shape `(1, )`, *optional*, returned when `return_loss` is `True`)
        Contrastive loss for image-text similarity.
    :param: logits_per_image: (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`)
        The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
        similarity scores.
    :param: logits_per_text: (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`)
        The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
        similarity scores.
    :param: text_embeds: (`torch.FloatTensor` of shape `(batch_size, output_dim`)
        The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPTextModel`].
    :param: image_embeds: (`torch.FloatTensor` of shape `(batch_size, output_dim`)
        The image embeddings obtained by applying the projection layer to the pooled output of [`CLIPVisionModel`].
    :param: text_model_output: (`BaseModelOutputWithPooling`)
        The output of the [`CLIPTextModel`].
    :param: vision_model_output: (`BaseModelOutputWithPooling`)
        The output of the [`CLIPVisionModel`].
    """
    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.Tensor = None
    logits_per_text: torch.Tensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any, ...]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
