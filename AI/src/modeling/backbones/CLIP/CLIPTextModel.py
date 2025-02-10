"""
Adopted from huggingface src code. Nothing changed !
"""
from typing import Optional, Union, Tuple


import torch

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip import CLIPPreTrainedModel, CLIPTextConfig


from .Transformer import TextTransformer

__all__ = ["CLIPTextModel", "clip_text"]


class CLIPTextModel(CLIPPreTrainedModel):
    config_class = CLIPTextConfig

    _no_split_modules = ["CLIPTextEmbeddings", "CLIPEncoderLayer"]

    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.text_model = TextTransformer(config)
        self.post_init()  # Initialize weights and apply final processing

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        :param input_ids: (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)

        :param attention_mask: (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

        :param position_ids: (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*)
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)

        :param output_attentions: (`bool`, *optional*)
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.

        :param output_hidden_states: (`bool`, *optional*)
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.

        :param return_dict: (`bool`, *optional*)
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


def clip_text(weights: str) -> torch.nn.Module:
    return CLIPTextModel.from_pretrained(weights, use_safetensors=True)
