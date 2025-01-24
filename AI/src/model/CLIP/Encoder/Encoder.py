"""
Adopted from huggingface src code. Nothing changed !
"""
from typing import Optional, Tuple, Union


import torch
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.clip import CLIPTextConfig, CLIPVisionConfig


from .EncoderLayer import EncoderLayer


__all__ = ["Encoder"]


class Encoder(torch.nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers.
    Each layer is a [`CLIPEncoderLayer`].
    """
    def __init__(self, config: Union[CLIPTextConfig, CLIPVisionConfig]):
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        :param inputs_embeds: (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        :param attention_mask: (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        :param causal_attention_mask: (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
            Causal mask for the text model. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        :param output_attentions: (`bool`, *optional*)
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        :param output_hidden_states: (`bool`, *optional*)
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more detail.
        :param return_dict: (`bool`, *optional*)
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )
