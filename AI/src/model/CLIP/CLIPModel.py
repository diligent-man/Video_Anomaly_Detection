"""
Adopted from huggingface src code. Nothing changed !
"""

from typing import Optional, Union, Tuple


import torch

from transformers.models.clip import (
    CLIPPreTrainedModel,
    CLIPConfig, CLIPTextConfig, CLIPVisionConfig,
)


from .CLIPOutput import CLIPOutput
from .CLIPTextModel import CLIPTextModel
from .CLIPVisionModel import CLIPVisionModel
from .utils import clip_loss, _get_vector_norm

__all__ = ["CLIPModel"]


class CLIPModel(CLIPPreTrainedModel):
    config_class = CLIPConfig
    _no_split_modules = ["CLIPTextEmbeddings", "CLIPEncoderLayer", "CLIPVisionEmbeddings"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        if not isinstance(config.text_config, CLIPTextConfig):
            raise TypeError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise TypeError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config: CLIPTextConfig = config.text_config
        vision_config: CLIPVisionConfig = config.vision_config

        self.projection_dim = config.projection_dim
        self.logit_scale = torch.nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        self.text_model = CLIPTextModel._from_config(text_config).text_model
        self.text_embed_dim = text_config.hidden_size
        self.text_projection = torch.nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)

        self.vision_model = CLIPVisionModel._from_config(vision_config).vision_model
        self.vision_embed_dim = vision_config.hidden_size
        self.visual_projection = torch.nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)

        self.post_init()  # Initialize weights and apply final processing

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        """
        :param: input_ids: (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)

        :param: attention_mask: (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

        :param: position_ids: (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*)
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)

        :param: output_attentions: (`bool`, *optional*)
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.

        :param: output_hidden_states: (`bool`, *optional*)
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.

        :param: return_dict: (`bool`, *optional*)
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        :return: text_features (torch.FloatTensor of shape (batch_size, output_dim)
        """

        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)
        return text_features

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        """
        :param: pixel_values: (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`)
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.

        :param: output_attentions: (`bool`, *optional*)
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.

        :param: output_hidden_states: (`bool`, *optional*)
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.

        :param: interpolate_pos_encoding: (`bool`, *optional*, defaults `False`)
            Whether to interpolate the pre-trained position encodings.

        :param: return_dict: (`bool`, *optional*)
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        :return: image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained
            by applying the projection layer to the pooled output of [`CLIPVisionModel`].

        """
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        # project pooled_output
        image_features = self.visual_projection(vision_outputs[1])
        return image_features

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPOutput]:
        """
        :param: input_ids: (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)

        :param: attention_mask: (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

        :param: position_ids: (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*)
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)

        :param: pixel_values: (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`)
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.

        :param: return_loss: (`bool`, *optional*)
            Whether or not to return the contrastive loss.

        :param: output_attentions: (`bool`, *optional*)
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.

        :param: output_hidden_states: (`bool`, *optional*)
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.

        :param: interpolate_pos_encoding: (`bool`, *optional*, defaults `False`)
            Whether to interpolate the pre-trained position encodings.

        :param: return_dict: (`bool`, *optional*)
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        :return: CLIPOutput() or tuple(torch.FloatTensor)
        """
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values,
            output_attentions,
            output_hidden_states,
            interpolate_pos_encoding,
            return_dict,
        )

        text_outputs = self.text_model(
            input_ids,
            attention_mask,
            position_ids,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / _get_vector_norm(image_embeds)
        text_embeds = text_embeds / _get_vector_norm(text_embeds)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device)) * logit_scale.to(text_embeds.device)
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
