from typing import Optional, Union

from allennlp.modules.transformer import (
    TransformerLayer,
    TransformerEmbeddings
)
from allennlp.modules.util import replicate_layers

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(
            self,
            num_hidden_layers: int,
            hidden_size: int,
            intermediate_size: int,
            num_attention_heads: int = 8,
            attention_dropout: int = 0.1,
            hidden_dropout: float = 0.1,
            activation: Union[str, nn.Module] = 'relu',
    ):
        super().__init__()

        layer = TransformerLayer(
            hidden_size,
            intermediate_size,
            num_attention_heads,
            attention_dropout,
            hidden_dropout,
            activation,
        )

        self.layers = replicate_layers(layer, num_hidden_layers)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
    ) -> torch.Tensor:

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layers):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        output_dict = {
            'last_hidden_state': hidden_states,
        }

        if output_attentions:
            output_dict['all_attentions'] = all_attentions

        if output_hidden_states:
            output_dict['all_hidden_states'] = all_hidden_states

        return output_dict
