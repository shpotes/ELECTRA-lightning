from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import get_activation
from transformers.file_utils import ModelOutput
from transformers.models.electra import ElectraModel

class DiscriminatorOutput(ModelOutput):
    logits: torch.Tensor = None
    hidden_state: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None

class DiscriminatorHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = get_activation(config.hidden_act)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)

    def forward(
            self,
            discriminator_hidden_states: torch.Tensor,
            input_ids: torch.Tensor,
            fake_input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            output_loss=True,
            output_hidden_state=True,
            output_logits=True,
    ):
        labels = (input_ids != fake_input_ids).float().detach()

        hidden_state = self.dense(discriminator_hidden_states)
        hidden_state = self.activation(hidden_state)
        logits = self.dense_prediction(hidden_state).squeeze(-1)

        disc_loss = None
        if output_loss:
            attended_indices = torch.nonzero(
                attention_mask,
                as_tuple=True
            )

            disc_loss = F.binary_cross_entropy_with_logits(
                logits[attended_indices],
                labels[attended_indices]
            )

        return DiscriminatorOutput(
            last_hidden_state=logits if output_logits else None,
            hidden_state=hidden_state if output_hidden_state else None,
            loss=disc_loss,
        )

class ElectraDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.backbone = ElectraModel(config)
        self.discriminator_head = DiscriminatorHead(config)

    def forward(self, input_text, fake_input_text, attention_mask, **kwargs):
        discriminator_hidden_state = self.backbone(
            fake_input_text,
            attention_mask
        ).last_hidden_state

        return self.discriminator_head(
            discriminator_hidden_state,
            input_text,
            fake_input_text,
            attention_mask,
            **kwargs
        )
