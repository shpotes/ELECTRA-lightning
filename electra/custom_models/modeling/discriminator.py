from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import get_activation
from transformers.file_utils import ModelOutput
from transformers.models.electra import ElectraModel

class DiscriminatorOutput(ModelOuput):
    logits: torch.Tensor = None
    hidden_states: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None

class DiscriminatorHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self._pad_token_id = config.pad_token_id

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = get_activation(config.hidden_act)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)

    def forward(
            self,
            discriminator_hidden_states: torch.Tensor,
            input_ids: torch.Tensor,
            fake_input_ids: torch.Tensor,
            output_loss=True,
            output_hidden_states=False,
            output_logits=True,
    ):
        labels = (input_ids != fake_input_ids).float().detach()

        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = self.activation(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze(-1)

        disc_loss = None
        if output_loss:
            non_padded_indices = torch.nonzeros(
                input_ids != self._path_token_id,
                as_tuple=True
            )

            disc_loss = F.binary_cross_entropy_with_logits(
                logits[non_padded_indices],
                labels[non_padded_indices]
            )

        return DiscriminatorOutput(
            logits=logits if output_logits else None,
            hidden_states=hidden_states if output_hidden_states else None,
            loss=disc_loss,
        )

class ElectraDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.backbone = ElectraModel(config)
        self.discriminator_head = DiscriminatorHead(config)

    def forward(self, input_text, fake_input_text, attention_mask):
        discriminator_hidden_state = self.backbone(
            fake_input_text,
            attention_mask
        ).last_hidden_state

        return self.discriminator_head(
            discriminator_hidden_state,
            input_text,
            fake_input_text,
        )
