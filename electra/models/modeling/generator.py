from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import get_activation
from transformers.file_utils import ModelOutput
from transformers.models.electra.modeling_electra import (
    ElectraEmbeddings,
    ElectraEncoder,
    ElectraModel,
)

from electra.utils import get_masking_strategy


def _gumble_sample(input_tensor: torch.Tensor, temp: int = 1):
    input_tensor = input_tensor.float()

    _eps = torch.finfo(torch.float).eps
    log = lambda x: torch.log(x + _eps)

    noise = torch.zeros_like(
        input_tensor,
        dtype=torch.float,
        device=input_tensor.device
    ).uniform_(0, 1)

    gumble_noise = -log(-log(noise))

    return torch.argmax((input_tensor / temp) + gumble_noise, dim=-1)

def _get_masked_text(
        input_ids: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor] = None,
        mask_prob: float = 0.25,
        masking_strategy: str = 'dynamic_masking',
        mask_token_id: int = 103,
) -> torch.Tensor:
    if special_tokens_mask is None:
        special_tokens_mask = torch.zeros_like(input_ids)

    masking_func = get_masking_strategy(masking_strategy)
    mask = masking_func(special_tokens_mask, mask_prob)

    masked_ids = input_ids.clone().detach()
    masked_ids = masked_ids.masked_fill(mask, mask_token_id)

    return mask, masked_ids

class GeneratorOutput(ModelOutput):
    last_hidden_state: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    sampled_tokens_ids: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None

class GeneratorHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self._temperature = config.temperature
        self._pad_token_id = config.pad_token_id

        self.layer_norm = nn.LayerNorm(config.embedding_size)
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.activation = get_activation(config.hidden_act)

    def forward(
            self,
            generator_hidden_state: torch.Tensor,
            input_ids: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            output_last_hidden_state=True,
            output_loss=True,
            output_mask=True,
            output_sampled_tokens_ids=True,
    ) -> torch.Tensor:
        hidden_state = self.dense(generator_hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.layer_norm(hidden_state)

        mlm_loss = None
        if output_loss or output_sampled_tokens_ids:
            labels = input_ids.masked_fill(~mask, self._pad_token_id)

            mlm_loss = F.cross_entropy(
                hidden_state.transpose(1, 2),
                labels,
                ignore_index = self._pad_token_id
            )

        sampled_tokens_ids = None
        if output_sampled_tokens_ids:
            mask_indices = torch.nonzero(mask, as_tuple=True)

            sample_logits = hidden_state[mask_indices]
            sampled_tokens_ids = _gumble_sample(sample_logits, self._temperature)

        return GeneratorOutput(
            last_hidden_state=hidden_state if output_last_hidden_state else None,
            loss=mlm_loss,
            sampled_tokens_ids=sampled_tokens_ids,
            mask=mask if output_mask else None,
        )

class ElectraGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self._mask_prob = config.mask_prob
        self._mask_token_id = config.mask_token_id
        self._masking_strategy = config.masking_strategy

        self.backbone = ElectraModel(config)
        self.generator_head = GeneratorHead(config)

    def forward(
            self,
            input_ids,
            attention_mask,
            special_tokens_mask,
            output_hidden_state=False,
            output_mask=True,
    ):
        mask, masked_text = _get_masked_text(
            input_ids,
            special_tokens_mask,
            self._mask_prob,
            self._masking_strategy,
            self._mask_token_id,
        )

        generator_hidden_state = self.backbone(
            masked_text, attention_mask
        ).last_hidden_state

        return self.generator_head(
            generator_hidden_state,
            input_ids,
            mask,
            output_hidden_state=output_hidden_state,
            output_loss=True,
            output_mask=output_mask,
            output_sampled_tokens_ids=True
        )
