from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from electra.models import TransformerEncoder

class GeneratorHead(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            tokenizer: transformers.PreTrainedTokenizerFast,
            mask_prob: float = 0.25,
            temperature: int = 1
    ):
        super().__init__()

        self.pad_token = tokenizer.pad_token_id
        self.cls_token = tokenizer.cls_token_id
        self.sep_token = tokenizer.sep_token_id
        self.mask_token = tokenizer.mask_token_id

        self.mask_prob = mask_prob

        self.linear = nn.Linear(hidden_size, tokenizer.vocab_size)

        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0.01)

        self.temperature = temperature

    def _gumble_sample(self, input_tensor):
        input_tensor = input_tensor.float()

        _eps = torch.finfo(torch.float).eps
        log = lambda x: torch.log(x + _eps)

        noise = torch.zeros_like(
            input_tensor,
            dtype=torch.float,
            device=input_tensor.device
        ).uniform_(0, 1)
        gumble_noise = -log(-log(noise))

        return torch.argmax((input_tensor / self.temperature) + gumble_noise, dim=-1)

    def get_masked_text(self, input_text):
        """
        Generate a masked text returns both the masked text and the
        corresponding mask

        TODO: there is probably a better place for this method
        """
        banned_tokens = torch.tensor([
            [[self.pad_token]],
            [[self.cls_token]],
            [[self.sep_token]],
        ], device=input_text.device)

        banned_tokens = torch.sum(
            input_text.unsqueeze(0) == banned_tokens, dim=0
        ).bool()

        # TODO: improve modularity, create a separete function dynamic masking
        # `dynamic_masking(banned_tokens, mask_prob) -> mask`

        prob_matrix = torch.full(input_text.shape, self.mask_prob, device=input_text.device)
        prob_matrix.masked_fill_(banned_tokens, value=0)

        mask = torch.bernoulli(prob_matrix).bool()

        masked_text = input_text.clone().detach()
        masked_text = masked_text.masked_fill(mask, self.mask_token)

        return mask, masked_text

    def forward(
            self,
            hidden_states: torch.Tensor,
            input_text: torch.Tensor,
            mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MLM loss
        """

        labels = input_text.masked_fill(~mask, self.pad_token)
        logits = self.linear(hidden_states)

        mlm_loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            ignore_index = self.pad_token
        )

        mask_indices = torch.nonzero(mask, as_tuple=True)
        sample_logits = logits[mask_indices]
        sampled_tokens_id = self._gumble_sample(sample_logits)

        output_dict = {
            'logits': logits,
            'loss': mlm_loss,
            'sampled_tokens_id': sampled_tokens_id,
            'mask': mask,
        }

        return output_dict

class DiscriminatorHead(nn.Module):
    def __init__(self, hidden_size, pad_token=0):
        super().__init__()

        self.linear = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0.01)

        self.pad_token = pad_token

    def forward(
            self,
            hidden_state: torch.Tensor,
            input_text: torch.Tensor,
            modified_text: torch.Tensor,
    ) -> torch.Tensor:
        labels = (input_text != modified_text).float().detach()
        logits = self.linear(hidden_state)

        logits = logits.reshape_as(labels)

        non_padded_indices = torch.nonzero(
            input_text != self.pad_token,
            as_tuple=True
        )

        disc_loss = F.binary_cross_entropy_with_logits(
            logits[non_padded_indices],
            labels[non_padded_indices]
        )

        return {'loss': disc_loss, 'logits': logits}

class ElectraGenerator(nn.Module):
    def __init__(
            self,
            embeddings: nn.Module,
            tokenizer: transformers.PreTrainedTokenizerFast,
            num_hidden_layers: int,
            hidden_size: int,
            intermediate_size: int,
            num_attention_heads: int = 8,
            attention_dropout: int = 0.1,
            hidden_dropout: float = 0.1,
            activation: Union[str, nn.Module] = 'relu',
            mask_prob: float = 0.25,
            temperature: int = 1,
    ):
        super().__init__()

        self.embeddings = embeddings
        self.encoder = TransformerEncoder(
            num_hidden_layers,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            attention_dropout,
            hidden_dropout,
            activation,
        )
        self.generator_head = GeneratorHead(
            hidden_size,
            tokenizer,
            mask_prob,
            temperature
        )

    def forward(self, input_text, attention_mask):
        mask, masked_text = self.generator_head.get_masked_text(input_text)
        embedded_text = self.embeddings(masked_text)
        encoded_output = self.encoder(embedded_text, attention_mask)

        generated_output = self.generator_head(
            encoded_output['last_hidden_state'],
            input_text, mask
        )

        return generated_output


class ElectraDiscriminator(nn.Module):
    def __init__(
            self,
            embeddings: nn.Module,
            tokenizer: transformers.PreTrainedTokenizerFast,
            num_hidden_layers: int,
            hidden_size: int,
            intermediate_size: int,
            num_attention_heads: int = 8,
            attention_dropout: int = 0.1,
            hidden_dropout: float = 0.1,
            activation: Union[str, nn.Module] = 'relu',
    ):
        super().__init__()

        self.embeddings = embeddings
        self.encoder = TransformerEncoder(
            num_hidden_layers,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            attention_dropout,
            hidden_dropout,
            activation,
        )
        self.discriminator_head = DiscriminatorHead(
            hidden_size,
            tokenizer.pad_token_id,
        )

    def forward(self, input_text, modified_text, attention_mask):
        embedded_text = self.embeddings(modified_text)
        encoded_output = self.encoder(embedded_text, attention_mask)

        discriminator_output = self.discriminator_head(
            encoded_output['last_hidden_state'],
            input_text, modified_text
        )

        return discriminator_output
