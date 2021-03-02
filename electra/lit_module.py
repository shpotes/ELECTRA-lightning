from typing import Union
from allennlp.modules.transformer import TransformerEmbeddings
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as M
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch_optimizer as optim
from transformers import PreTrainedTokenizerBase
from electra.models import ElectraDiscriminator, ElectraGenerator

class ElectraLitModule(pl.LightningModule):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            hidden_size: int,
            num_gen_hidden_layers: int,
            gen_intermediate_size: int,
            num_disc_hidden_layers: int,
            disc_intermediate_size: int,
            num_attention_heads: int = 8,
            attention_dropout: float = 0.1,
            hidden_dropout: float = 0.1,
            activation: Union[str, nn.Module] = 'relu',
            embeddings_dropout=0.1,
            mask_prob: float = 0.25,
            temperature: int = 1,
            learning_rate: float = 3e-4,
            gen_weight: float = 1.,
            disc_weight: float = 50.,
    ):
        super().__init__()

        self.embeddings = TransformerEmbeddings(
            vocab_size=tokenizer.vocab_size,
            embedding_size=hidden_size,
            pad_token_id=tokenizer.pad_token_id,
            max_position_embeddings=tokenizer.model_max_length,
            type_vocab_size=0,
            dropout=embeddings_dropout,
        )

        self.generator = ElectraGenerator(
            self.embeddings,
            tokenizer,
            num_gen_hidden_layers,
            hidden_size,
            gen_intermediate_size,
            num_attention_heads,
            attention_dropout,
            hidden_dropout,
            activation,
            mask_prob,
            temperature,
        )

        self.discriminator = ElectraDiscriminator(
            self.embeddings,
            tokenizer,
            num_disc_hidden_layers,
            hidden_size,
            disc_intermediate_size,
            num_attention_heads,
            attention_dropout,
            hidden_dropout,
            activation,
        )

        self.learning_rate = learning_rate
        self.gen_weight = gen_weight
        self.disc_weight = disc_weight

    def forward(self, input_text, attention_mask):
        gen_output = self.generator(input_text, attention_mask)

        mask_idx = torch.nonzero(gen_output['mask'], as_tuple=True)

        fake_text = input_text.clone().detach()
        fake_text[mask_idx] = gen_output['sampled_tokens_id'].detach()

        disc_output = self.discriminator(
            input_text,
            fake_text,
            attention_mask
        )

        output_dict = {
            'mlm_loss': gen_output['loss'],
            'disc_loss': disc_output['loss'],
            'gen_logits': gen_output['logits'],
            'disc_logits': disc_output['logits'],
            'mask': gen_output['mask'],
            'fake_text': fake_text,
        }

        return output_dict

    def _common_step(self, batch_tokens):
        input_text = batch_tokens['token_ids']
        attention_mask = batch_tokens['mask']

        output_dict = self(input_text, attention_mask)

        loss = self.gen_weight * output_dict['mlm_loss'] + \
            self.disc_weight * output_dict['disc_loss']

        mask = output_dict['mask']

        disc_label = (output_dict['fake_text'] != input_text).float().detach()

        gen_pred = torch.argmax(output_dict['gen_logits'], dim=-1)
        disc_pred = torch.sign(output_dict['disc_logits']) + 1
        disc_pred = torch.round(disc_pred * 0.5)

        gen_acc = M.accuracy(input_text[mask], gen_pred[mask])
        disc_acc = M.accuracy(disc_label[mask], disc_pred[mask])

        return loss, gen_acc, disc_acc

    def training_step(self, batch, _):
        loss, gen_acc, disc_acc = self._common_step(batch['source']['tokens'])

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_gen_acc', gen_acc)
        self.log('train_disc_acc', disc_acc)

        return loss

    def validation_step(self, batch, _):
        loss, gen_acc, disc_acc = self._common_step(batch['source']['tokens'])

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_gen_acc', gen_acc)
        self.log('val_disc_acc', disc_acc)


    def configure_optimizers(self):
        # Adam optimizer without bias correction
        # Have lower learning rates for layers closer to the input.
        # polynomial decay
        optimizer = optim.Lamb(self.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 100)

        return [optimizer], [scheduler]
