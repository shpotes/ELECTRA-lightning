from typing import Dict, Union
from omegaconf import OmegaConf
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as M
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.optim import lr_scheduler
from torch import optim
# import torch_optimizer as optim
from transformers import PreTrainedTokenizerBase
from transformers.models.electra import ElectraConfig
from electra.models import (
    ElectraGenerator,
    ElectraDiscriminator
)

class ElectraLitModule(pl.LightningModule):
    def __init__(
            self,
            config: ElectraConfig,
            hparams: Union[Dict, OmegaConf]
    ):
        super().__init__()

        # self.save_hyperparameters(config.to_dict(), hparams)

        self.generator = ElectraGenerator(config)
        self.discriminator = ElectraDiscriminator(config)

        self.discriminator.backbone.embeddings = self.generator.backbone.embeddings

        self.learning_rate = hparams['learning_rate']
        self.gen_weight = hparams['gen_weight']
        self.disc_weight = hparams['disc_weight']

    def forward(self, input_ids, attention_mask, special_tokens_mask):
        gen_output = self.generator(input_ids, attention_mask, special_tokens_mask)

        mask_idx = torch.nonzero(gen_output['mask'], as_tuple=True)

        fake_input_ids = input_ids.clone().detach()
        fake_input_ids[mask_idx] = gen_output['sampled_tokens_ids'].clone().detach()

        disc_output = self.discriminator(input_ids, fake_input_ids, attention_mask)

        output_dict = {
            'mlm_loss': gen_output['loss'],
            'disc_loss': disc_output['loss'],
            # 'gen_logits': gen_output['last_hidden_state'],
            'disc_logits': disc_output['last_hidden_state'],
            'mask': gen_output['mask'],
            'fake_input_ids': fake_input_ids,
        }

        return output_dict

    def _common_step(self, batch):
        input_ids, attention_mask , special_tokens_mask = batch.values()
        output_dict = self(input_ids, attention_mask, special_tokens_mask)

        loss = self.gen_weight * output_dict['mlm_loss'] + \
            self.disc_weight * output_dict['disc_loss']

        mask = output_dict['mask']

        disc_label = (output_dict['fake_input_ids'] != input_ids).int().detach()

        gen_pred = output_dict['fake_input_ids']
        disc_pred = torch.sign(output_dict['disc_logits']) + 1
        disc_pred = torch.round(disc_pred * 0.5).int()

        gen_acc = M.accuracy(input_ids[mask], gen_pred[mask])
        disc_acc = M.accuracy(disc_label[mask], disc_pred[mask])

        return loss, gen_acc, disc_acc

    def training_step(self, batch, _):
        loss, gen_acc, disc_acc = self._common_step(batch)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_gen_acc', gen_acc)
        self.log('train_disc_acc', disc_acc)

        return loss

    def validation_step(self, batch, _):
        loss, gen_acc, disc_acc = self._common_step(batch)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_gen_acc', gen_acc)
        self.log('val_disc_acc', disc_acc)


    def configure_optimizers(self):
        # Adam optimizer without bias correction
        # Have lower learning rates for layers closer to the input.
        # polynomial decay
        # optimizer = optim.Lamb(self.parameters(), lr=self.learning_rate)
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)

        return [optimizer], [scheduler]
