import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl

from electra import AllennlpDataModule
from electra import ElectraLitModule
from electra.dataset_readers import OpenWebTextDatasetReader

def get_datamodule(batch_size):
    dm = AllennlpDataModule(
        'data/openwebtext',
        OpenWebTextDatasetReader,
        'bert-base-uncased',
        batch_size=batch_size,
    )

    return dm

def get_model(L, H):
    model = ElectraLitModule(
        dm.tokenizer.tokenizer,
        hidden_size=128,
        num_gen_hidden_layers=L,
        gen_intermediate_size=H,
        num_disc_hidden_layers=L,
        disc_intermediate_size=H,
        num_attention_heads=H // 64, # Sec 5. https://arxiv.org/pdf/1908.08962.pdf
        activation='gelu',
    )


if __name__ == '__main__':
    dm = get_datamodule(batch_size=128)
    model = get_model(L=4, H=512)

    trainer = pl.Trainer(gpus=1, fast_dev_run=True)
    trainer.fit(model, dm)
