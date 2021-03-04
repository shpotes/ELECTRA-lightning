from typing import Callable

from omegaconf import DictConfig
import pytorch_lightning as pl

from electra import AllennlpDataModule
from electra import dataset_readers
from electra import ElectraLitModule

def create_datamodule(conf: DictConfig) -> pl.LightningDataModule:
    target = AllennlpDataModule(**conf.datamodule)
    return target

def create_model(datamodule, conf) -> pl.LightningModule:
    return ElectraLitModule(
        datamodule.tokenizer,
        **conf.architecture,
        **conf.training
    )

def create_trainer(conf) -> pl.Trainer:
    logger = pl.logging.WandbLogger(conf.name)

    return pl.Trainer(
        logger=logger
        **conf.trainer
    )
