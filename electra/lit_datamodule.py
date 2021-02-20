from itertools import chain
import pathlib

from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.vocabulary import Vocabulary
import pytorch_lightning as pl

class AllennlpDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            data_path,
            dataset_reader_cls,
            model_name,
            batch_size,
            max_length=512,
    ):
        super().__init__()
        self._data_path = pathlib.Path(data_path)
        self.batch_size = batch_size 

        self.tokenizer = PretrainedTransformerTokenizer(model_name)
        self.indexer = {
            'tokens': PretrainedTransformerIndexer(
                model_name, 
                max_length=max_length)
        }

        self.reader = dataset_reader_cls(
            self.tokenizer,
            self.indexer,
            max_sequence_length=512
        )

    def setup(self, stage=None):
        for reader_input in self._data_path.iterdir():
            if stage == 'fit' or stage is None:
                if 'train' in str(reader_input):
                    _train_reader_input = reader_input
                    self._train_dataloader = MultiProcessDataLoader(
                        self.reader,
                        _train_reader_input,
                        batch_size = self.batch_size,
                        max_instances_in_memory=10 * self.batch_size,
                        shuffle=True,
                    )
                if 'valid' in str(reader_input):
                    _val_reader_input = reader_input
                    self._val_dataloader = MultiProcessDataLoader(
                        self.reader,
                        _val_reader_input,
                        batch_size = self.batch_size,
                        shuffle=False,
                    )

            if stage == 'test' or stage is None:
                if 'test' in str(reader_input):
                    _test_reader_input = reader_input
                    self._test_dataloader = MultiProcessDataLoader(
                        self.reader,
                        _test_reader_input,
                        batch_size=self.batch_size,
                        shuffle=False,
                    )

        self.vocab = Vocabulary.from_instances(
            chain(
                self._train_dataloader.iter_instances(),
                self._val_dataloader.iter_instances()
            )
        )

        if stage == 'fit' or stage is None:
            self._train_dataloader.index_with(self.vocab)
            self._val_dataloader.index_with(self.vocab)

        if stage == 'test' or stage is None:
            self._test_dataloader.index_with(self.vocab)

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader
