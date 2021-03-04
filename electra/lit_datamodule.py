from typing import Callable, Union, Optional, List
from itertools import chain
import os
import pathlib
import random

from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.vocabulary import Vocabulary
import pytorch_lightning as pl

from electra import dataset_readers

class AllennlpDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_path: str,
            dataset_reader_cls: Union[str, Callable],
            model_name: str,
            batch_size: int,
            vocab_dir: Optional[str] = None,
            max_length: Optional[int] = None,
            train_val_test_split: List[int] = [0.7, 0.15, 0.15]
    ):
        super().__init__()

        root_path = pathlib.Path(__file__).parents[1]
        
        self._data_path = root_path / (data_path)
        self.batch_size = batch_size

        if isinstance(dataset_reader_cls, str):
            dataset_reader_cls = getattr(dataset_readers, dataset_reader_cls)

        self.reader = dataset_reader_cls(model_name, max_length)
        self.tokenizer = self.reader.get_tokenizer()
        self._vocab_dir = root_path / vocab_dir
        self._split_size = train_val_test_split

    def _setup_raw_files(self):
        data_dir = self._data_path
        split_dir = {data_dir / 'train', data_dir / 'valid', data_dir / 'test'}
        target_files = set(data_dir.iterdir()) - split_dir

        if target_files:
            (data_dir / 'train').mkdir(exist_ok=True)
            (data_dir / 'valid').mkdir(exist_ok=True)
            (data_dir / 'test').mkdir(exist_ok=True)

            total_size = len(target_files)
            train_size = int(total_size * self._split_size[0])
            valid_size = int(total_size * self._split_size[1])

            for train_file in random.sample(target_files, train_size):
                train_file.rename(data_dir / 'train' / train_file.name)

            target_files = set(data_dir.iterdir()) - split_dir

            for valid_file in random.sample(target_files, valid_size):
                valid_file.rename(data_dir / 'valid' / valid_file.name)

            target_files = set(data_dir.iterdir()) - split_dir

            for test_file in target_files:
                test_file.rename(data_dir / 'test' / test_file.name)


    def setup(self, stage=None):
        self._setup_raw_files()
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

        if self._vocab_dir is None or not os.path.exists(self._vocab_dir):
            loaders_instances = chain(
                self._train_dataloader.iter_instances(),
                self._val_dataloader.iter_instances()
            )

            self.vocab = Vocabulary.from_instances(
                loaders_instances,
                max_vocab_size=self.tokenizer.vocab_size,
                padding_token=self.reader._pad_token,
                oov_token=self.reader._unk_token,
            )
            if self._vocab_dir is not None:
                self.vocab.save_to_files(self._vocab_dir)

        else:
            self.vocab = Vocabulary.from_files(
                self._vocab_dir,
                padding_token=self.reader._pad_token,
                oov_token=self.reader._unk_token,
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
