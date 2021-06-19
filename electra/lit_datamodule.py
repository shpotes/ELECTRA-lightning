from typing import Any, Dict, List, Optional

import datasets
import pytorch_lightning as pl
import transformers
from torch.utils.data import DataLoader

from electra import dataset as electra_dataset

class HuggingDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_path: str,
            model_name: str,
            batch_size: int,
            data_name: Optional[str] = None,
            max_length: Optional[int] = None,
            train_val_test_split: List[int] = [0.7, 0.15, 0.15],
            from_disk: bool = False,
            data_kwargs: Dict[str, Any] = {},
            **data_loader_kwargs
    ):
        super().__init__()

        self._from_disk = from_disk
        self._data_args = {
            'path': data_path,
            'name': data_name,
            **data_kwargs
        }

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_length if max_length else self.tokenizer.model_max_length

        self.batch_size = batch_size
        self._data_kwargs = data_loader_kwargs

        self._split_size = train_val_test_split


    def prepare_data(self):
        if not self._from_disk:
            datasets.load_dataset(**self._data_args)

    def setup(self, _):
        if self._from_disk:
            data = datasets.load_from_disk(self._data_args['path'])
        else:
            data = datasets.load_dataset(**self._data_args)

        train_data = data['train']
        print(train_data)

        if all(split in data for split in ['test', 'validation']):
            test_data = data['test']
            val_data = data['validation']
        else:
            _, val_size, test_size = self._split_size
            test_size = int(len(train_data) * test_size)
            val_size = int(len(train_data) * val_size)

            train_data = train_data.shuffle()

            test_data = train_data[:test_size]
            val_data = train_data[test_size:val_size+test_size]
            train_data = train_data[val_size+test_size:]

        print('create dataset')

        self.train_dataset = electra_dataset.MLMDataset(
            train_data, self.tokenizer, **self._data_kwargs
        )

        self.val_dataset = electra_dataset.MLMDataset(
            val_data, self.tokenizer, **self._data_kwargs
        )
        self.test_dataset = electra_dataset.MLMDataset(
            test_data, self.tokenizer, **self._data_kwargs
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=8
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=8
        )


    def save_to_disk(self, path: str):
        template = datasets.DatasetDict()
        template['train'] = self.train_dataset._dataset
        template['validation'] = self.val_dataset._dataset
        template['test'] = self.test_dataset._dataset

        template.save_to_disk(path)

    @staticmethod
    def load_from_disk(
            path: str,
            model_name: str,
            batch_size: int,
    ):
        return HuggingDataModule(
            path,
            model_name,
            batch_size,
            from_disk=True,
            prepare_data=False,
        )
