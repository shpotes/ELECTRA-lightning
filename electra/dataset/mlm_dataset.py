import itertools
from typing import Any, Dict, Union, List

import datasets
from transformers import (
        PreTrainedTokenizerBase,
        AutoTokenizer
    )
from torch.utils.data import Dataset as TorchDataset

class MLMDataset(TorchDataset):
    def __init__(
            self,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizerBase,
            num_proc: int = 1,
            data_batch_size: int = 1000,
            text_column_name: str = 'text',
            prepare_data: bool = True,
            padding: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_len = self.tokenizer.model_max_length

        self._batch_size = data_batch_size
        self._num_proc = num_proc

        self._text_column = text_column_name

        self.padding = padding

        if prepare_data:
            self._dataset = self.prepare_dataset(dataset)
            self._dataset.set_format('torch')
        else:
            self._dataset = dataset


    def __getitem__(self, idx):
        return self._dataset[idx]

    def __len__(self):
        return len(self._dataset)

    def __str__(self):
        return self._dataset.__str__()

    def prepare_dataset(self, raw_dataset: datasets.Dataset):
        print('tokenizing....')
        tokenized_dataset = raw_dataset.map(
            self._tokenize_func,
            batched=True,
            num_proc=self._num_proc,
            batch_size=self._batch_size,
            writer_batch_size=self._batch_size,
            remove_columns=[self._text_column],
        )
        print('grouping...')
        processed_dataset = tokenized_dataset.map(
            self._group_text,
            batched=True,
            num_proc=self._num_proc,
            batch_size=self._batch_size,
            writer_batch_size=self._batch_size,
        )

        return processed_dataset

    def _filtering_func(self, examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
        return [line.lower() for line in examples[self._text_column]
                if len(line) > 0 and not line.isspace()]

    def _tokenize_func(self, examples: Dict[str, List[str]]) -> Dict[str, List[int]]:
        examples[self._text_column] = self._filtering_func(examples)
        return self.tokenizer(
            examples[self._text_column],
            return_special_tokens_mask=True,
            add_special_tokens=False,
            return_token_type_ids=False,
        )

    def _group_text(self, examples: Dict[str, List[int]]) -> Dict[str, List[int]]:
        sep_token = {
            'attention_mask': 1,
            'special_tokens_mask': 1,
            'input_ids': self.tokenizer.sep_token_id,
        }

        cls_token = {
            'attention_mask': 1,
            'special_tokens_mask': 1,
            'input_ids': self.tokenizer.cls_token_id,
        }

        pad_token = {
            'attention_mask': 0,
            'special_tokens_mask': 1,
            'input_ids': self.tokenizer.pad_token_id,
        }

        concatenated_examples = {
            k: itertools.chain(
                *map(lambda x: x + [sep_token[k]], examples[k])
            ) for k in examples.keys()
        }

        results = {}
        for k in concatenated_examples:
            iter_chunk = iter(
                lambda: list(itertools.islice(concatenated_examples[k], self.max_len - 2)), []
            )
            batch_list = [[cls_token[k], *batch, sep_token[k]] for batch in iter_chunk]

            if self.padding: 
                tokens_to_pad = batch_list[-1][:-1]
                pad_size = self.max_len - len(tokens_to_pad)
                batch_list[-1] = tokens_to_pad + ([pad_token[k]] * pad_size)

            results[k] = batch_list

        return results
