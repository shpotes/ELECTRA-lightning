from typing import Any, Optional, Dict, Generator

from overrides import overrides

from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

class LMDatasetReader(DatasetReader):
    def __init__(
        self, 
        model_name: str,
        max_length: int = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._tokenizer = PretrainedTransformerTokenizer(
            model_name,
            max_length=max_length,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        self._token_indexers = {
            'tokens': PretrainedTransformerIndexer(
                model_name,
                max_length=max_length,
                tokenizer_kwargs=tokenizer_kwargs,
            )
        }

        self._max_length = self.get_tokenizer().model_max_length

        self._unk_token = Token(self.get_tokenizer()._unk_token)
        self._sep_token = Token(self.get_tokenizer()._sep_token)
        self._pad_token = Token(self.get_tokenizer()._pad_token)
        self._cls_token = Token(self.get_tokenizer()._cls_token)


    def get_tokenizer(self):
        return self._tokenizer.tokenizer

    @overrides
    def text_to_instance(self, sentence: str) -> Instance:
        tokenized = self._tokenizer.tokenize(sentence)
        new_set = True

        while tokenized:
            tokenized_with_ends = []
            _current_tokens = []

            start_token = self._cls_token if new_set else self._sep_token
            tokenized_with_ends.append(start_token)

            if len(tokenized) > self._max_length - 1:
                _current_tokens = tokenized[:(self._max_length-1)]
                tokenized = tokenized[(self._max_length-1):]
                new_set = False
            else:
                _current_tokens = tokenized
                tokenized = []
                new_set = True

            tokenized_with_ends.extend(_current_tokens)
            tokenized_with_ends.append(self._sep_token)

            yield_instance = Instance({
                'source': TextField(tokenized_with_ends, self._token_indexers)
            })

            yield yield_instance

    @overrides
    def _read(self, file_path: str):
        raise NotImplementedError
