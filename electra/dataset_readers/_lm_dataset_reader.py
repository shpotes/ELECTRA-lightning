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

        return_instance = Instance({
            'source': TextField(tokenized, self._token_indexers)
        })

        return return_instance

    @overrides
    def _read(self, file_path: str):
        raise NotImplementedError
