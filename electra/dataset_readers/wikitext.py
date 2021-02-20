from typing import Iterable

import re
import logging

from allennlp.data.instance import Instance
from allennlp_models.lm import SimpleLanguageModelingDatasetReader
from overrides import overrides

logger = logging.getLogger(__name__)

class WikiTextDatasetReader(SimpleLanguageModelingDatasetReader):
    """
    WikiText Dataset Reader
    """
    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:

        logger.info('Loading data from %s', file_path)
        dropped_instances = 0

        is_title = re.compile(' (= )+(.+ )+(= )+')

        with open(file_path) as file:
            for line in file:
                line = line.replace('\n', '')
                if line.strip() and not is_title.match(line):
                    instance = self.text_to_instance(line.lstrip())
                    if instance.fields['source'].sequence_length() <= self._max_sequence_length:
                        yield instance
                    else:
                        dropped_instances += 1
        if not dropped_instances:
            logger.info(f'No instances dropped from {file_path}.')
        else:
            logger.warning(f'Dropped {dropped_instances} instances from {file_path}.')
