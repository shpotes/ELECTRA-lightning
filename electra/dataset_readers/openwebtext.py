import pathlib
import tarfile
from overrides import overrides
from ._lm_dataset_reader import LMDatasetReader

class OpenWebTextDatasetReader(LMDatasetReader):
    @overrides
    def _read(self, folder_path: str):
        data_dir = pathlib.Path(folder_path)
        for file_path in data_dir.iterdir():
            for instance in self._read_file(file_path):
                yield instance

    def _read_file(self, file_path: str):
        dropped_instances = 0

        with tarfile.open(file_path, mode='r:xz') as xz_buf:
            for mem in xz_buf.getmembers():
                content = xz_buf.extractfile(mem)
                sentence = content.read().decode('utf-8')
                instance = self.text_to_instance(sentence)

                if instance.fields['source'].sequence_length() <= self._max_length:
                    yield instance
                else:
                    dropped_instances += 1
