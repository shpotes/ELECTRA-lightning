from electra import HuggingDataModule

print('logging data')


dm = HuggingDataModule(
    'openwebtext',
    'bert-base-uncased',
    32,
    num_proc=8,
)


print('setup dataset')
dm.setup(None)


print('saving...')
dm.save_to_disk('data/bert_dataset')

print('done!')
