import hydra
from electra.utils.hydra import (
    create_datamodule,
    create_model,
    create_trainer
)

import pathlib

@hydra.main(config_path='conf', config_name='config')
def run(cfg):
    print(cfg.pretty())

    print(pathlib.Path('.').absolute())

    dm = create_datamodule(cfg)
    model = create_model(dm, cfg)
    trainer = create_trainer(cfg, None)

    trainer.fit(model, dm)

if __name__ == '__main__':
    run()
