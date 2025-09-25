import os
import argparse
import datetime
import lightning as pl
from lightning.pytorch import loggers

from src.llie.utils.config import load_config, override_config, get_model, get_datamodule
from src.llie.utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="path to the checkpoint file")
    parser.add_argument("--config", type=str, required=True, help="path to the test config file")
    parser.add_argument("--device", type=str, default='gpu', help="device to use, default is 'gpu'", choices=['gpu', 'cpu'])

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ckpt = args.ckpt
    test_config_path = args.config
    device = args.device

    config_path = os.path.join(os.path.dirname(ckpt), "../../../config.yaml")
    config = load_config(config_path)
    model_config, data_config = config["model"], config["data"]
    test_config = load_config(test_config_path)["test"]
    override_config(test_config, data_config)
    pl.seed_everything(getattr(data_config, "seed", 42))

    log_dir = os.path.join(os.path.dirname(ckpt), "../../../..", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    logger = loggers.TensorBoardLogger(save_dir=log_dir)
    extra_logger = get_logger(os.path.join(log_dir, "test.log"))

    model = get_model(config, extra_logger)
    model = model.__class__.load_from_checkpoint(ckpt, config=config, logger=extra_logger)
    datamodule = get_datamodule(data_config, extra_logger)
    datamodule.setup("test")

    trainer = pl.Trainer(accelerator=device, devices="auto", logger=logger, default_root_dir=log_dir)
    trainer.predict(model=model, dataloaders=datamodule.test_dataloader())


if __name__ == '__main__':
    main()