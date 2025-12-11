import argparse
import os
import datetime
import lightning as pl
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import WeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn

from src.llie.utils.config import load_config, save_config, get_model, get_datamodule
import src.llie.utils.logger as log


def parse_args():
    parser = argparse.ArgumentParser(description='LLIE')
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument("--device", type=str, default='gpu', help="device to use, default is 'gpu'", choices=['gpu', 'cpu'])
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume training")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config_path = args.config
    device = args.device
    resume = args.resume

    if resume is not None:
        config_path = os.path.join(os.path.dirname(resume), "../../../config.yaml")
    config = load_config(config_path)
    model_config, train_config, data_config = config["model"], config["train"], config["data"]
    pl.seed_everything(data_config.get("seed", 42))

    log_dir = os.path.join(train_config["log_dir"], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    save_config(config, os.path.join(log_dir, "config.yaml"))
    logger = loggers.TensorBoardLogger(save_dir=log_dir)
    log.default_logger = log.get_logger(os.path.join(log_dir, "train.log"))

    model = get_model(config)
    datamodule = get_datamodule(data_config)

    trainer = pl.Trainer(
        max_epochs=train_config["epochs"], accelerator=device, devices="auto",
        gradient_clip_val=train_config.get("grad_clip", 0),
        logger=logger, default_root_dir=log_dir,
        callbacks=WeightAveraging(avg_fn=get_ema_avg_fn(train_config.get("ema_decay", 0.999))) if train_config.get("use_ema", False) else None
    )
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == '__main__':
    main()