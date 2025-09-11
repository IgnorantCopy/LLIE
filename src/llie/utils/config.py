import yaml
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import lightning as pl
from loguru import Logger


def load_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f.read())
    return config


def save_config(config, config_file):
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def get_model(config, logger: Logger) -> pl.LightningModule:
    name = config["model"]["name"]

    logger.info(f"Loading model: {name}")
    if name == "RetinexNet":
        from src.llie.models.retinex_net import RetinexNet
        model = RetinexNet(config, logger)
    else:
        logger.error(f"Invalid model name: {name}")
        raise ValueError(f"Invalid model name: {name}")

    logger.info(f"{name} model loaded")
    return model


def get_optimizer(train_config, model: nn.Module, logger: Logger) -> optim.Optimizer:
    optimizer_config = train_config["optimizer"]
    optimizer_name = optimizer_config["name"]
    lr = train_config["lr"]

    logger.info(f"Loading optimizer: {optimizer_name}")
    if optimizer_name == "Adam":
        weight_decay = optimizer_config.get("weight_decay", 5e-4)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        momentum = optimizer_config.get("momentum", 0.9)
        weight_decay = optimizer_config.get("weight_decay", 5e-4)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        logger.error(f"Unsupported optimizer: {optimizer_name}")
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    logger.info(f"{optimizer_name} optimizer loaded")
    return optimizer


def get_scheduler(train_config, optimizer: optim.Optimizer, logger: Logger):
    scheduler_config = train_config["scheduler"]
    scheduler_name = scheduler_config["name"]

    logger.info(f"Loading scheduler: {scheduler_name}")
    if scheduler_name == "ReduceLROnPlateau":
        factor = scheduler_config.get("factor", 0.5)
        patience = scheduler_config.get("patience", 10)
        min_lr = scheduler_config.get("min_lr", 1e-6)
        scheduler =  lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr)
    else:
        logger.error(f"Unsupported scheduler: {scheduler_name}")
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    logger.info(f"{scheduler_name} scheduler loaded")
    return scheduler


def get_datamodule(data_config, logger: Logger):
    name = data_config["name"]

    logger.info(f"Loading dataset: {name}")
    if name == "LOLv1" or name == "LOLv2_Real" or name == "LOLv2_Synthetic":
        from src.llie.data.lol import LOLDataModule
        data_module = LOLDataModule(data_config)
    elif name == "LOLBlur":
        from src.llie.data.lol_blur import LOLBlurDataModule
        data_module = LOLBlurDataModule(data_config)
    elif name == "SICE":
        from src.llie.data.sice import SICEDataModule
        data_module = SICEDataModule(data_config)
    elif name == "Unpaired":
        from src.llie.data.unpaired import UnpairedDataModule
        data_module = UnpairedDataModule(data_config)
    else:
        logger.error(f"Invalid dataset: {name}")
        raise ValueError(f"Invalid dataset: {name}")

    logger.info(f"{name} dataset loaded")
    return data_module