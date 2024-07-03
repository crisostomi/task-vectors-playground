import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer

wandb.login()
wandb_logger = WandbLogger(name="test run",log_model="all")
trainer = Trainer(logger=wandb_logger)
