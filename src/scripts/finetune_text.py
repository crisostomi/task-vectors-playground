import logging
import os
import time
from typing import Dict, List, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Callback, LightningModule
from tqdm import tqdm

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

from tvp.data.datasets.registry import get_text_dataset
from tvp.modules.text_encoder import TextEncoder
from tvp.modules.text_heads import get_classification_head
from tvp.pl_module.text_classifier import TextClassifier
from tvp.utils.io_utils import get_class, load_model_from_artifact
from tvp.utils.utils import LabelSmoothing, build_callbacks

pylogger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")

num_to_th = {
    1: "st",
    2: "nd",
    3: "rd",
    4: "th",
    5: "th",
    6: "th",
    7: "th",
    8: "th",
    9: "th",
    10:"th",
    11: "th",
    12: "th",
    13: "th",
    14: "th",
    15: "th",
    16: "th",
    17: "th",
    18: "th",
    19: "th",
    20:"th"
}

def run(cfg: DictConfig):
    
    seed_index_everything(cfg)

    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )

    logger: NNLogger = NNLogger(
        logging_cfg=cfg.train.logging, 
        cfg=cfg, 
        resume_id=template_core.resume_id
    )
    
    classification_head_identifier = f"{cfg.nn.module.model.model_name}_{cfg.nn.data.dataset.dataset_name}_head"
    
    text_encoder: TextEncoder = hydra.utils.instantiate(cfg.nn.module.model, keep_lang=False)
        
    model_class = get_class(text_encoder)
    metadata = {"model_name": cfg.nn.module.model.model_name, "model_class": model_class}
    
    if cfg.reset_classification_head:
        classification_head = get_classification_head(
            input_size=cfg.nn.module.model.hidden_size,
            num_classes=cfg.nn.data.dataset.num_classes
        )

        model_class = get_class(classification_head)
        
        metadata = {
            "model_name": cfg.nn.module.model.model_name,
            "model_class": model_class,
            "num_classes": cfg.nn.data.dataset.num_classes,
            "input_size": cfg.nn.module.model.hidden_size,
        }

    else:
        classification_head = load_model_from_artifact(
            artifact_path=f"{classification_head_identifier}:latest", 
            run=logger.experiment
        )

    model: TextClassifier = hydra.utils.instantiate(
        cfg.nn.module, 
        encoder=text_encoder, classifier=classification_head, 
        _recursive_=False,
        save_grad_norms=cfg.train.save_grad_norms
    )

    dataset = get_text_dataset(
        dataset_name=cfg.nn.data.train_dataset,
        tokenizer_name=cfg.nn.module.model.model_name,
        train_split_ratio_for_val=cfg.nn.data.splits_pct.val,
        max_seq_length=cfg.nn.data.max_seq_length,
        batch_size=cfg.nn.data.batch_size.train,
        num_workers=cfg.nn.data.num_workers.train
    )

    model.freeze_head()

    callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

    storage_dir: str = cfg.core.storage_dir

    pylogger.info("Instantiating the <Trainer>")
    trainer = pl.Trainer(
        default_root_dir=storage_dir,
        plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
        # max_epochs=cfg.epochs, 
        max_epochs=cfg.nn.data.dataset.ft_epochs,
        logger=logger,
        callbacks=callbacks,
        **cfg.train.trainer,
    )

    pylogger.info(f"Starting fine-tuning on {cfg.ft_on_data_split} data split!")
    if cfg.ft_on_data_split == "train":
        ft_dataloader = dataset.train_loader
    elif cfg.ft_on_data_split == "val":
        ft_dataloader = dataset.val_loader
    else:
        raise ValueError(f"Unknown data split to fine-tune on: {cfg.ft_on_data_split}. Possible values: \"train\" or \"val\"")

    pylogger.info("Starting training!")
    trainer.fit(
        model=model, 
        train_dataloaders=ft_dataloader, 
        ckpt_path=template_core.trainer_ckpt_path
    )

    pylogger.info("Starting testing!")
    trainer.test(model=model, dataloaders=dataset.test_loader)

    model_class = get_class(text_encoder)
    
    metadata = {
        "model_name": cfg.nn.module.model.model_name, 
        "model_class": model_class
    }

    if logger is not None:
        logger.experiment.finish()
    
    encoder_ckpt = os.path.join(storage_dir, "encoder.pt")
    torch.save(model.encoder.state_dict(), encoder_ckpt)

    head_ckpt = os.path.join(storage_dir, "head.pt")
    torch.save(model.classification_head.state_dict(), head_ckpt)



@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="finetune.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)

if __name__ == "__main__":
    main()