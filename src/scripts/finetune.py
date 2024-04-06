import logging
import os
import time
from typing import List, Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn.utils.prune as prune
import wandb
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Callback, LightningModule
from tqdm import tqdm

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

from tvp.data.datasets.common import get_dataloader, maybe_dictionarize
from tvp.data.datasets.registry import get_dataset
from tvp.modules.encoder import ImageEncoder
from tvp.modules.heads import get_classification_head
from tvp.pl_module.image_classifier import ImageClassifier
from tvp.utils.args import parse_arguments
from tvp.utils.eval import evaluate
from tvp.utils.io_utils import load_model_from_artifact
from tvp.utils.utils import LabelSmoothing, cosine_lr, print_mask_summary, print_params_summary


def build_callbacks(cfg: ListConfig, *args: Callback) -> List[Callback]:
    """Instantiate the callbacks given their configuration.

    Args:
        cfg: a list of callbacks instantiable configuration
        *args: a list of extra callbacks already instantiated

    Returns:
        the complete list of callbacks to use
    """
    callbacks: List[Callback] = list(args)

    for callback in cfg:
        pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

    return callbacks


pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig):
    seed_index_everything(cfg)

    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )

    logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)

    zeroshot_identifier = f"{cfg.nn.module.model.model_name}_pt"

    if cfg.reset_pretrained_model:
        image_encoder = hydra.utils.instantiate(cfg.nn.module.model, keep_lang=False)

        classification_head = get_classification_head(
            cfg.nn.module.model.model_name,
            cfg.nn.data.train_dataset,
            cfg.nn.data.data_path,
            cfg.misc.ckpt_path,
            cache_dir=cfg.misc.cache_dir,
            openclip_cachedir=cfg.misc.openclip_cachedir,
        )

        model = hydra.utils.instantiate(
            cfg.nn.module, encoder=image_encoder, classifier=classification_head, _recursive_=False
        )

        upload_model_to_wandb(model, zeroshot_identifier, logger.experiment, cfg)

    else:
        model = load_model_from_artifact(artifact_path=f"{zeroshot_identifier}:latest", run=logger.experiment)

    dataset = get_dataset(
        cfg.nn.data.train_dataset,
        preprocess_fn=model.encoder.train_preprocess,
        location=cfg.nn.data.data_path,
        batch_size=cfg.nn.data.batch_size.train,
    )

    model.freeze_head()

    callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

    storage_dir: str = cfg.core.storage_dir

    pylogger.info("Instantiating the <Trainer>")
    trainer = pl.Trainer(
        default_root_dir=storage_dir,
        plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
        max_epochs=cfg.nn.data.dataset.ft_epochs,
        logger=logger,
        callbacks=callbacks,
        **cfg.train.trainer,
    )

    pylogger.info("Starting training!")
    trainer.fit(model=model, train_dataloaders=dataset.train_loader, ckpt_path=template_core.trainer_ckpt_path)

    pylogger.info("Starting testing!")
    trainer.test(model=model, dataloaders=dataset.test_loader)

    artifact_name = f"{cfg.nn.data.dataset.dataset_name}_{cfg.nn.module.model.model_name}_{cfg.seed_index}"
    upload_model_to_wandb(model, artifact_name, logger.experiment, cfg)

    if logger is not None:
        logger.experiment.finish()


def upload_model_to_wandb(model: LightningModule, artifact_name, run, cfg: DictConfig):
    pylogger.info(f"Uploading artifact {artifact_name}")

    trainer = pl.Trainer(
        plugins=[NNCheckpointIO(jailing_dir="./tmp")],
    )

    temp_path = "temp_checkpoint.ckpt"

    trainer.strategy.connect(model)
    trainer.save_checkpoint(temp_path)

    model_class = model.__class__.__module__ + "." + model.__class__.__qualname__

    model_artifact = wandb.Artifact(
        name=artifact_name,
        type="checkpoint",
        metadata={"model_identifier": cfg.nn.module.model.model_name, "model_class": model_class},
    )

    model_artifact.add_file(temp_path + ".zip", name="trained.ckpt.zip")
    run.log_artifact(model_artifact)

    os.remove(temp_path + ".zip")


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="finetune.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
