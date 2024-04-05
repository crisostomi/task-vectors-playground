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
    image_encoder = hydra.utils.instantiate(cfg.nn.module.model, keep_lang=False)

    classification_head = get_classification_head(
        cfg.nn.module.model.model_name,
        cfg.data.train_dataset,
        cfg.data.data_path,
        cfg.misc.ckpt_path,
        cache_dir=cfg.misc.cache_dir,
        openclip_cachedir=cfg.misc.openclip_cachedir,
    )

    dataset = get_dataset(
        cfg.data.train_dataset,
        preprocess_fn=image_encoder.train_preprocess,
        location=cfg.data.data_path,
        batch_size=cfg.training.batch_size,
    )

    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()

    num_batches = len(dataset.train_loader)

    loss = hydra.utils.instantiate(cfg.training.loss_fn)

    # Saving zero-shot model
    if cfg.misc.save_pretrained:
        pylogger.info(f"Saving zero-shot model to {cfg.misc.ckpt_path}")
        os.makedirs(cfg.misc.ckpt_path, exist_ok=True)
        model_path = os.path.join(
            cfg.misc.ckpt_path,
            f"zeroshot_{cfg.training.ft_type}"
            + ("_pct_" + str(int(cfg.training.masked_pct * 100)) if cfg.training.ft_type == "masked" else "")
            + ".pt",
        )
        model.module.image_encoder.save(model_path)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.training.lr, weight_decay=cfg.training.wd)

    num_epochs = cfg.training.epochs[cfg.data.dataset_name]
    scheduler = cosine_lr(optimizer, cfg.training.lr, cfg.training.warmup_length, num_epochs * num_batches)

    # NOTE: keep this ALWAYS after the zero-shot model saving,
    # as it adds masking nn.Module attributes (e.g. .weight, .weight_orig, .weight_mask, etc.)

    print_params_summary(model)
    print_mask_summary(model, "by_layer")

    if cfg.training.ft_type == "masked":
        model.module.image_encoder.pick_params_to_prune_by_nn(cfg.training.masked_pct)
        print_params_summary(model)
        print_mask_summary(model, "by_nn")

    # OUR STUFF
    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )
    callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

    storage_dir: str = cfg.core.storage_dir

    logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)

    pylogger.info("Instantiating the <Trainer>")
    trainer = pl.Trainer(
        default_root_dir=storage_dir,
        plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
        logger=logger,
        callbacks=callbacks,
        **cfg.train.trainer,
    )

    print(trainer, loss, scheduler)

    pylogger.info("Starting training!")

    # trainer.fit(model=model, train_dataloaders=dataset., ckpt_path=template_core.trainer_ckpt_path)

    # if "test" in cfg.nn.data.dataset and trainer.checkpoint_callback.best_model_path is not None:
    #     pylogger.info("Starting testing!")
    #     trainer.test(datamodule=datamodule)

    # if logger is not None:
    #     logger.experiment.finish()

    # # Evaluate
    # image_encoder = model.module.image_encoder
    # evaluate(image_encoder, args)

    # if args.save is not None:
    #     zs_path = os.path.join(cfg.misc.ckpt_path, "zeroshot.pt")

    # if cfg.training.ft_type == "full":
    #     ft_path = os.path.join(
    #         cfg.misc.ckpt_path,
    #         f"finetuned_{cfg.training.ft_type}"
    #         + ("_seed_" + str(cfg.seed_index) if cfg.seed_index is not None else "")
    #         + ".pt",
    #     )
    #     image_encoder.save(ft_path)

    # elif cfg.training.ft_type == "masked":
    #     ft_path = os.path.join(
    #         cfg.misc.ckpt_path,
    #         f"finetuned_{cfg.training.ft_type}_pct_{str(int(cfg.training.masked_pct * 100))}"
    #         + ("_seed_" + str(cfg.seed_index) if cfg.seed_index is not None else "")
    #         + "_with_pruning_metadata.pt",
    #     )
    #     image_encoder.save(ft_path)

    #     image_encoder.make_pruning_effective()
    #     ft_path = ft_path.replace("_with_pruning_metadata", "")
    #     image_encoder.save(ft_path)

    return None


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="finetune.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
