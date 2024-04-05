import logging
import os
from typing import List, Optional

import hydra
import lightning.pytorch as pl
import omegaconf
import torch
from lightning.pytorch import Callback
from omegaconf import DictConfig, ListConfig

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

# Force the execution of __init__.py if this file is executed directly.
import tvp  # noqa
from tvp.data.datamodule import MetaData
from tvp.task_vectors.task_vectors import TaskVector
from tvp.utils.args import parse_arguments
from tvp.utils.eval import eval_single_dataset, evaluate

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


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


def run(cfg: DictConfig) -> str:
    """Generic train loop.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Returns:
        the run directory inside the storage_dir used by the current experiment
    """
    seed_index_everything(cfg)

    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    # Config
    datasets_finetuned_name = {
        task_vector: f"finetuned_full_seed_{cfg.seed_index}.pt" for task_vector in cfg.task_vectors.to_apply
    }

    model = cfg.model.model
    pretrained_checkpoint = cfg.misc.pretrained_checkpoint

    # Create the task vectors
    task_vectors = [
        TaskVector(
            pretrained_checkpoint, f"{PROJECT_ROOT}/checkpoints/{model}/{dataset}/{datasets_finetuned_name[dataset]}"
        )
        for dataset in cfg.task_vectors.to_apply
    ]

    # Sum the task vectors
    task_vector_sum = sum(task_vectors)

    # Apply the resulting task vector
    image_encoder = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=cfg.task_vectors.scaling_coefficient)

    # Evaluate
    evaluate(image_encoder, cfg)

    # Instantiate the callbacks
    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )
    print(template_core)

    # callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

    # logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)

    # # Instantiate datamodule
    # pylogger.info(f"Instantiating <{cfg.nn.data['_target_']}>")
    # datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)
    # datamodule.setup(stage=None)

    # metadata: Optional[MetaData] = getattr(datamodule, "metadata", None)
    # if metadata is None:
    #     pylogger.warning(f"No 'metadata' attribute found in datamodule <{datamodule.__class__.__name__}>")

    # # Instantiate model
    # pylogger.info(f"Instantiating <{cfg.nn.module['_target_']}>")
    # model: pl.LightningModule = hydra.utils.instantiate(cfg.nn.module, _recursive_=False, metadata=metadata)

    # # Instantiate the callbacks
    # template_core: NNTemplateCore = NNTemplateCore(
    #     restore_cfg=cfg.train.get("restore", None),
    # )
    # callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

    # storage_dir: str = cfg.core.storage_dir

    # logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)

    # pylogger.info("Instantiating the <Trainer>")
    # trainer = pl.Trainer(
    #     default_root_dir=storage_dir,
    #     plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
    #     logger=logger,
    #     callbacks=callbacks,
    #     **cfg.train.trainer,
    # )

    # pylogger.info("Starting training!")
    # trainer.fit(model=model, datamodule=datamodule, ckpt_path=template_core.trainer_ckpt_path)

    # if fast_dev_run:
    #     pylogger.info("Skipping testing in 'fast_dev_run' mode!")
    # else:
    #     if datamodule.test_dataset is not None and trainer.checkpoint_callback.best_model_path is not None:
    #         pylogger.info("Starting testing!")
    #         trainer.test(datamodule=datamodule)

    # Logger closing to release resources/avoid multi-run conflicts
    # if logger is not None:
    # logger.experiment.finish()

    return None


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="task_vectors.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()


# Evaluate # NOTE this evaluates on the training set, not the validation set
# for dataset in args.apply_task_vectors:
#     eval_single_dataset(image_encoder, dataset, args)
#     pass

# Save model with applied task vectors
# datasets_str = '_and_'.join(args.apply_task_vectors)
# export_path = f'checkpoints/{model}/{datasets_str}'
# os.makedirs(export_path, exist_ok=True)
# if args.save is not None:
#     image_encoder.save(f"{export_path}/finetuned_full_seed_{args.manual_seed}_scaling_coeff_{args.tv_scaling_coeff}.pt")
