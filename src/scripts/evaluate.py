## Imports
import logging
import os
from typing import List, Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from lightning.pytorch import Callback
from omegaconf import DictConfig, ListConfig
import copy

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

# Force the execution of __init__.py if this file is executed directly.
import tvp  # noqa
from tvp.data.datamodule import MetaData
from tvp.data.datasets.registry import get_dataset
from tvp.task_vectors.task_vectors import TaskVector
from tvp.utils.io_utils import load_model_from_artifact
from tvp.utils.utils import build_callbacks
from torch.nn.utils import vector_to_parameters
from torch.nn.utils import parameters_to_vector
from hydra.utils import instantiate
import hydra
from hydra import initialize, compose
from typing import Dict, List

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def run(cfg: DictConfig) -> str:
    """Generic train loop.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Returns:
        the run directory inside the storage_dir used by the current experiment
    """

    seed_index_everything(cfg)

    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )
    logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)

    zeroshot_identifier = f"{cfg.nn.module.model.model_name}_pt"

    zeroshot_model = load_model_from_artifact(artifact_path=f"{zeroshot_identifier}:latest", run=logger.experiment)

    finetuned_id_fn = lambda dataset: f"{cfg.nn.module.model.model_name}_{dataset}_{cfg.seed_index}_sparseClipping:latest"

    finetuned_models = {
        dataset: load_model_from_artifact(artifact_path=finetuned_id_fn(dataset), run=logger.experiment)
        for dataset in cfg.task_vectors.to_apply
    }

    zeroshot_orig_weights = copy.deepcopy(zeroshot_model.state_dict())

    # Task vectors
    flatten = lambda model: parameters_to_vector(model.parameters())

    zeroshot_vec = flatten(zeroshot_model)
    task_vectors = [
        TaskVector.from_models(zeroshot_model, finetuned_models[dataset]) for dataset in cfg.task_vectors.to_apply
    ]

    def apply_task_vector(model, task_vector):
        model.load_state_dict({k: v + task_vector[k] for k, v in model.state_dict().items()})

    with torch.no_grad():
        task_vectors = torch.stack(
            [flatten(finetuned_models[dataset]) - zeroshot_vec for dataset in cfg.task_vectors.to_apply]
        )

    task_vector_aggregator = instantiate(cfg.task_vectors.aggregator)
    multi_task_vector = task_vector_aggregator(task_vectors)

    delta_model = copy.deepcopy(zeroshot_model)
    vector_to_parameters(multi_task_vector, delta_model.parameters())
    task_equipped_model = copy.deepcopy(zeroshot_model)
    apply_task_vector(task_equipped_model, delta_model.state_dict())

    seed_index_everything(cfg)

    results = {}

    for dataset_name in cfg.eval_datasets:

        classification_head_identifier = f"{cfg.nn.module.model.model_name}_{dataset_name}_head"
        classification_head = load_model_from_artifact(
            artifact_path=f"{classification_head_identifier}:latest", run=logger.experiment
        )

        model = hydra.utils.instantiate(
            cfg.nn.module, encoder=task_equipped_model, classifier=classification_head, _recursive_=False
        )

        dataset = get_dataset(
            dataset_name,
            preprocess_fn=model.encoder.train_preprocess,
            location=cfg.nn.data.data_path,
            batch_size=cfg.nn.data.batch_size.train,
        )

        callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

        storage_dir: str = cfg.core.storage_dir

        pylogger.info("Instantiating the <Trainer>")
        trainer = pl.Trainer(
            default_root_dir=storage_dir,
            plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
            logger=False,
            callbacks=callbacks,
            **cfg.train.trainer,
        )

        # Evaluation
        if cfg.eval_on_train:
            pylogger.info("Evaluating on the training set")
            trainer.test(model=model, dataloaders=dataset.train_loader)

        pylogger.info("Evaluating on the test set!")
        test_results = trainer.test(model=model, dataloaders=dataset.test_loader)

        results[dataset_name] = test_results

    print(results)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="task_vectors.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
