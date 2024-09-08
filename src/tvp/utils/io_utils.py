import logging
from pathlib import Path
from pydoc import locate

import torch

from nn_core.serialization import load_model

from tvp.modules.encoder import ClassificationHead, ImageEncoder
from tvp.modules.text_encoder import TextEncoder
from tvp.modules.text_heads import TextClassificationHead

pylogger = logging.getLogger(__name__)


def load_model_from_artifact(run, artifact_path):
    pylogger.info(f"Loading model from artifact {artifact_path}")

    artifact = run.use_artifact(artifact_path)
    artifact.download()

    ckpt_path = Path(artifact.file())

    model_class = locate(artifact.metadata["model_class"])

    if model_class == ImageEncoder:
        model = model_class(**artifact.metadata)
    elif model_class == ClassificationHead:
        model = model_class(normalize=True, **artifact.metadata)
    elif model_class == TextEncoder:
        model = model_class(**artifact.metadata)
    elif model_class == TextClassificationHead:
        model = model_class(**artifact.metadata)
    else:
        raise ValueError(f"Unknown model class {model_class}")

    model.load_state_dict(torch.load(ckpt_path))

    return model


def get_class(model):
    return model.__class__.__module__ + "." + model.__class__.__qualname__
