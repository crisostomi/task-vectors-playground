import logging
from pathlib import Path

from nn_core.serialization import load_model

from tvp.pl_module.image_classifier import ImageClassifier

pylogger = logging.getLogger(__name__)


def load_model_from_artifact(run, artifact_path):
    pylogger.info(f"Loading model from artifact {artifact_path}")

    artifact = run.use_artifact(artifact_path)
    artifact.download()

    model = load_model(ImageClassifier, checkpoint_path=Path(artifact.file()))[0]
    model.eval()

    return model
