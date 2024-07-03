import argparse
import os

import torch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arbitrary-tvs",
        default=None,
        type=lambda x: x.split(","),
        help="Task vector combinations to plot side-by-side. Split by comma, e.g. MNISTVal_and_MNISTVal,MNISTVal_and_EuroSAT_and_STL10Val. (used in activation_stats_side_by_side_arbitrary) ",
    )
    parser.add_argument(
        "--multiple-dataset-names",
        default=None,
        type=lambda x: x.split(","),
        help="Dataset names (used in activations_stats_multiple_models).",
    )
    parser.add_argument("--dry-run", default=False, type=bool, help="Whether to run a dry run or not.")
    parser.add_argument(
        "--apply-task-vectors",
        default=None,
        type=lambda x: x.split(","),
        help="Which task vectors to apply. Split by comma, e.g. MNIST,EuroSAT. ",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Dataset name (used in activations_compute).",
    )
    parser.add_argument(
        "--tv-scaling-coeff",
        type=float,
        help="Task vectors scaling coefficient for when TVs are applied to the base model.",
    )
    parser.add_argument(
        "--ft-type",
        type=str,
        help='What kind of fine-tune type to use. Supported: "full", "masked"',
    )
    parser.add_argument(
        "--masked-pct",
        type=float,
        default=None,
        help="What percentage of the weights of each layer to mask during masked fine-tuning. Value between 0 and 1.",
    )
    parser.add_argument(
        "--manual-seed",
        type=int,
        default=None,
        help="What seed to use.",
    )
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser("~/data"),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. ",
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--exp_name", type=str, default=None, help="Name of the experiment, for organization purposes only."
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--ls", type=float, default=0.0, help="Label smoothing.")
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir", type=str, default="./openclip_cache", help="Directory for caching models from OpenCLIP"
    )
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]

    if parsed_args.ft_type == "masked" and parsed_args.masked_pct is None:
        raise ValueError('"masked" fine-tuning type requires to specify "--masked-pct".')

    return parsed_args
