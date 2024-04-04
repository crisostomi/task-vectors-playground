import argparse
from math import ceil

import torch
from latentis.measure.base import CKA, CKAMode
from rich import print

parser = argparse.ArgumentParser(description="Example script with argparse")
parser.add_argument(
    "--tvs-applied-to-x", type=lambda x: "_and_".join(x.split(",")), help="Task vectors that have been applied to X"
)
parser.add_argument(
    "--tvs-applied-to-y", type=lambda x: "_and_".join(x.split(",")), help="Task vectors that have been applied to Y"
)
parser.add_argument("--dataset-name", type=str, help="Dataset to perform evaluation on")
args = parser.parse_args()

if args.tvs_applied_to_x is None:
    if "_and_" not in args.tvs_applied_to_y:
        args.tvs_applied_to_x = args.tvs_applied_to_y
    else:
        args.tvs_applied_to_x = "_and_".join(args.tvs_applied_to_y.split("_and_")[:-1])
    print("[bold yellow]tvs_applied_to_x not provided, continuing with sequential mode considering tvs_applied_to_y!\n")

print(f"{args.tvs_applied_to_x}\nvs.\n{args.tvs_applied_to_y}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cka = CKA(mode=CKAMode.LINEAR, device=device)
MODEL_NAME = "ViT-B-16"

DATASET_NAME = args.dataset_name + "Val"

BASE_PATH = f"activations_with_tvs_applied/{MODEL_NAME}/{DATASET_NAME}"

X_PATH = f"{BASE_PATH}/{args.tvs_applied_to_x}/act_maps_mean_by_hidden_size_and_num_channels.pt"
x_all = torch.load(X_PATH)

Y_PATH = f"{BASE_PATH}/{args.tvs_applied_to_y}/act_maps_mean_by_hidden_size_and_num_channels.pt"
y_all = torch.load(Y_PATH)

BATCH_SIZE = 128

cka_resblocks = []

for resblock_id in range(12):
    x_resblock = x_all[f"resblock_{resblock_id}"]
    y_resblock = y_all[f"resblock_{resblock_id}"]

    assert x_resblock.size(0) == y_resblock.size(
        0
    ), f"Resblock {resblock_id} - X and Y have different number of samples: {x_resblock.size(0)} vs {y_resblock.size(0)}"

    x_resblock_chunks = x_resblock.chunk(ceil(x_resblock.size(0) / BATCH_SIZE), dim=0)
    y_resblock_chunks = y_resblock.chunk(ceil(y_resblock.size(0) / BATCH_SIZE), dim=0)

    cka_vals = []

    for x, y in zip(x_resblock_chunks, y_resblock_chunks):
        cka_vals.append(cka(x.unsqueeze(-1), y.unsqueeze(-1)))

        # print(f"Resblock {resblock_id} - CKA: {cka_vals[-1]:.4f}")

    cka_all = torch.mean(torch.stack(cka_vals))
    cka_resblocks.append(cka_all)

    # print(f"Resblock {resblock_id} - Mean CKA: {cka_all:.4f}")
    print(f"{cka_all:.4f}")

# print(f"Resblock AVG - Mean CKA: {torch.mean(torch.stack(cka_resblocks)):.4f}")
print(f"{torch.mean(torch.stack(cka_resblocks)):.4f}")
