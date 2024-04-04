import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from args import parse_arguments
from tqdm import tqdm

args = parse_arguments()

DPI = 200
nrows = 12  # num of resblocks
ncols = len(args.apply_task_vectors)
figsize_x = 5 * ncols
figsize_y = 5 * nrows

dataset_name = args.dataset_name + "Val"

apply_task_vectors = args.apply_task_vectors

ACT_MAP_TYPES = ["mean", "var"]

for act_map_type in ACT_MAP_TYPES:
    fig, axs = plt.subplots(nrows, ncols, figsize=(figsize_x, figsize_y))
    fig.suptitle(
        f"{args.model} w/ {', '.join(args.apply_task_vectors)} TVs\nApplied in increasing order\n(1st, 1st + 2nd, 1st + 2nd + 3rd, ...)"
    )

    for tv_id, tv_name in enumerate(apply_task_vectors):
        apply_tvs = "_and_".join(apply_task_vectors[: tv_id + 1])

        act_map: torch.Tensor = torch.load(
            f"./activations_with_tvs_applied/{args.model}/{dataset_name}/{apply_tvs}/act_maps_{act_map_type}_by_hidden_size_and_num_channels.pt"
        )

        for resblock_id in tqdm(range(12), desc=f"{act_map_type} up to {apply_task_vectors[tv_id]} TV"):
            sns.histplot(data=act_map[f"resblock_{resblock_id}"].numpy(), ax=axs[resblock_id, tv_id], kde=True)

            if resblock_id == 0:
                axs[resblock_id, tv_id].set_title(f"{tv_name}")

    plt.tight_layout()
    fig.subplots_adjust(top=0.95)

    # Save the final figure
    out_base_path = (
        f"./act_distribs/side_by_side/sequential/{args.model}/{dataset_name}/{'_and_'.join(args.apply_task_vectors)}"
    )
    os.makedirs(out_base_path, exist_ok=True)
    plt.savefig(f"{out_base_path}/{act_map_type}_by_hidden_size_and_num_channels.png", dpi=DPI)

    print("\n\n")
