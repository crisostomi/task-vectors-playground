import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from args import parse_arguments
from tqdm import tqdm

args = parse_arguments()

print(args.arbitrary_tvs)

DPI = 200
nrows = 12  # num of resblocks
ncols = len(args.arbitrary_tvs)
figsize_x = 5 * ncols
figsize_y = 5 * nrows

dataset_name = args.dataset_name + "Val"

ACT_MAP_TYPES = ["mean", "var"]

for act_map_type in ACT_MAP_TYPES:
    fig, axs = plt.subplots(nrows, ncols, figsize=(figsize_x, figsize_y))
    fig.suptitle(f"{args.model} evaluated on {args.dataset_name}")

    for tv_id, tv_name in enumerate(args.arbitrary_tvs):
        act_map: torch.Tensor = torch.load(
            f"./activations_with_tvs_applied/{args.model}/{dataset_name}/{tv_name}/act_maps_{act_map_type}_by_hidden_size_and_num_channels.pt"
        )

        for resblock_id in tqdm(
            range(12), desc=f"{act_map_type} on {tv_name.replace('Val', '').replace('_and_', ', ')} TVs"
        ):
            sns.histplot(data=act_map[f"resblock_{resblock_id}"].numpy(), ax=axs[resblock_id, tv_id], kde=True)

            if resblock_id == 0:
                axs[resblock_id, tv_id].set_title(f"{tv_name}")

        plt.tight_layout()
        fig.subplots_adjust(top=0.95)

        # Save the final figure

        comparison_name = "_VS_".join(args.arbitrary_tvs)
        comparison_name = comparison_name.replace("Val", "")
        out_base_path = f"./act_distribs/side_by_side/arbitrary/{args.model}/{dataset_name}/"
        os.makedirs(out_base_path, exist_ok=True)
        plt.savefig(f"{out_base_path}/{comparison_name}_{act_map_type}_by_hidden_size_and_num_channels.png", dpi=DPI)

    print("\n\n")
