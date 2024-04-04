import gc
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from args import parse_arguments
from tqdm import tqdm

args = parse_arguments()

if args.apply_task_vectors is None:
    BASE_PATH = f"./activations/{args.model}/{args.dataset_name + 'Val'}"
else:
    BASE_PATH = f"./activations_with_tvs_applied/{args.model}/{args.dataset_name + 'Val'}/{'_and_'.join(args.apply_task_vectors)}"

DPI = 300

acts_to_plot = [
    # "act_maps_mean_by_hidden_size",
    # "act_maps_var_by_hidden_size",
    # "act_maps_mean_by_num_channels",
    # "act_maps_var_by_num_channels",
    "act_maps_mean_by_hidden_size_and_num_channels",
    "act_maps_var_by_hidden_size_and_num_channels",
    # "act_maps_mean_global",
    # "act_maps_var_global",
    # "model",
    # "dataset"
]


def save_plot_by_hidden_size(act_maps_by_hidden_size, act_base_path: str, resblock_ids: list):
    for resblock_id, resblock_key in tqdm(
        enumerate(act_maps_by_hidden_size.keys()), desc="Resblocks", total=len(act_maps_by_hidden_size.keys())
    ):
        if resblock_id not in resblock_ids:
            continue

        act_map = act_maps_by_hidden_size[resblock_key]

        # print(f"Resblock {resblock_id}: {act_map.shape}") # (num_channels, dataset_size)

        act_path = f"{act_base_path}/resblock_{str(resblock_id).zfill(2)}"
        os.makedirs(act_path, exist_ok=True)

        for channel_id in tqdm(
            range(act_map.shape[0]), desc=f"Channels of resblock {resblock_id}", total=act_map.shape[0]
        ):
            # print(f"  Channel ID {channel_id}: {act_map[channel_id]}")
            # print(f"  Channel ID {channel_id}")

            dist = sns.displot(act_map[channel_id].numpy(), kde=True)
            dist.savefig(f"{act_path}/channel_{str(channel_id).zfill(3)}.png", dpi=DPI)
            plt.close("all")

            if channel_id > 3:
                if args.dry_run:
                    break

        if resblock_id > 2:
            if args.dry_run:
                break


if "act_maps_mean_by_hidden_size" in acts_to_plot:
    act_maps_mean_by_hidden_size = torch.load(f"{BASE_PATH}/act_maps_mean_by_hidden_size.pt")
    act_base_path = f"{BASE_PATH}/plots/mean_by_hidden_size"
    os.makedirs(act_base_path, exist_ok=True)
    # resblock_ids = range(12)
    resblock_ids = [11]
    save_plot_by_hidden_size(act_maps_mean_by_hidden_size, act_base_path, resblock_ids)
    del act_maps_mean_by_hidden_size
    gc.collect()


if "act_maps_var_by_hidden_size" in acts_to_plot:
    act_maps_var_by_hidden_size = torch.load(f"{BASE_PATH}/act_maps_var_by_hidden_size.pt")
    act_base_path = f"{BASE_PATH}/plots/var_by_hidden_size"
    os.makedirs(act_base_path, exist_ok=True)
    # resblock_ids = range(12)
    resblock_ids = [11]
    save_plot_by_hidden_size(act_maps_var_by_hidden_size, act_base_path, resblock_ids)
    del act_maps_var_by_hidden_size
    gc.collect()


def save_plot_by_num_channels(act_maps_by_hidden_size, act_base_path: str, resblock_ids: list):
    for resblock_id, resblock_key in tqdm(
        enumerate(act_maps_by_hidden_size.keys()), desc="Resblocks", total=len(act_maps_by_hidden_size.keys())
    ):
        if resblock_id not in resblock_ids:
            continue

        act_map = act_maps_by_hidden_size[resblock_key]

        # print(f"Resblock {resblock_id}: {act_map.shape}") # (dataset_size, hidden_size)

        act_path = f"{act_base_path}/resblock_{str(resblock_id).zfill(2)}"
        os.makedirs(act_path, exist_ok=True)

        for hidden_dim in tqdm(
            range(0, act_map.shape[1]), desc=f"Channels of resblock {resblock_id}", total=abs(0 - act_map.shape[1])
        ):
            # print(f"  Hidden dim {hidden_dim}: {act_map[hidden_dim]}")
            # print(f"  Hidden dim {hidden_dim}")

            dist = sns.displot(act_map[:, hidden_dim].numpy(), kde=True)
            dist.savefig(f"{act_path}/hidden_dim_{str(hidden_dim).zfill(3)}.png", dpi=DPI, format="svg")
            plt.close("all")

            if hidden_dim > 3:
                if args.dry_run:
                    break

        if resblock_id > 2:
            if args.dry_run:
                break


if "act_maps_mean_by_num_channels" in acts_to_plot:
    act_maps_mean_by_num_channels = torch.load(f"{BASE_PATH}/act_maps_mean_by_num_channels.pt")
    act_base_path = f"{BASE_PATH}/plots/mean_by_num_channels"
    os.makedirs(act_base_path, exist_ok=True)
    # resblock_ids = range(12)
    resblock_ids = [11]
    save_plot_by_num_channels(act_maps_mean_by_num_channels, act_base_path, resblock_ids)
    del act_maps_mean_by_num_channels
    gc.collect()


if "act_maps_var_by_num_channels" in acts_to_plot:
    act_maps_var_by_num_channels = torch.load(f"{BASE_PATH}/act_maps_var_by_num_channels.pt")
    act_base_path = f"{BASE_PATH}/plots/var_by_num_channels"
    os.makedirs(act_base_path, exist_ok=True)
    # resblock_ids = range(12)
    resblock_ids = [11]
    save_plot_by_num_channels(act_maps_var_by_num_channels, act_base_path, resblock_ids)
    del act_maps_var_by_num_channels
    gc.collect()


def save_plot_by_hidden_size_and_num_channels(act_maps_by_hidden_size, act_base_path: str, resblock_ids: list):
    for resblock_id, resblock_key in tqdm(
        enumerate(act_maps_by_hidden_size.keys()), desc="Resblocks", total=len(act_maps_by_hidden_size.keys())
    ):
        if resblock_id not in resblock_ids:
            continue

        act_map = act_maps_by_hidden_size[resblock_key]

        # print(f"Resblock {resblock_id}: {act_map.shape}") # (dataset_size)

        act_path = f"{act_base_path}/resblock_{str(resblock_id).zfill(2)}"
        # os.makedirs(act_path, exist_ok=True) # not needed as we do not have block-wise subdirs!

        dist = sns.displot(act_map.numpy(), kde=True)
        dist.savefig(f"{act_path}_dist.png", dpi=DPI)
        plt.close("all")

        if resblock_id > 2:
            if args.dry_run:
                break


if "act_maps_mean_by_hidden_size_and_num_channels" in acts_to_plot:
    act_maps_mean_by_hidden_size_and_num_channels = torch.load(
        f"{BASE_PATH}/act_maps_mean_by_hidden_size_and_num_channels.pt"
    )
    act_base_path = f"{BASE_PATH}/plots/mean_by_hidden_size_and_num_channels"
    os.makedirs(act_base_path, exist_ok=True)
    resblock_ids = range(12)
    # resblock_ids = [11]
    save_plot_by_hidden_size_and_num_channels(
        act_maps_mean_by_hidden_size_and_num_channels, act_base_path, resblock_ids
    )
    del act_maps_mean_by_hidden_size_and_num_channels
    gc.collect()


if "act_maps_var_by_hidden_size_and_num_channels" in acts_to_plot:
    act_maps_var_by_hidden_size_and_num_channels = torch.load(
        f"{BASE_PATH}/act_maps_var_by_hidden_size_and_num_channels.pt"
    )
    act_base_path = f"{BASE_PATH}/plots/var_by_hidden_size_and_num_channels"
    os.makedirs(act_base_path, exist_ok=True)
    resblock_ids = range(12)
    # resblock_ids = [11]
    save_plot_by_hidden_size_and_num_channels(act_maps_var_by_hidden_size_and_num_channels, act_base_path, resblock_ids)
    del act_maps_var_by_hidden_size_and_num_channels
    gc.collect()
