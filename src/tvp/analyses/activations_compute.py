import os

import torch
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval import evaluate
from src.heads import get_classification_head
from src.modeling import ImageClassifier, ImageEncoder
from task_vectors import TaskVector
from tqdm import tqdm


def activations(args):
    if args.manual_seed is None:
        import rich

        rich.print("[bold red]WARNING: No manual seed provided. This will result in non-reproducible results.")

    if args.manual_seed is not None:
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_dataset = args.train_dataset
    # ckpdir = os.path.join(args.save, train_dataset)
    act_dir = os.path.join(args.act_save, train_dataset)
    if not os.path.exists(act_dir):
        os.makedirs(act_dir)

    assert train_dataset is not None, "Please provide a training dataset."

    if args.load is not None and args.load.endswith("pt"):
        image_encoder = ImageEncoder.load(args.load)
    else:
        print("Building image encoder.")
        image_encoder = ImageEncoder(args, keep_lang=False)

    if args.use_task_vectors:
        act_dir = os.path.join(act_dir, args.applied_task_vectors_str)
        if not os.path.exists(act_dir):
            os.makedirs(act_dir)

        print(f"Applying task vectors {', '.join(args.apply_task_vectors)} to the model {args.model}.")
        # create task vectors
        task_vectors = [
            TaskVector(args.pretrained_checkpoint, args.finetuned_checkpoints[task_vector_name])
            for task_vector_name in args.apply_task_vectors
        ]

        # Sum the task vectors
        task_vector_sum = sum(task_vectors)

        # Apply the resulting task vector
        image_encoder = task_vector_sum.apply_to(pretrained_model=image_encoder, scaling_coef=args.tv_scaling_coeff)

    classification_head = get_classification_head(args, train_dataset)

    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()

    preprocess_fn = model.train_preprocess

    dataset = get_dataset(train_dataset, preprocess_fn, location=args.data_location, batch_size=args.batch_size)
    num_batches = len(dataset.train_loader)

    devices = list(range(torch.cuda.device_count()))
    print("Using devices", devices)
    model = torch.nn.DataParallel(model, device_ids=devices)

    model.eval()
    data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)

    act_maps_mean_by_hidden_size = {f"resblock_{k}": [] for k in range(0, 12)}
    act_maps_mean_by_num_channels = {f"resblock_{k}": [] for k in range(0, 12)}
    act_maps_mean_by_hidden_size_and_num_channels = {f"resblock_{k}": [] for k in range(0, 12)}
    # act_maps_mean_global = {f"resblock_{k}": [] for k in range(0, 12)}

    act_maps_var_by_hidden_size = {f"resblock_{k}": [] for k in range(0, 12)}
    act_maps_var_by_num_channels = {f"resblock_{k}": [] for k in range(0, 12)}
    act_maps_var_by_hidden_size_and_num_channels = {f"resblock_{k}": [] for k in range(0, 12)}
    # act_maps_var_global = {f"resblock_{k}": [] for k in range(0, 12)}

    torch.set_grad_enabled(False)

    for i, batch in tqdm(enumerate(data_loader), desc="Computing activations", total=num_batches):
        batch = maybe_dictionarize(batch)
        inputs = batch["images"].to("cuda:0")
        # labels = batch["labels"].to("cuda:0")

        logits, activation_maps = model(inputs)

        for k in range(0, 12):
            act_map: torch.Tensor = activation_maps[f"resblock_{k}"]

            # act_maps_mean_by_hidden_size[f"resblock_{k}"].append(act_map.mean(dim=-1).cpu())
            # act_maps_var_by_hidden_size[f"resblock_{k}"].append(act_map.var(dim=-1).cpu())

            # act_maps_mean_by_num_channels[f"resblock_{k}"].append(act_map.mean(dim=0).cpu())
            # act_maps_var_by_num_channels[f"resblock_{k}"].append(act_map.var(dim=0).cpu())

            batch_size = act_map.shape[1]
            temp = act_map.reshape(batch_size, -1)

            act_maps_mean_by_hidden_size_and_num_channels[f"resblock_{k}"].append(temp.mean(dim=-1).cpu())
            act_maps_var_by_hidden_size_and_num_channels[f"resblock_{k}"].append(temp.var(dim=-1).cpu())

            # temp = act_map.reshape(-1).unsqueeze(1)

            # act_maps_mean_global[f"resblock_{k}"].append(temp.mean(dim=0).cpu())
            # act_maps_var_global[f"resblock_{k}"].append(temp.var(dim=0).cpu())

            # act_map = act_map.cpu()

            # print(act_maps_mean_by_hidden_size[f"resblock_{k}"][-1].shape)
            # print(act_maps_var_by_hidden_size[f"resblock_{k}"][-1].shape)
            # print(act_maps_mean_by_num_channels[f"resblock_{k}"][-1].shape)
            # print(act_maps_var_by_num_channels[f"resblock_{k}"][-1].shape)
            # print(act_maps_mean_by_hidden_size_and_num_channels[f"resblock_{k}"][-1].shape)
            # print(act_maps_var_by_hidden_size_and_num_channels[f"resblock_{k}"][-1].shape)
            # print("\n\n--\n\n")

    for k in range(0, 12):
        # act_maps_mean_by_hidden_size[f"resblock_{k}"] = torch.cat(act_maps_mean_by_hidden_size[f"resblock_{k}"], dim=-1).cpu()
        # act_maps_var_by_hidden_size[f"resblock_{k}"] = torch.cat(act_maps_var_by_hidden_size[f"resblock_{k}"], dim=-1).cpu()

        # act_maps_mean_by_num_channels[f"resblock_{k}"] = torch.cat(act_maps_mean_by_num_channels[f"resblock_{k}"], dim=0).cpu()
        # act_maps_var_by_num_channels[f"resblock_{k}"] = torch.cat(act_maps_var_by_num_channels[f"resblock_{k}"], dim=0).cpu()

        act_maps_mean_by_hidden_size_and_num_channels[f"resblock_{k}"] = torch.cat(
            act_maps_mean_by_hidden_size_and_num_channels[f"resblock_{k}"], dim=0
        ).cpu()
        act_maps_var_by_hidden_size_and_num_channels[f"resblock_{k}"] = torch.cat(
            act_maps_var_by_hidden_size_and_num_channels[f"resblock_{k}"], dim=0
        ).cpu()

        # act_maps_mean_global[f"resblock_{k}"] = torch.cat(act_maps_mean_global[f"resblock_{k}"], dim=0).cpu()
        # act_maps_var_global[f"resblock_{k}"] = torch.cat(act_maps_var_global[f"resblock_{k}"], dim=0).cpu()

        # print(act_maps_mean_by_hidden_size[f"resblock_{k}"].shape)
        # print(act_maps_var_by_hidden_size[f"resblock_{k}"].shape)
        # print(act_maps_mean_by_num_channels[f"resblock_{k}"].shape)
        # print(act_maps_var_by_num_channels[f"resblock_{k}"].shape)
        # print(act_maps_mean_by_hidden_size_and_num_channels[f"resblock_{k}"].shape)
        # print(act_maps_var_by_hidden_size_and_num_channels[f"resblock_{k}"].shape)

    # Evaluate
    image_encoder = model.module.image_encoder
    evaluate(image_encoder, args)
    # eval_single_dataset(image_encoder, dataset, args)

    # TODO store tv scaling coeff in activations file name
    elements_to_save = {
        "act_maps_mean_by_hidden_size": act_maps_mean_by_hidden_size,
        "act_maps_var_by_hidden_size": act_maps_var_by_hidden_size,
        "act_maps_mean_by_num_channels": act_maps_mean_by_num_channels,
        "act_maps_var_by_num_channels": act_maps_var_by_num_channels,
        "act_maps_mean_by_hidden_size_and_num_channels": act_maps_mean_by_hidden_size_and_num_channels,
        "act_maps_var_by_hidden_size_and_num_channels": act_maps_var_by_hidden_size_and_num_channels,
        # "act_maps_mean_global": act_maps_mean_global,
        # "act_maps_var_global": act_maps_var_global,
        "model": args.model,
        "dataset": args.train_dataset,
    }

    # Iterate over the dictionary elements and save them with a tqdm progress bar
    for key, value in tqdm(elements_to_save.items(), desc="Saving"):
        torch.save(value, os.path.join(act_dir, f"{key}.pt"))


if __name__ == "__main__":
    args = parse_arguments()

    data_location = "./data"

    args.use_task_vectors = args.apply_task_vectors is not None

    if args.use_task_vectors:
        # args.apply_task_vectors = ["MNISTVal", "RESISC45Val", "SVHNVal"] # passed directly from commandline!

        args.pretrained_checkpoint = f"checkpoints/{args.model}/zeroshot.pt"
        args.finetuned_checkpoints = {
            args.apply_task_vectors[
                i
            ]: f"checkpoints/{args.model}/{args.apply_task_vectors[i]}/finetuned_full_seed_{args.manual_seed}.pt"
            for i in range(len(args.apply_task_vectors))
        }

        args.applied_task_vectors_str = "_and_".join(args.apply_task_vectors)

        args.act_save = f"activations_with_tvs_applied/{args.model}"

    else:
        args.act_save = f"activations/{args.model}"

    print("=" * 100)
    print(f"Looking at activations of {args.model} on {args.dataset_name}")
    print("=" * 100)

    args.epochs = -69

    args.data_location = data_location
    args.train_dataset = args.dataset_name + "Val"
    args.eval_datasets = [args.dataset_name]
    args.batch_size = 128

    args.save = f"checkpoints/{args.model}"

    activations(args)
