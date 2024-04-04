import os
import time

import torch
import torch.nn.utils.prune as prune
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.finetune.eval import evaluate
from src.models.heads import get_classification_head
from src.models.modeling import ImageClassifier, ImageEncoder
from src.utils import LabelSmoothing, cosine_lr, print_mask_summary, print_params_summary
from src.utils.args import parse_arguments
from tqdm import tqdm


def finetune(args):
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
    ckpdir = os.path.join(args.save, train_dataset)

    # Check if checkpoints already exist
    zs_path = os.path.join(args.save, train_dataset, "checkpoint_0.pt")
    ft_path = os.path.join(args.save, train_dataset, f"checkpoint_{args.epochs}.pt")
    if os.path.exists(zs_path) and os.path.exists(ft_path):
        print(f"Skipping fine-tuning because {ft_path} exists.")
        return zs_path, ft_path

    assert train_dataset is not None, "Please provide a training dataset."
    if args.load is not None and args.load.endswith("pt"):
        image_encoder = ImageEncoder.load(args.load)
    else:
        print("Building image encoder.")
        image_encoder = ImageEncoder(args, keep_lang=False)

    classification_head = get_classification_head(args, train_dataset)

    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()

    preprocess_fn = model.train_preprocess
    print_every = 100000000000

    dataset = get_dataset(train_dataset, preprocess_fn, location=args.data_location, batch_size=args.batch_size)
    num_batches = len(dataset.train_loader)

    devices = list(range(torch.cuda.device_count()))
    print("Using devices", devices)
    model = torch.nn.DataParallel(model, device_ids=devices)

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    # Saving zero-shot model
    if args.save is not None:
        print(f"Saving zero-shot model to {ckpdir}")
        os.makedirs(ckpdir, exist_ok=True)
        model_path = os.path.join(
            ckpdir,
            f"zeroshot_{args.ft_type}"
            + ("_pct_" + str(int(args.masked_pct * 100)) if args.ft_type == "masked" else "")
            + ".pt",
        )
        model.module.image_encoder.save(model_path)

    # NOTE: keep this ALWAYS after the zero-shot model saving,
    # as it adds masking nn.Module attributes (e.g. .weight, .weight_orig, .weight_mask, etc.)
    model = model.cuda()

    print_params_summary(model)
    print_mask_summary(model, "by_layer")

    if args.ft_type == "masked":
        # model.module.image_encoder.pick_params_to_prune_by_layers(args.masked_pct)
        # print_params_summary(model)
        # print_mask_summary(model, "by_layer")

        model.module.image_encoder.pick_params_to_prune_by_nn(args.masked_pct)
        print_params_summary(model)
        print_mask_summary(model, "by_nn")

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        model = model.cuda()
        model.train()
        data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)

        for i, batch in tqdm(enumerate(data_loader), desc=f"Epoch {epoch+1}", total=num_batches):
            start_time = time.time()

            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to("cuda:0")
            labels = batch["labels"].to("cuda:0")
            data_time = time.time() - start_time

            # This workaround https://github.com/pytorch/pytorch/issues/69353 is essential to be executed before the forward
            # in order to prevent PyTorch RuntimeError when pruning and using Attention-based layers
            # It ain't pretty, but it works :). Thx user on GitHub <3
            for i in range(len(model.module.image_encoder.model.visual.transformer.resblocks)):
                module = model.module.image_encoder.model.visual.transformer.resblocks[i].attn.out_proj
                for hook in module._forward_pre_hooks.values():
                    if isinstance(hook, prune.BasePruningMethod):
                        hook(module, None)

            logits, activation_maps = model(inputs)

            loss = loss_fn(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            batch_time = time.time() - start_time

            if step % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",
                    flush=True,
                )

    # Evaluate
    image_encoder = model.module.image_encoder
    evaluate(image_encoder, args)

    if args.save is not None:
        zs_path = os.path.join(ckpdir, "zeroshot.pt")

        if args.ft_type == "full":
            ft_path = os.path.join(
                ckpdir,
                f"finetuned_{args.ft_type}"
                + ("_seed_" + str(args.manual_seed) if args.manual_seed is not None else "")
                + ".pt",
            )
            image_encoder.save(ft_path)

        elif args.ft_type == "masked":
            ft_path = os.path.join(
                ckpdir,
                f"finetuned_{args.ft_type}_pct_{str(int(args.masked_pct * 100))}"
                + ("_seed_" + str(args.manual_seed) if args.manual_seed is not None else "")
                + "_with_pruning_metadata.pt",
            )
            image_encoder.save(ft_path)

            image_encoder.make_pruning_effective()
            ft_path = ft_path.replace("_with_pruning_metadata", "")
            image_encoder.save(ft_path)

        return zs_path, ft_path


if __name__ == "__main__":
    data_location = "./data"

    # models = ['ViT-B-32', 'ViT-B-16', 'ViT-L-14']
    models = ["ViT-B-16"]

    # datasets = ['Cars', 'DTD', 'EuroSAT', 'GTSRB', 'MNIST', 'RESISC45', 'SUN397', 'SVHN']
    datasets = ["RESISC45"]

    epochs = {
        "Cars": 35,
        "DTD": 76,
        "EuroSAT": 12,
        "GTSRB": 11,
        "MNIST": 5,
        "RESISC45": 15,
        "SUN397": 14,
        "SVHN": 4,
        "ImageNet": 6,
        "STL10": 6,
        "CIFAR100": 6,
    }

    for model in models:
        for dataset in datasets:
            print("=" * 100)
            print(f"Finetuning {model} on {dataset}")
            print("=" * 100)
            args = parse_arguments()
            args.lr = 1e-5
            args.epochs = epochs[dataset]
            args.data_location = data_location
            args.train_dataset = dataset + "Val"
            args.eval_datasets = datasets
            args.batch_size = 64
            args.model = model
            args.save = f"checkpoints/{model}"
            finetune(args)
