import open_clip
import torch
from torch.nn.utils import prune

from tvp import utils
from tvp.utils.utils import torch_load, torch_save


class ImageEncoder(torch.nn.Module):
    def __init__(self, model_name: str, openclip_cachedir=None, cache_dir=None, keep_lang=False, **kwargs):
        super().__init__()

        print(f"Loading {model_name} pre-trained weights.")
        if "__pretrained__" in model_name:
            name, pretrained = model_name.split("__pretrained__")
        else:
            name = model_name
            pretrained = "openai"

        self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, cache_dir=openclip_cachedir
        )

        self.cache_dir = cache_dir

        if not keep_lang and hasattr(self.model, "transformer"):
            delattr(self.model, "transformer")

        # NOTE excluding the classification head
        # TODO eval whether it should be included as well
        self.MODULE_NAMES_ELIGIBLE_FOR_FREEZING = [
            "conv1",
            "ln_pre",
            "ln_1",
            "ln_2",
            "c_fc",
            "c_proj",
            "ln_post",
            "ln_final",
            "token_embedding",
            "out_proj",  # gotta properly handle it (https://github.com/pytorch/pytorch/issues/69353 <3) to prevent RuntimeError
        ]

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image encoder to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, model_name, filename):
        print(f"Loading image encoder from {filename}")
        state_dict = torch.load(filename)
        return cls.load(model_name, state_dict)

    # @classmethod
    # def load_from_state_dict(cls, model_name, state_dict):
    #     cls.model, cls.train_preprocess, cls.val_preprocess = open_clip.create_model_and_transforms(
    #         model_name, pretrained=pretrained, cache_dir=args.openclip_cachedir
    #     )
    #     cls.model.load_from_state_dict(state_dict)

    # NOTE this and the following base result in the same percentage of frozen params for each layer
    # TODO eval what is the best approach and keep only one
    # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#pruning-a-module
    # def pick_params_to_prune_by_layers(self, pct: float):
    #     for name, module in self.named_modules():
    #         if not hasattr(module, "weight"):
    #             continue

    #         if not any(substring in name for substring in self.MODULE_NAMES_ELIGIBLE_FOR_FREEZING):
    #             continue

    #         prune.random_unstructured(module, name="weight", amount=pct)

    # # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#global-pruning
    # def pick_params_to_prune_by_nn(self, pct: float):
    #     modules_to_freeze = []

    #     for name, module in self.named_modules():
    #         if not hasattr(module, "weight"):
    #             continue

    #         if not any(substring in name for substring in self.MODULE_NAMES_ELIGIBLE_FOR_FREEZING):
    #             continue

    #         modules_to_freeze.append((module, "weight"))

    #     prune.global_unstructured(
    #         modules_to_freeze,
    #         pruning_method=prune.RandomUnstructured,
    #         amount=pct,
    #     )

    # # NOTE alternative name: remove_pruning_metadata
    # def make_pruning_effective(self):
    #     for name, module in self.named_modules():
    #         if not hasattr(module, "weight"):
    #             continue

    #         if not any(substring in name for substring in self.MODULE_NAMES_ELIGIBLE_FOR_FREEZING):
    #             continue

    #         prune.remove(module, "weight")


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, input_size=None, num_classes=None, weights=None, biases=None, **kwargs):
        assert (input_size is not None and num_classes is not None) or weights is not None

        if weights is not None:
            input_size, num_classes = weights.shape

        super().__init__(input_size, num_classes)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving classification head to {filename}")
        torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading classification head from {filename}")
        return torch_load(filename)
