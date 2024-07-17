import open_clip
import torch
from torch.nn.utils import prune
import numpy as np

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

        self.pretrained_state_dict = self.get_pretrained_weights()
        self.cache_dir = cache_dir
        self.tv_mask = None

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
    
    
    def get_pretrained_weights(self):
        return {name: param.detach().clone() for name, param in self.model.named_parameters()}

    def save_pretrained_weights_to_file(self, filename):
        torch.save(self.pretrained_state_dict, filename)
        print(f"Pretrained weights saved to {filename}")

    def reset_weights_by_thresh(self, threshold=1e-4):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.pretrained_state_dict:
                    change_magnitudes = torch.abs(param - self.pretrained_state_dict[name].to(param.device))
                    mask = change_magnitudes <= threshold
                    param[mask] = self.pretrained_state_dict[name].to(param.device)[mask].clone()

    def compute_threshold_by_percentile(self, percentile):
        changes = []
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.pretrained_state_dict:
                    change_magnitude = torch.abs(param - self.pretrained_state_dict[name].to(param.device))
                    changes.append(change_magnitude.view(-1))
        changes = torch.cat(changes)
        threshold = np.percentile(changes.cpu().numpy(), 100 - percentile * 100)
        return threshold

    def reset_weights_by_percentile(self, percentile):
        threshold = self.compute_threshold_by_percentile(percentile)
        self.reset_weights_by_thresh(threshold)


    def get_tv_sparsity(self):
        current_state_dict = self.model.state_dict()
        total_params = 0
        unchanged_params = 0

        for name, current_param in current_state_dict.items():
            pretrained_param = self.pretrained_state_dict[name].to(current_param.device)
            total_params += current_param.numel()
            unchanged_params += torch.sum(current_param == pretrained_param).item()

        unchanged_percentage = (unchanged_params / total_params) * 100
        #print(f"Percentage of parameters that remained pretrained: {unchanged_percentage:.2f}%")
        return 100-unchanged_percentage
    
    def get_tv_sparsity_mask(self):
        mask = {}
        current_state_dict = self.model.state_dict()
        
        for name, current_param in current_state_dict.items():
            pretrained_param = self.pretrained_state_dict[name].to(current_param.device)
            mask[name] = (current_param != pretrained_param).int()  # 0 if same, 1 if different
        return mask
    
    def create_tv_mask(self):
        self.tv_mask = self.get_tv_sparsity_mask()



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
            num_classes, input_size = weights.shape

        super().__init__(in_features=input_size, out_features=num_classes)
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
