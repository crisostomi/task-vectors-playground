import torch
import open_clip

from transformers import AutoModel
from tvp.utils import utils


class ClipTextEncoder(torch.nn.Module):

    def __init__(self,  model_name: str, openclip_cachedir=None, cache_dir=None, **kwargs):
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

    def forward(self, input_ids, attention_mask):

        return self.model.encode_text(
            text=input_ids,
        )

    def __call__(self, input_ids, attention_mask):
        return self.forward(input_ids, attention_mask)
    
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


