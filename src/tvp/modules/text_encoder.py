import torch

from transformers import AutoModel


class TextEncoder(torch.nn.Module):

    def __init__(self, model_name: str, cache_dir: str="cache", **kwargs):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_name,
            cache_dir=cache_dir
        )

    
    def forward(self, input_ids, attention_mask):
        
        return self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    def __call__(self, input_ids, attention_mask):
        return self.forward(input_ids, attention_mask)


# utility method to quickly upload the pre-trained text encoder to wandb
def upload_pt_artifact_to_wandb():

    import wandb

    from tvp.utils.io_utils import get_class

    model = TextEncoder(
        model_name="roberta-base",
        cache_dir="cache"
    )

    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

    from src.scripts.finetune import upload_model_to_wandb

    upload_model_to_wandb(
        model=model,
        artifact_name="RoBERTa-base_pt",
        run=wandb.init(
            project="higher-order-task-arithmetics",
            entity="dansolombrinoandfriends",
        ),
        cfg=None,
        metadata={
            "model_name": "RoBERTa-base", 
            "model_class": "tvp.modules.text_encoder.TextEncoder"
        }
    )


if __name__ == "__main__":
    upload_pt_artifact_to_wandb()

    pass