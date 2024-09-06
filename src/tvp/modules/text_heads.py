import torch

from torch import Tensor


class TextClassificationHead(torch.nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        
        super().__init__()
        
        self.classification_head: torch.nn.Linear = torch.nn.Linear(
            hidden_size, num_classes
        )

        self.out_features = num_classes

    def forward(self, x: Tensor) -> Tensor:
        return self.classification_head.forward(x)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


def get_classification_head(
    hidden_size: int, 
    num_classes: int
):
    classification_head = TextClassificationHead(hidden_size, num_classes)

    return classification_head
