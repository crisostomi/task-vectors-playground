import torch


class TaskVector(torch.Tensor):

    def __new__(cls, vector, reference_model=None):
        return super().__new__(cls, vector)

    def __init__(self, vector, reference_model=None):
        # No need to call the base class __init__ method
        self.reference_model = reference_model

    @staticmethod
    def from_models(pretrained_model, finetuned_model):
        """Create a task vector from a pretrained model and a finetuned one."""
        with torch.no_grad():
            pretrained_state_dict = pretrained_model.state_dict()
            finetuned_state_dict = finetuned_model.state_dict()

            parameters = []
            for key in pretrained_state_dict:
                # batchnorm statistics and similar information
                if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                    print(f"Skipping key {key}")
                    continue
                difference = finetuned_state_dict[key] - pretrained_state_dict[key]
                difference = difference.flatten()
                parameters.append(difference)

        return TaskVector(torch.cat(parameters), reference_model=pretrained_model)

    def to_parameters(self):
        return torch.nn.utils.vector_to_parameters(self, self.reference_model.parameters())

    def apply_to(self, target_model, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            new_state_dict = {}

            state_dict = self.to_state_dict()
            target_model_state_dict = target_model.state_dict()

            for key in state_dict:
                if key not in self.vector:
                    print(f"Warning: key {key} is present in the pretrained state dict but not in the task vector")
                    continue

                new_state_dict[key] = target_model_state_dict[key] + scaling_coef * self.vector[key]

        target_model.load_state_dict(new_state_dict, strict=False)

        return target_model

    def to_state_dict(self):
        """
        Convert a flattened parameter vector into a state_dict for the model.
        """
        state_dict = self.reference_model.state_dict()

        pointer = 0
        for name, param in state_dict.items():
            num_param = param.numel()  # Number of elements in the parameter

            # Replace the original parameter with the corresponding part of the vector
            state_dict[name].copy_(self[pointer : pointer + num_param].view_as(param))

            pointer += num_param

        return state_dict
