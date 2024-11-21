import copy
from typing import List

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from tvp.task_vectors.aggregation import slerp, spherical_weighted_average
from tvp.task_vectors.task_singular_vectors import compute_and_sum_svd_mem_reduction


class Aggregator:
    def __init__(self, **kwargs):
        pass

    def __call__(self, task_vectors):
        return self.aggregate(task_vectors)

    def aggregate(self, task_vectors):
        pass


class SphericalAggregator(Aggregator):
    def __init__(self, **kwargs):
        super(SphericalAggregator, self).__init__(**kwargs)

    def aggregate(self, task_vectors, weights=None):
        if isinstance(task_vectors, List):
            task_vectors = torch.stack(task_vectors)

        if weights is None:
            weights = torch.full(size=(len(task_vectors),), fill_value=1 / len(task_vectors)).cuda()

        multi_task_vector = spherical_weighted_average(
            copy.deepcopy(task_vectors), weights, tol=1e-7, max_iter=200, dim=task_vectors.shape[-1] - 1
        )

        return multi_task_vector.cpu()


class SlerpAggregator(Aggregator):
    def __init__(self, **kwargs):
        super(SlerpAggregator, self).__init__(**kwargs)

    def aggregate(self, task_vectors, weight=0.5):
        assert len(task_vectors) == 2

        multi_task_vector = slerp(0.5, task_vectors[0], task_vectors[1]).cpu()

        return multi_task_vector


class SumAggregator(Aggregator):
    def __init__(self, mean=False, rescaling=1.0, **kwargs):
        super(SumAggregator, self).__init__(**kwargs)

        self.mean = mean
        self.rescaling = rescaling

    def aggregate(self, task_vectors):
        if isinstance(task_vectors, List):
            task_vectors = torch.stack(task_vectors)

        multi_task_vector = torch.sum(task_vectors, dim=0)

        if self.mean:
            multi_task_vector /= len(task_vectors)

        return multi_task_vector * self.rescaling


class TaskSingularVectorAggregator(Aggregator):
    def __init__(self, zeroshot_model):
        super().__init__()

        self.zeroshot_model = zeroshot_model

    def aggregate(self, task_vectors):
        if isinstance(task_vectors, torch.Tensor):
            task_vectors = list(task_vectors)

        delta_models = []
        for task_vector in task_vectors:
            delta_model = copy.deepcopy(self.zeroshot_model)
            vector_to_parameters(task_vector, delta_model.parameters())

            delta_models.append(delta_model)

        delta_aggregated_state_dict = compute_and_sum_svd_mem_reduction(delta_models)

        delta_model = copy.deepcopy(self.zeroshot_model)
        delta_model.load_state_dict(delta_aggregated_state_dict)

        delta_aggregated_vector = parameters_to_vector(delta_model.parameters())

        return delta_aggregated_vector
