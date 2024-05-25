import copy
from typing import List

import torch
from tvp.task_vectors.aggregation import slerp, spherical_weighted_average


class Aggregator:

    def __init__(self):
        pass

    def __call__(self, task_vectors):
        return self.aggregate(task_vectors)

    def aggregate(self, task_vectors):
        pass


class SphericalAggregator(Aggregator):

    def __init__(self):
        super(SphericalAggregator, self).__init__()

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

    def __init__(self):
        super(SlerpAggregator, self).__init__()

    def aggregate(self, task_vectors, weight=0.5):

        assert len(task_vectors) == 2

        multi_task_vector = slerp(0.5, task_vectors[0], task_vectors[1]).cpu()

        return multi_task_vector


class SumAggregator(Aggregator):

    def __init__(self, mean=False, rescaling=1.0):
        super(SumAggregator, self).__init__()

        self.mean = mean
        self.rescaling = rescaling

    def aggregate(self, task_vectors):
        if isinstance(task_vectors, List):
            task_vectors = torch.stack(task_vectors)

        multi_task_vector = torch.sum(task_vectors, dim=0)

        if self.mean:
            multi_task_vector /= len(task_vectors)

        return multi_task_vector * self.rescaling
