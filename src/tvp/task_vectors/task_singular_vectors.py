import copy

import torch
from tqdm import tqdm


def compute_and_sum_svd_mem_reduction(delta_models, device="cuda"):
    """
    Computes the Singular Value Decomposition (SVD) for each vector in the task_vectors,
    reduces the dimensionality of the vectors based on the sv_reduction factor, and concatenate
    the low-rank matrices. If the vector is not a 2D tensor or is "text_projection", it computes the mean of the vectors.
    Computation of the SVD is performed also for the second operation.

    Args:
        task_vectors (list): A list of task vector objects, where each object contains a
                             dictionary of vectors.
        config (object): Configuration object containing the following attributes:
                         - DATASETS (list): List of datasets.
                         - device (torch.device): The device to perform computations on.

    Returns:
        dict: A dictionary containing the new vectors after SVD computation and merging.
    """
    num_tasks = len(delta_models)
    sv_reduction = 1 / num_tasks

    aggregated_model_dict = copy.deepcopy(delta_models[0].state_dict())
    layer_names = list(aggregated_model_dict.keys())

    with torch.no_grad():
        for layer_name in tqdm(layer_names, desc="Computing SVD"):
            is_matrix = aggregated_model_dict[layer_name].dim() == 2

            # low-rank approximation step
            for i, delta_model in enumerate(delta_models):
                delta_layer = delta_model.state_dict()[layer_name].to(device)

                if is_matrix and "text_projection" not in layer_name:
                    u, s, v = torch.linalg.svd(delta_layer, full_matrices=False)

                    if i == 0:
                        print(f"Computed SVD for {layer_name}...")
                        sum_u = torch.zeros_like(u, device=device)
                        sum_s = torch.zeros_like(s, device=device)
                        sum_v = torch.zeros_like(v, device=device)

                    reduced_index_s = int(s.shape[0] * sv_reduction)

                    # select only the first reduced_index_s columns of u and place them
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[:, :reduced_index_s]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[:reduced_index_s]
                    # select only the first reduced_index_s rows of v and place them
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[:reduced_index_s, :]

                else:
                    if i == 0:
                        aggregated_model_dict[layer_name] = delta_layer.clone()
                    else:
                        aggregated_model_dict[layer_name] += (delta_layer - aggregated_model_dict[layer_name]) / (i + 1)

            # aggregation step
            if is_matrix and "text_projection" not in layer_name:
                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

                aggregated_model_dict[layer_name] = torch.linalg.multi_dot(
                    (
                        u_u,
                        v_u,
                        torch.diag(sum_s),
                        u_v,
                        v_v,
                    )
                )

    return aggregated_model_dict
