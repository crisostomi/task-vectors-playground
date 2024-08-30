import numpy as np

def model_breadcrumbs(task_vectors,beta=0.85, gamma=0.993):
    pruned_task_vectors = []
    
    for task_vector in task_vectors:
        # Step 1: Rank TV weights by magnitude
        sorted_indices = np.argsort(np.abs(task_vector))
        num_params = len(task_vector)
        
        # Step 2: Prune the left β% and right γ% of the ranked weights
        beta_cutoff = int(beta * num_params)
        gamma_cutoff = int((1 - gamma) * num_params)
        
        pruned_vector = np.zeros_like(task_vector)
        pruned_vector[sorted_indices[beta_cutoff:gamma_cutoff]] = task_vector[sorted_indices[beta_cutoff:gamma_cutoff]]
        
        pruned_task_vectors.append(pruned_vector)
    
    pruned_task_vectors = np.array(pruned_task_vectors)
    return pruned_task_vectors


"""task_vectors = np.array([
    [0.2, 0.1, -0.4, 0.3],
    [-0.1, 0.4, -0.2, 0.1],
    [0.3, -0.3, 0.2, -0.4]
])
beta = 0.1  # Prune the bottom 10% of weights
gamma = 0.1  # Prune the top 10% of weights
strength_coeff = 0.5  # Apply a strength coefficient

pruned_vectors = model_breadcrumbs(task_vectors, beta, gamma, strength_coeff)
print("Pruned Task Vectors:\n", pruned_vectors)"""
