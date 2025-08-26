import numpy as np
import torch
from scipy.optimize import minimize
import warnings
import pandas as pd
import sklearn
import torch.optim as optim


warnings.filterwarnings('ignore')


# Precompute adjacent mutants once
def precompute_adjacency(n_genotypes):
    """Precompute all adjacent mutants (Hamming distance 1)"""
    N = int(np.log2(n_genotypes))
    adjacency = {}

    for i in range(n_genotypes):
        adj_mut = [i ^ (1 << m) for m in range(N)]
        adj_mut.append(i)  # Include self
        adjacency[i] = adj_mut

    return adjacency


# Global variables for caching
_adjacency_cache = None
_last_n_genotypes = None


def get_adjacency(n_genotypes):
    """Get adjacency with caching"""
    global _adjacency_cache, _last_n_genotypes

    if _adjacency_cache is None or _last_n_genotypes != n_genotypes:
        _adjacency_cache = precompute_adjacency(n_genotypes)
        _last_n_genotypes = n_genotypes

    return _adjacency_cache


def compute_stationary_distribution_fast(policy_matrix, fitness_matrix):
    """
    Ultra-fast stationary distribution computation - no matrix operations needed!
    """
    n_drugs, n_genotypes = policy_matrix.shape
    adjacency = get_adjacency(n_genotypes)

    # Build transition destination mapping directly (no matrix needed)
    transition_dest = np.zeros(n_genotypes, dtype=int)

    for i in range(n_genotypes):
        adj_mut = adjacency[i]

        # Vectorized fitness computation - this is the key bottleneck
        # Pre-multiply policy and fitness for this genotype
        policy_i = policy_matrix[:, i]
        fitness_vals = fitness_matrix[:, adj_mut]  # shape: (n_drugs, len(adj_mut))

        # Single matrix multiplication instead of loop
        fitness_weighted = policy_i @ fitness_vals

        # Find the fittest mutant
        fittest_idx = np.argmax(fitness_weighted)
        transition_dest[i] = adj_mut[fittest_idx]

    # Power method without matrix - just array indexing
    x = np.ones(n_genotypes) / n_genotypes  # Start uniform

    for iteration in range(30):  # Reduced iterations
        x_new = np.zeros(n_genotypes)

        # Apply transition: x_new[j] += x[i] for all i that transition to j
        for i in range(n_genotypes):
            x_new[transition_dest[i]] += x[i]

        # Check convergence before normalization (saves computation)
        if iteration > 5 and np.allclose(x, x_new, rtol=1e-6):  # Relaxed tolerance
            break

        x = x_new
    dist = x / np.sum(x)


    return dist

def compute_stationary_dist(policy_matrix, fitness_matrix, final_computation_step = False):


    n_drugs, n_genotypes = policy_matrix.shape



    transition_matrices = get_transition_matrices_sella_hirsh(fitness_matrix)

    TM_net = np.zeros((n_genotypes, n_genotypes))

    for drug in range(n_drugs):
        TM = transition_matrices[drug]
        for i in range(n_genotypes):
            for j in range(n_genotypes):
                TM_net[i, j] += policy_matrix[drug, i] * TM[i , j]


    for i in range(len(TM_net)):
        TM_net[i, :] = TM_net[i, :] / np.sum(TM_net[i, :])
    eigvals, eigvecs = np.linalg.eig(TM_net.T)
    stationary_indices = np.where(np.isclose(eigvals, 1))
    if len(stationary_indices[0]) != 1:
        print("More than one stationary distribution or no stationary distribution found")
        return np.absolute(eigvecs[stationary_indices[0][np.random.randint(2)]]).flatten() / np.sum(np.absolute(eigvecs[stationary_indices[0][np.random.randint(2)]]))

    if len(stationary_indices[0]) >= 2 and final_computation_step:
        return None

    return np.absolute(eigvecs[stationary_indices]).flatten()/ np.sum(np.absolute(eigvecs[stationary_indices]))



def get_transition_matrices_sswm(fitness_matrix):
    num_drugs, num_genotypes = fitness_matrix.shape
    N = int(np.log2(num_genotypes))
    # print(N)
    mut = range(N)  # Creates a list (0, 1, ..., N) to call for bitshifting mutations.
    TMs = []
    for drug in range(num_drugs):
        TM = np.zeros((2 ** N,
                                2 ** N))  # Transition matrix will be sparse (most genotypes unaccessible in one step) so initializes a TM with mostly 0s to do most work for us.

        for i in range(2 ** N):
            adjMut = [i ^ (1 << m) for m in
                      mut]  # For the current genotype i, creates list of genotypes that are 1 mutation away.

            adjFit = np.array([fitness_matrix[drug, j] for j in
                               adjMut])  # Creates list of fitnesses for each corresponding genotype that is 1 mutation away.

            fittest = adjMut[np.argmax(adjFit)]  # Find the most fit mutation
            if fitness_matrix[drug, fittest] < fitness_matrix[
                drug, i]:  # If the most fit mutation is less fit than the current genotype, stay in the current genotype.
                TM[i, i] = 1
            else:
                TM[i, fittest] = 1

        TMs.append(TM)

    return np.array(TMs)

def get_transition_matrices_sella_hirsh(fitness_matrix, cell_type = "haploid", mutation_rate = 1e-4, N = 1e4):
    """
    This gets transition matrices for the Sella-Hirsh model. (Sella and Hirsh, 2005).
    NOTE: This only works if no two fitness values are equal (division by zero)
    and if no fitness value is zero (division by zero).
    Args:
        fitness_matrix:
        cell_type: "haploid" or "diploid"
        mutation_rate: mutation rate (default 1e-4)
        N: population size (default 1e4)
    Returns:

    """
    if cell_type == "haploid":
        a = 2
    else:
        a = 1
    num_drugs, num_genotypes = fitness_matrix.shape

    TMs = []
    N = int(np.log2(num_genotypes))
    # print(N)

    mut = range(N)
    for drug in range(num_drugs):
        TM = np.zeros((num_genotypes, num_genotypes))
        for i in range(num_genotypes):
            adjMut = [i ^ (1 << m) for m in
                      mut]
            for j in adjMut: #mutation from i to j
                f_i = fitness_matrix[drug, i]
                f_j = fitness_matrix[drug, j]
                TM[i, j] = 2/a * N * mutation_rate * (1-(f_i/f_j)**a)/(1 - (f_i/f_j)**(2*N))

            TM[i, i] = 1 - np.sum(TM[i, :])  # Ensure row sums to 1
        TMs.append(TM)

    return np.array(TMs)





def compute_all_metrics_fast(policy_matrix, fitness_matrix):
    """
    Compute all metrics with corrected calculations
    """
    # Compute stationary distribution

    if type(policy_matrix) is not np.ndarray:
        policy_matrix_np = policy_matrix.detach().numpy()
    else:
        policy_matrix_np = policy_matrix

    stationary_dist = compute_stationary_dist(policy_matrix_np, fitness_matrix)

    # print(stationary_dist)
    # Compute marginal drug probabilities: P(drug) = sum over genotypes of P(drug|genotype) * P(genotype)
    drug_probs = np.sum(policy_matrix_np * stationary_dist[np.newaxis, :], axis=1)

    # CORRECTED: Mean fitness calculation
    # Mean fitness should be the expected fitness under the policy:
    # E[fitness] = sum_{genotype, drug} P(genotype) * P(drug|genotype) * fitness(drug, genotype)
    mean_fit = 0.0
    for i in range(policy_matrix_np.shape[1]):  # For each genotype
        for d in range(policy_matrix_np.shape[0]):  # For each drug
            mean_fit += stationary_dist[i] * policy_matrix_np[d, i] * fitness_matrix[d, i]

    # CORRECTED: Mutual information calculation
    # MI = sum_{d,g} P(d,g) * log(P(d,g) / (P(d) * P(g)))
    # where P(d,g) = P(g) * P(d|g) = stationary_dist[g] * policy_matrix[d,g]
    mutual_info = 0.0
    for i in range(policy_matrix_np.shape[1]):  # For each genotype
        if stationary_dist[i] > 1e-12:  # Only if genotype has significant probability
            for d in range(policy_matrix_np.shape[0]):  # For each drug
                p_dg = stationary_dist[i] * policy_matrix_np[d, i]  # P(drug, genotype)
                p_d = drug_probs[d]  # P(drug)
                p_g = stationary_dist[i]  # P(genotype)

                if p_dg > 1e-12 and p_d > 1e-12:  # Avoid log(0)
                    mutual_info += p_dg * np.log(p_dg / (p_d * p_g))

    return stationary_dist, drug_probs, mean_fit, mutual_info


def lagrangian_objective_fast(policy_flat, fitness_matrix, lagrange_multiplier):
    """
    Objective function for Lagrange multiplier optimization.
    CORRECTED: We want to MINIMIZE fitness (better treatment) while controlling MI
    So we minimize: Mean_Fitness + λ * Mutual_Information
    """
    n_drugs, n_genotypes = fitness_matrix.shape
    policy_matrix = policy_flat.reshape(n_drugs, n_genotypes)

    # Compute all metrics in one pass
    metrics = compute_all_metrics_fast(policy_matrix, fitness_matrix)

    _, _, mean_fit, mutual_info = metrics

    # Minimize fitness + λ * mutual_information
    return mean_fit + lagrange_multiplier * mutual_info


def optimize_policy_lagrange_fast(fitness_matrix, lagrange_multiplier,
                                  method='SLSQP', max_restarts=1):
    """
    Streamlined fast policy optimization
    """
    n_drugs, n_genotypes = fitness_matrix.shape

    # Simple uniform initialization (faster than random)
    initial_policy = np.ones((n_drugs, n_genotypes)) / n_drugs
    initial_flat = initial_policy.flatten()

    # Simpler bounds
    eps = 1e-10
    bounds = [(eps, 1.0)] * len(initial_flat)

    # Fixed constraints using correct indexing
    constraints = []
    for j in range(n_genotypes):  # For each genotype
        indices = [i * n_genotypes + j for i in range(n_drugs)]

        def make_constraint(idx_list):
            def constraint_func(x):
                return np.sum(x[idx_list]) - 1.0

            return constraint_func

        constraints.append({
            'type': 'eq',
            'fun': make_constraint(indices)
        })

    # Single optimization attempt with good settings

    # Implement SGD

    random_policy = np.random.random((n_drugs, n_genotypes))
    random_policy = random_policy / np.sum(random_policy, axis=0, keepdims=True)  # Normalize

    # result = sgd_optimize_policy(lagrangian_objective_fast, random_policy, params = (fitness_matrix, lagrange_multiplier)) #sgd optimize policy returns a policy matrix





    result = minimize(
        fun=lambda x: lagrangian_objective_fast(x, fitness_matrix, lagrange_multiplier),
        x0=initial_flat,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={
            'maxiter': 500,  # Reduced iterations
            'ftol': 1e-6,  # Relaxed tolerance
            'disp': False
        }
    )

    # Extract and normalize policy
    optimal_policy = result.x.reshape(n_drugs, n_genotypes)
    # Ensure normalization (handle any small numerical errors)
    optimal_policy = np.maximum(optimal_policy, eps)  # Ensure positive
    optimal_policy = optimal_policy / np.sum(optimal_policy, axis=0, keepdims=True)

    # Compute final metrics
    stationary_dist, drug_probs, mean_fit, mutual_info = compute_all_metrics_fast(
        optimal_policy, fitness_matrix
    )


    optim_success = True
    if compute_stationary_dist(optimal_policy, fitness_matrix, final_computation_step=True) is None:
         optim_success = False

    return {
        'optimal_policy': optimal_policy,
        'stationary_distribution': stationary_dist,
        'drug_probabilities': drug_probs,
        'mean_fitness': mean_fit,
        'mutual_information': mutual_info,
        'lagrange_multiplier': lagrange_multiplier,
        'optimization_result': result,
        'success': (result.success and optim_success)  # More lenient success criteria
    }

# def sgd_optimize_policy(fun, init_policy, params = None, lr = 0.001):
#     """
#     Implements Stochastic Gradient Descent (SGD) optimization because I was too lazy to read the documentation
#     Args:
#         fun: function to be optimized. It is assumed that params come after the optimizer variable
#         params: additional params that come after the optimizer variable
#         lr: learning rate
#
#     Returns: optimized result for policy matrix
#
#     """
#
#     # Initialize policy
#     policy = init_policy.copy()
#
#
#     for _ in range(500):  # Reduced iterations for faster convergence
#         #find gradient of fun with respect to policy
#
#         gradient_matrix = np.zeros(policy.shape)
#
#
#         surround_range = 1e-7
#
#         for i in range(policy.shape[0]):
#             for j in range(policy.shape[1]):
#
#                 var_policy = policy.copy()
#
#                 var_policy[i, j] += np.random.uniform(-surround_range, surround_range)
#
#                 perturbed_policy_low = var_policy.copy()
#                 perturbed_policy_high = var_policy.copy()
#
#                 perturbed_policy_low[i, j] -= 1e-6
#                 perturbed_policy_high[i, j] += 1e-6
#                 perturbed_policy_low = np.maximum(perturbed_policy_low, 1e-10)  # Ensure positive
#
#                 gradient = ((fun(perturbed_policy_high, *params) - fun(var_policy, *params)) / (1e-6) + (fun(var_policy, *params) - fun(perturbed_policy_low, *params)) / (1e-6))/2  # Central difference approximation
#
#                 gradient_matrix[i, j] = gradient
#
#
#         # Update policy using gradient descent
#         policy -= lr * gradient_matrix
#
#
#     return policy




    # # Convert to torch tensor for autograd
    # policy_tensor = torch.tensor(policy, requires_grad=True)
    #
    # # Optimizer
    # optimizer = optim.SGD([policy_tensor], lr=lr)
    #
    # # Optimization loop
    # for _ in range(100):  # Reduced iterations
    #     optimizer.zero_grad()
    #
    #     # Compute objective function
    #     loss = fun(policy_tensor, *params)  # Pass additional params if needed
    #
    #     # Backpropagation
    #     loss.backward()
    #     optimizer.step()
    #
    #     # Normalize policy after each step
    #     with torch.no_grad():
    #         policy_tensor.data = torch.clamp(policy_tensor.data, min=1e-10, max=1.0)
    #         policy_tensor.data /= policy_tensor.data.sum(dim=0, keepdim=True)
    #
    # return policy_tensor.detach().numpy()




def sweep_lagrange_multipliers_fast(fitness_matrix, lambda_values,
                                    warm_start=True, max_restarts=1):
    """
    Fast sweep with optional warm starting
    """
    results = []

    print(f"Sweeping {len(lambda_values)} Lagrange multiplier values (fast version)...")

    for i, lam in enumerate(lambda_values):
        print(f"λ = {lam:.4f} ({i + 1}/{len(lambda_values)})", end=" ")

        result = optimize_policy_lagrange_fast(
            fitness_matrix, lam, max_restarts=max_restarts
        )
        results.append(result)

        print(f"Fit: {result['mean_fitness']:.4f}, MI: {result['mutual_information']:.4f}, "
              f"Success: {result['success']}")

    return results


def define_mira_landscapes(as_dict=False):
    '''
    Function to define the landscapes described in
    Mira PM, Crona K, Greene D, Meza JC, Sturmfels B, Barlow M (2015)
    '''
    drugs = {}
    drugs['AMP'] = [1.851, 2.082, 1.948, 2.434, 2.024, 2.198, 2.033, 0.034, 1.57, 2.165, 0.051, 0.083, 2.186, 2.322,
                    0.088, 2.821]
    drugs['AM'] = [1.778, 1.782, 2.042, 1.752, 1.448, 1.544, 1.184, 0.063, 1.72, 2.008, 1.799, 2.005, 1.557, 2.247,
                   1.768, 2.047]
    drugs['CEC'] = [2.258, 1.996, 2.151, 2.648, 2.396, 1.846, 2.23, 0.214, 0.234, 0.172, 2.242, 0.093, 2.15, 0.095,
                    2.64, 0.516]
    drugs['CTX'] = [0.16, 0.085, 1.936, 2.348, 1.653, 0.138, 2.295, 2.269, 0.185, 0.14, 1.969, 0.203, 0.225, 0.092,
                    0.119, 2.412]
    drugs['ZOX'] = [0.993, 0.805, 2.069, 2.683, 1.698, 2.01, 2.138, 2.688, 1.106, 1.171, 1.894, 0.681, 1.116, 1.105,
                    1.103, 2.591]
    drugs['CXM'] = [1.748, 1.7, 2.07, 1.938, 2.94, 2.173, 2.918, 3.272, 0.423, 1.578, 1.911, 2.754, 2.024, 1.678,
                    1.591, 2.923]
    drugs['CRO'] = [1.092, 0.287, 2.554, 3.042, 2.88, 0.656, 2.732, 0.436, 0.83, 0.54, 3.173, 1.153, 1.407, 0.751,
                    2.74, 3.227]
    drugs['AMC'] = [1.435, 1.573, 1.061, 1.457, 1.672, 1.625, 0.073, 0.068, 1.417, 1.351, 1.538, 1.59, 1.377, 1.914,
                    1.307, 1.728]
    drugs['CAZ'] = [2.134, 2.656, 2.618, 2.688, 2.042, 2.756, 2.924, 0.251, 0.288, 0.576, 1.604, 1.378, 2.63, 2.677,
                    2.893, 2.563]
    drugs['CTT'] = [2.125, 1.922, 2.804, 0.588, 3.291, 2.888, 3.082, 3.508, 3.238, 2.966, 2.883, 0.89, 0.546, 3.181,
                    3.193, 2.543]
    drugs['SAM'] = [1.879, 2.533, 0.133, 0.094, 2.456, 2.437, 0.083, 0.091, 2.198, 2.57, 2.308, 2.886, 2.504, 3.002,
                    2.528, 3.453]
    drugs['CPR'] = [1.743, 1.662, 1.763, 1.785, 2.018, 2.05, 2.042, 0.218, 1.553, 0.256, 0.165, 0.221, 0.223, 0.239,
                    1.811, 0.288]
    drugs['CPD'] = [0.595, 0.245, 2.604, 3.043, 1.761, 1.471, 2.91, 3.096, 0.432, 0.388, 2.651, 1.103, 0.638, 0.986,
                    0.963, 3.268]
    drugs['TZP'] = [2.679, 2.906, 2.427, 0.141, 3.038, 3.309, 2.528, 0.143, 2.709, 2.5, 0.172, 0.093, 2.453, 2.739,
                    0.609, 0.171]
    drugs['FEP'] = [2.59, 2.572, 2.393, 2.832, 2.44, 2.808, 2.652, 0.611, 2.067, 2.446, 2.957, 2.633, 2.735, 2.863,
                    2.796, 3.203]
    return drugs if as_dict else np.array(list(drugs.values()))


def define_simple_landscapes():
    drugs = np.array([
        [2.0, 1.0, 1.5, 0.5],
        [3.1, 2.2, 0.2, 0.1],
        [0.6, 0.1, 0.5, 4.0],
        # [1.5, 1.6, 0.5, 0.6]
                     ])
    return drugs


def define_basic_landscapes():
    drugs = np.array([
        [2.0, 1.0, 1.5, 0.5],
        [3.1, 2.2, 0.2, 0.1],
        [0.6, 0.1, 0.5, 4.0],

        # [2.5, 0.5],

    ])
    return drugs

def define_random_landscapes(drugs, genotypes):
    return np.random.random((drugs, genotypes)) * 4 + 0.1  # Random fitness values between 0.1 and 10

# Test the optimized version
def solve_pareto_frontier(fitness_landscape, plot_results=True, plot_random = True):
    # Load fitness landscapes
    fitness_matrix = fitness_landscape
    print(f"Loaded fitness matrix: {fitness_matrix.shape[0]} drugs, {fitness_matrix.shape[1]} genotypes")

    print(f"Fitness matrix:\n{fitness_matrix}")

    # Use same lambda range for comparison
    lambda_range = np.arange(-2, 2, 0.01)

    # lambda_range = np.linspace(0, 2, 101)
    #eigenvalue 1

    print(f"\n{'=' * 50}")
    print("FAST PARETO FRONT ANALYSIS")
    print(f"{'=' * 50}")

    import time

    start_time = time.time()

    pareto_results = sweep_lagrange_multipliers_fast(fitness_matrix, lambda_range)

    end_time = time.time()
    print(f"\nTotal time: {end_time - start_time:.2f} seconds")

    # Summary
    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    print(f"{'Lambda':<12} {'Mean Fit':<12} {'Mut Info':<12} {'Success':<8}")
    print("-" * 50)

    for i, result in enumerate(pareto_results):
        print(f"{lambda_range[i]:<12.6f} {result['mean_fitness']:<12.6f} "
              f"{result['mutual_information']:<12.6f} {result['success']}")

    # Plot mutual information vs mean fitness (mean fitness on y-axis, mutual information on x-axis)
    import matplotlib.pyplot as plt

    # Extract data for plotting
    mutual_infos = [r['mutual_information'] for r in pareto_results]
    mean_fitnesses = [r['mean_fitness'] for r in pareto_results]
    lambdas = [r['lagrange_multiplier'] for r in pareto_results]
    success_flags = [r['success'] for r in pareto_results]

    # Plot successful results in blue, failed in red
    successful_mi = [mi for mi, success in zip(mutual_infos, success_flags) if success]
    successful_mf = [mf for mf, success in zip(mean_fitnesses, success_flags) if success]
    failed_mi = [mi for mi, success in zip(mutual_infos, success_flags) if not success]
    failed_mf = [mf for mf, success in zip(mean_fitnesses, success_flags) if not success]



    if plot_results:
        # Create the plot
        plt.figure(figsize=(10, 8))



        # Plot points
        if successful_mi:
            plt.scatter(successful_mi, successful_mf, c='blue', s=60, alpha=0.7,
                        label='Successful optimization', edgecolors='darkblue', linewidth=1)
        # if failed_mi:
        #     plt.scatter(failed_mi, failed_mf, c='red', s=60, alpha=0.7,
        #                 label='Failed optimization', marker='x', linewidth=2)

        if plot_random:
            for i in range(2000):
                random_policy = np.random.random(fitness_matrix.shape)
                for col in range(random_policy.shape[1]):
                    random_policy[:, col] /= np.sum(random_policy[:, col])

                _, __, random_mf, random_mi = compute_all_metrics_fast(random_policy, fitness_matrix)
                plt.scatter(random_mi, random_mf, c='grey', s=20, alpha=0.3, marker='o', label="Random policy" if i == 0 else "")


        # Connect successful points to show Pareto front
        if len(successful_mi) > 1:
            # Sort by mutual information for connecting line
            sorted_pairs = sorted(zip(successful_mi, successful_mf))
            sorted_mi, sorted_mf = zip(*sorted_pairs)
            plt.plot(sorted_mi, sorted_mf, 'b--', alpha=0.5, linewidth=1, label='Pareto front')

        # Add lambda value annotations for some points
        for i, (mi, mf, lam, success) in enumerate(zip(mutual_infos, mean_fitnesses, lambdas, success_flags)):
            if success and i % 50 == 0:  # Annotate every 500th successful point to avoid clutter
                plt.annotate(f'λ={lam:.2f}', (mi, mf), xytext=(5, 5),
                             textcoords='offset points', fontsize=8, alpha=0.7)

        plt.xlabel('Mutual Information I(Drug; Genotype)', fontsize=12, fontweight='bold')
        plt.ylabel('Mean Fitness', fontsize=12, fontweight='bold')
        plt.title(
            'Pareto Front: Trade-off between Mean Fitness and Mutual Information\n(Lower fitness = better resistance suppression)',
            fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)

        # Add text box with interpretation
        textstr = ('Higher λ → Higher mutual information\n'
                   'Lower mutual information → More random policy\n'
                   'Lower mean fitness → Better treatment')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=9,
                 verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.show()

    smooth_fit = check_smooth_curve(successful_mi, successful_mf)
    print("Smooth curve fit:", smooth_fit)
    # Print some key statistics
    if successful_mi:
        print(f"\n{'=' * 50}")
        print("PARETO FRONT STATISTICS")
        print(f"{'=' * 50}")
        print(f"Mutual Information range: {min(successful_mi):.4f} to {max(successful_mi):.4f}")
        print(f"Mean Fitness range: {min(successful_mf):.4f} to {max(successful_mf):.4f}")
        print(f"Successful optimizations: {sum(success_flags)}/{len(success_flags)}")

        # Find extreme points
        min_mi_idx = successful_mi.index(min(successful_mi))
        max_mi_idx = successful_mi.index(max(successful_mi))
        min_fit_idx = successful_mf.index(min(successful_mf))
        max_fit_idx = successful_mf.index(max(successful_mf))

        print(f"\nExtreme points:")
        print(
            f"Minimum MI: λ={lambdas[min_mi_idx]:.3f}, MI={min(successful_mi):.4f}, Fitness={successful_mf[min_mi_idx]:.4f}")
        print(
            f"Maximum MI: λ={lambdas[max_mi_idx]:.3f}, MI={max(successful_mi):.4f}, Fitness={successful_mf[max_mi_idx]:.4f}")
        print(
            f"Minimum Fitness: λ={lambdas[min_fit_idx]:.3f}, MI={successful_mi[min_fit_idx]:.4f}, Fitness={min(successful_mf):.4f}")
        print(
            f"Maximum Fitness: λ={lambdas[max_fit_idx]:.3f}, MI={successful_mi[max_fit_idx]:.4f}, Fitness={max(successful_mf):.4f}")

    # Save results to CSV
    results_df = pd.DataFrame({'Mutual Info': successful_mi, 'Mean Fitness': successful_mf,})

    results_df.to_csv('pareto_frontier_results.csv', index=False)

    return True


def check_smooth_curve(mutual_info : np.ndarray | list, mean_fitness : np.ndarray | list):
    from numpy.polynomial.polynomial import Polynomial as P
    import matplotlib.pyplot as plt

    fit, results = P.fit(mutual_info, mean_fitness, deg=8, full=True)

    print("Fit coefficients:", fit)
    mse = results[0]
    print("MSE:", results[0])
    # print(fit.)
    # plt.plot(fit.linspace()[0], fit.linspace()[1], 'r-', label='Fitted Curve')
    #
    # plt.plot(mutual_info, mean_fitness, 'bo', label='Pareto Front')
    # # plt.plot(np.arange(0, 1, 100), fit[np.arange(0, 1, 100)], 'r--', label='Pareto Front')
    # plt.legend()
    # plt.show()
    return mse < 0.01



def define_successful_landscapes():
    return np.array([[1.20506869, 3.35382954, 0.32273345, 0.29391833],
     [3.99748972, 0.5007246, 1.94397629, 0.66432379],
     [2.13112861, 0.59541999, 1.32083556, 1.48314829],
     [1.0306329, 3.99639049, 0.70993479, 0.74922469],
     [2.73580711, 0.43070899, 3.9541098, 3.44483566],
     [3.00333663, 0.24205831, 2.57842129, 0.91931369],
     [1.23278399, 2.31578297, 2.80822274, 2.46058061],
     [2.80129708, 0.23133693, 0.33508322, 3.83364791]])



solve_pareto_frontier(define_successful_landscapes())

