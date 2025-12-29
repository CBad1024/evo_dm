import numpy as np
import pytest

from evodm.theoretical_model_compute import compute_stationary_dist, get_transition_matrices_sswm, define_mira_landscapes



@pytest.mark.parametrize("policy_matrix, fitness_matrix, expected", [
    (np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([[1, 0], [0, 1]]), np.array([0.5, 0.5])),
    (np.array([[0.25, 0], [0.75, 1]]), np.array([[0.25, 0.75], [0, 1]]), np.array([0., 1])),
    (np.array([[0.25, 0.9], [0.75, 0.1]]), np.array([[0.25, 0.75], [0, 1]]), np.array([0, 1])),


])
def test_compute_stationary_dist(policy_matrix, fitness_matrix, expected):
    stationary_dist = compute_stationary_dist(policy_matrix, fitness_matrix)
    print(stationary_dist)
    assert np.allclose(stationary_dist, expected)

@pytest.mark.parametrize("matrix, expected", [
    (np.array([[1, 0], [0, 1]]), np.array(
                [np.array([[1, 0], [1, 0]]),
                 np.array([[0, 1], [0, 1]])])),

    (np.array([[1, 0], [1, 0]]), np.array(
                [np.array([[1, 0], [1, 0]]),
                np.array([[1, 0], [1, 0]])])),

    (np.array([[0.5, 0], [0.7, 0.2]]), np.array(
                [np.array([[1, 0], [1, 0]]),
                 np.array([[1, 0], [1, 0]])])),

])

def test_get_transition_matrices(matrix, expected):
    m = get_transition_matrices_sswm(matrix)
    assert np.array_equal(m, expected)

@pytest.mark.parametrize("mutual_info, mean_fitness, expected", [
    (np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), True),
    (np.arange(0, 1, 0.01), np.arange(0, 1, 0.01) ** 2, True),
    (np.arange(0, 1, 0.01), np.random.uniform(0, 1, size = (100,)), False),
])

def test_check_smooth_curve(mutual_info, mean_fitness, expected):
    from evodm.theoretical_model_compute import check_smooth_curve
    result = check_smooth_curve(mutual_info, mean_fitness)
    assert result == expected


@pytest.mark.parametrize("num_drugs, num_genotypes, is_smooth_curve", [
    (2, 2, True),
    (2, 4, True),
    (3, 2, False),
    # (4, 2, False),
    # (2, 8, True),
    # (2, 16, True),
    (3, 2, False),
    (3, 2, False),
    (3, 2, False),
    (3, 2, False),

])

def test_solve_pareto_frontier(num_drugs, num_genotypes, is_smooth_curve):
    from evodm.theoretical_model_compute import solve_pareto_frontier
    ls = define_mira_landscapes()
    np.random.shuffle(ls)

    fitness_matrix = ls[:num_drugs, :num_genotypes]

    print(fitness_matrix)
    # random_matrix = np.random.random((num_drugs, num_genotypes))
    # print(random_matrix)
    result = solve_pareto_frontier(fitness_matrix)
    assert result == is_smooth_curve