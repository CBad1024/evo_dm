import pytest
import numpy as np
from evodm.envs.wright_fisher_env import WrightFisherEnv
from evodm.core.landscapes import Seascape
from gymnasium import spaces

@pytest.fixture
def default_wf():
    return WrightFisherEnv(pop_size=1000, seq_length=2, gen_per_step=10, total_generations=50)

def test_init_default_parameters(default_wf):
    assert default_wf.pop_size == 1000
    assert default_wf.seq_length == 2
    assert default_wf.generation == 0
    assert len(default_wf.genotypes) == 4
    assert default_wf.num_drugs == 10 # Default
    assert isinstance(default_wf.observation_space, spaces.Box)
    assert default_wf.observation_space.shape == (4,)
    assert default_wf.action_space.n == 10 # 10 drugs * 1 conc (default seascapes=False)

def test_reset_default_start(default_wf):
    obs, info = default_wf.reset()
    assert default_wf.generation == 0
    assert default_wf.pop == {'00': 1000}
    # Observation is frequencies
    expected_obs = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(obs, expected_obs)

def test_reset_random_start():
    env = WrightFisherEnv(pop_size=1000, seq_length=2, random_start=True)
    genotypes_seen = set()
    for _ in range(20):
        env.reset()
        # Find the genotype with the population
        geno = list(env.pop.keys())[0]
        genotypes_seen.add(geno)
    assert len(genotypes_seen) > 1

def test_step_action_decoding_simple(default_wf):
    default_wf.reset()
    # In simple mode (seascapes=False), current_conc is fixed to 2
    default_wf.step(5)
    assert default_wf.current_drug == 5
    assert default_wf.current_conc == 2

def test_step_action_decoding_seascapes():
    env = WrightFisherEnv(pop_size=1000, seq_length=2, seascapes=True, num_drugs=5)
    # Action = conc * num_drugs + drug
    # If action = 7, drug = 7 % 5 = 2, conc = 7 // 5 = 1
    env.step(7)
    assert env.current_drug == 2
    assert env.current_conc == 1

def test_step_population_dynamics(default_wf):
    default_wf.reset()
    # High mutation rate to ensure we see some changes quickly
    default_wf.mutation_rate = 0.5 
    obs, reward, terminated, truncated, info = default_wf.step(0)
    assert default_wf.generation == 10 # gen_per_step
    # Observation should be frequencies
    assert np.isclose(np.sum(obs), 1.0)
    assert obs.shape == (4,)

def test_mutation_step(default_wf):
    default_wf.reset()
    default_wf.pop = {'00': 1000}
    default_wf.mutation_rate = 1.0 # Very high mutation rate
    default_wf.mutation_step()
    # Population size should be conserved
    assert sum(default_wf.pop.values()) == 1000
    # Should have some mutants
    assert '00' in default_wf.pop
    assert len(default_wf.pop) > 1

def test_offspring_step(default_wf):
    default_wf.reset()
    # state 00, 01, 10, 11
    # Let's say we have equal proportions but fitness is different
    default_wf.pop = {'00': 250, '01': 250, '10': 250, '11': 250}
    fitness = {'00': 1.0, '01': 0.1, '10': 0.1, '11': 0.1}
    default_wf.offspring_step(fitness)
    # 00 is much fitter, should dominate
    assert default_wf.pop.get('00', 0) > 250
    assert sum(default_wf.pop.values()) == 1000

def test_reward_scaling_simple(default_wf):
    default_wf.reset()
    # First step sets initial stats
    _, reward1, _, _, _ = default_wf.step(0)
    # Statistics should update. Reward calculation involves ema of min/max.
    assert -2.0 <= reward1 <= 2.0 # Reasonable range

def test_seascape_mode_reward_and_penalty():
    env = WrightFisherEnv(pop_size=1000, seq_length=2, seascapes=True, num_drugs=5)
    env.reset()
    # Medium concentration (index 2)
    # Action for drug 0, conc 2: 2 * 5 + 0 = 10
    _, reward1, _, _, _ = env.step(10)
    
    # High concentration (index 0)
    # Action for drug 0, conc 0: 0 * 5 + 0 = 0
    env.reset()
    _, reward2, _, _, _ = env.step(0)
    
    # Penalty is -0.06 * log10(conc)
    # concs: [0.1, 0.05, 0.01, ...]
    # log10(0.1) = -1 -> penalty = +0.06
    # log10(0.01) = -2 -> penalty = +0.12
    # So higher concentrations (larger values) have SMALLER penalties (more negative numbers added)
    # Wait, 0.06 * np.log10(conc). 
    # If conc = 0.1, log10 = -1, reward -= -0.06 -> reward += 0.06
    # If conc = 0.0001, log10 = -4, reward -= -0.24 -> reward += 0.24
    # So LOWER concentrations have HIGHER rewards (less drug used).
    assert reward2 > reward1 # conc 0.1 vs conc 0.01

def test_get_fitness(default_wf):
    default_wf.reset()
    default_wf.pop = {'00': 1000}
    fit = default_wf.get_fitness()
    # Should match drug_seascapes[0, 2, 0]
    expected = default_wf.drug_seascapes[0, 2, 0]
    assert fit == pytest.approx(expected)

def test_termination(default_wf):
    default_wf.reset()
    # 50 total generations, 10 per step -> 5 steps
    for _ in range(4):
        _, _, terminated, _, _ = default_wf.step(0)
        assert not terminated
    _, _, terminated, _, _ = default_wf.step(0)
    assert terminated
