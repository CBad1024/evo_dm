import pytest
import numpy as np
from evodm.envs.sswm_env import SSWMEnv
from gymnasium import spaces

@pytest.fixture
def default_sswm():
    return SSWMEnv(N=2, total_generations=10)

@pytest.fixture
def custom_landscape():
    # 2^2 = 4 genotypes
    # Drug 0: state 0 is best
    # Drug 1: state 3 is best
    ls = np.array([
        [1.0, 0.5, 0.5, 0.1], # Drug 0
        [0.1, 0.5, 0.5, 1.0]  # Drug 1
    ])
    return ls

def test_init_default_parameters(default_sswm):
    assert default_sswm.N == 2
    assert default_sswm.total_generations == 10
    assert default_sswm.random_start == False
    assert default_sswm.num_drugs == 15 # Default MIRA has 15 drugs
    assert isinstance(default_sswm.observation_space, spaces.Box)
    assert isinstance(default_sswm.action_space, spaces.Discrete)
    assert default_sswm.observation_space.shape == (4,)
    assert default_sswm.action_space.n == 15

def test_init_custom_landscapes(custom_landscape):
    env = SSWMEnv(N=2, landscapes=custom_landscape)
    assert env.num_drugs == 2
    assert env.action_space.n == 2
    np.testing.assert_array_equal(env.landscapes, custom_landscape)

def test_reset_default_start(default_sswm):
    obs, info = default_sswm.reset()
    assert default_sswm.state == 0
    assert default_sswm.generation == 0
    # One-hot encoding for state 0
    expected_obs = np.array([1, 0, 0, 0], dtype=np.float32)
    np.testing.assert_array_equal(obs, expected_obs)
    assert info == {}

def test_reset_random_start():
    # Use a seed for reproducibility in tests if needed, 
    # but here we just check it randomizes
    env = SSWMEnv(N=2, random_start=True)
    states = set()
    for _ in range(20):
        env.reset()
        states.add(env.state)
    assert len(states) > 1

def test_step_action_validation(default_sswm):
    default_sswm.reset()
    with pytest.raises(ValueError):
        default_sswm.step(15) # Only 0-14 valid

def test_step_reward_calculation(custom_landscape):
    env = SSWMEnv(N=2, landscapes=custom_landscape)
    env.reset() # state 0
    # Reward = 1.5 - fitness
    # state 0, drug 0 fitness = 1.0
    obs, reward, terminated, truncated, info = env.step(0)
    assert reward == pytest.approx(1.5 - 1.0)
    assert info['fitness'] == 1.0

def test_step_state_transition_greedy(custom_landscape):
    env = SSWMEnv(N=2, landscapes=custom_landscape)
    # Landscapes:
    # state 0: 1.0, state 1: 0.5, state 2: 0.5, state 3: 0.1 (Drug 0)
    # Adjacent to 0 (binary 00) are 1 (01) and 2 (10) for N=2
    
    # Let's check transition from state 3 (binary 11) for Drug 1
    # Adjacent to 3 (11) are 1 (01) and 2 (10)
    # Drug 1 fitness: state 1 (0.5), state 2 (0.5), state 3 (1.0)
    # 3 is fitter than neighbors, should stay.
    
    env.state = 3
    env.current_drug = 1
    next_state = env.get_next_state(custom_landscape[1], 3)
    assert next_state == 3
    
    # Transition from state 0 (00) for Drug 1
    # Adjacent to 0 are 1 (0.5) and 2 (0.5)
    # Fittest neighbor is 1 or 2. argmax will take first.
    next_state = env.get_next_state(custom_landscape[1], 0)
    assert next_state in [1, 2]

def test_step_termination(default_sswm):
    default_sswm.reset()
    for i in range(9):
        _, _, terminated, _, _ = default_sswm.step(0)
        assert not terminated
    _, _, terminated, _, _ = default_sswm.step(0)
    assert terminated

def test_get_fitness(custom_landscape):
    env = SSWMEnv(N=2, landscapes=custom_landscape)
    env.reset()
    env.state = 3
    env.current_drug = 1
    assert env.get_fitness() == 1.0

def test_episode_complete_trajectory(default_sswm):
    obs, _ = default_sswm.reset()
    total_reward = 0
    for _ in range(10):
        obs, reward, terminated, truncated, info = default_sswm.step(0)
        total_reward += reward
        assert obs.shape == (4,)
        assert np.isclose(np.sum(obs), 1.0)
    assert terminated
