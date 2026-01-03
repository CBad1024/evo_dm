import pytest
import numpy as np
from evodm.envs.sswm_env import SSWMEnv
from evodm.envs.wright_fisher_env import WrightFisherEnv
from tianshou.env import DummyVectorEnv

def test_sswm_getEnv():
    n_train = 2
    n_test = 1
    train_envs, test_envs = SSWMEnv.getEnv(n_train=n_train, n_test=n_test, N=2)
    
    assert isinstance(train_envs, DummyVectorEnv)
    assert len(train_envs) == n_train
    assert len(test_envs) == n_test
    
    obs, info = train_envs.reset()
    assert obs.shape == (n_train, 4) # 2^2 = 4
    
    action = np.zeros(n_train, dtype=int)
    obs, reward, terminated, truncated, info = train_envs.step(action)
    assert obs.shape == (n_train, 4)
    assert reward.shape == (n_train,)

def test_wf_getEnv():
    n_train = 2
    n_test = 1
    # WrightFisherEnv.getEnv is a @classmethod
    train_envs, test_envs = WrightFisherEnv.getEnv(n_train=n_train, n_test=n_test, seq_length=2)
    
    assert isinstance(train_envs, DummyVectorEnv)
    assert len(train_envs) == n_train
    
    obs, info = train_envs.reset()
    assert obs.shape == (n_train, 4)
    
    action = np.zeros(n_train, dtype=int)
    obs, reward, terminated, truncated, info = train_envs.step(action)
    assert obs.shape == (n_train, 4)
    assert reward.shape == (n_train,)

def test_sswm_full_episode_vectorized():
    n_envs = 3
    train_envs, _ = SSWMEnv.getEnv(n_train=n_envs, n_test=1, N=2)
    train_envs.reset()
    
    for _ in range(40): # total_generations default
        action = np.random.randint(0, 15, size=n_envs)
        obs, reward, terminated, truncated, info = train_envs.step(action)
        if np.any(terminated):
            break
    
    assert np.all(terminated)

def test_wf_full_episode_vectorized():
    n_envs = 2
    # 50 total gens, 10 per step -> 5 steps
    train_envs, _ = WrightFisherEnv.getEnv(n_train=n_envs, n_test=1, seq_length=2, gen_per_step=10)
    train_envs.reset()
    
    # WF env total_generations is set in __init__
    # default total_generations in WrightFisherEnv is 1000
    # Let's specify it for a faster test
    def make_wf():
        return WrightFisherEnv(pop_size=100, seq_length=2, gen_per_step=10, total_generations=30)
    
    envs = DummyVectorEnv([make_wf for _ in range(n_envs)])
    envs.reset()
    
    for _ in range(3):
        action = np.zeros(n_envs, dtype=int)
        obs, reward, terminated, truncated, info = envs.step(action)
    
    assert np.all(terminated)
