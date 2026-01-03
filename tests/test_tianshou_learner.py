import pytest
from evodm.agents.tianshou_agent import *
from evodm.envs.wright_fisher_env import WrightFisherEnv
from evodm.core.hyperparameters import Hyperparameters, Presets

def test_load_file():
    p = Presets.p1_test()
    train_wf_landscapes(p, seascapes=True)
    testing_envs = load_testing_envs()
    assert isinstance(testing_envs[0], WrightFisherEnv)