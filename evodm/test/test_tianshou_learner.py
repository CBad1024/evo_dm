import pytest
from evodm.tianshou_learner import *
from evodm.evol_game import WrightFisherEnv
from evodm.hyperparameters import hyperparameters, Presets

def test_load_file():
    p = Presets.p1_test()
    train_wf_landscapes(p, seascapes=True)
    testing_envs = load_testing_envs()
    assert isinstance(testing_envs[0], WrightFisherEnv)