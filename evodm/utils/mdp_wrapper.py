import numpy as np
from abc import ABC, abstractmethod

class BasePolicyWrapper(ABC):
    """Base class for wrapping external policies."""
    
    @abstractmethod
    def get_action(self, obs):
        """Returns action given observation."""
        pass

class MDPPolicyWrapper(BasePolicyWrapper):
    """
    Wrapper for an analytical MDP policy.
    The MDP policy typically expects a discrete state (e.g., the dominant genotype index).
    """
    def __init__(self, policy_table):
        """
        Args:
            policy_table: A 1D array where policy_table[state_index] = action
        """
        self.policy_table = policy_table
        
    def get_action(self, obs):
        """
        Maps frequency vector observation to discrete state (dominant genotype) 
        and returns the corresponding MDP action.
        """
        # Map frequency vector to dominant genotype index
        state_index = np.argmax(obs)
        return self.policy_table[state_index]

class RLPolicyWrapper(BasePolicyWrapper):
    """Wrapper for Tianshou RL policies."""
    def __init__(self, policy):
        self.policy = policy
        
    def get_action(self, obs):
        """Expects obs as 1D numpy array."""
        # Tianshou policy expects a Batch or extra dimension
        import torch
        from tianshou.data import Batch
        
        obs_batch = Batch(obs=obs.reshape(1, -1))
        result = self.policy(obs_batch)
        return result.act[0]
