import numpy as np
import pandas as pd
from typing import List, Callable, Dict
import itertools

def evaluate_policy(env, policy_wrapper, num_episodes=10, episode_length=20):
    """
    Runs a policy in the environment and returns a DataFrame of results.
    """
    all_results = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        for step in range(episode_length):
            action = policy_wrapper.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Record state, action, reward, and info (like mean fitness)
            res = {
                "Episode": ep,
                "Step": step,
                "Action": action,
                "Reward": reward,
                "Observation": obs.copy()
            }
            res.update(info)
            all_results.append(res)
            
            obs = next_obs
            if terminated or truncated:
                break
                
    return pd.DataFrame(all_results)

def generate_simplex_lattice(N_genotypes: int, resolution: int) -> np.ndarray:
    """
    Generates a simplex lattice of points for a given number of genotypes and resolution.
    Resolution R is the number of subdivisions per dimension.
    Total points: C(N+R-1, R)
    """
    def _generate(n, r):
        if n == 1:
            yield (r,)
            return
        for i in range(r + 1):
            for tail in _generate(n - 1, r - i):
                yield (i,) + tail

    points = []
    for counts in _generate(N_genotypes, resolution):
        points.append(np.array(counts) / resolution)
            
    return np.array(points)

def compare_policies_on_lattice(lattice: np.ndarray, policy1, policy2):
    """
    Compares two policies on a grid of points.
    Returns lattice points, actions from p1, actions from p2, and agreement mask.
    """
    actions1 = np.array([policy1.get_action(p) for p in lattice])
    actions2 = np.array([policy2.get_action(p) for p in lattice])
    agreement = (actions1 == actions2)
    
    return actions1, actions2, agreement

def get_stationary_distribution_stats(results_df, seq_length):
    """
    Computes mean and std of final states (stationary distributions).
    Assumes results_df has 'Observation' column.
    """
    # Get last step of each episode
    final_states = results_df.groupby("Episode").last()["Observation"].values
    final_states = np.stack(final_states) # (episodes, 2^seq_length)
    
    means = np.mean(final_states, axis=0)
    stds = np.std(final_states, axis=0)
    
    return means, stds
