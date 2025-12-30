import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evodm.envs import WrightFisherEnv
from evodm.agents.tianshou_agent import load_best_policy
from evodm.core.hyperparameters import Presets
from evodm.utils.mdp_wrapper import RLPolicyWrapper, MDPPolicyWrapper
from evodm.utils.analysis import (
    evaluate_policy, 
    generate_simplex_lattice, 
    compare_policies_on_lattice, 
    get_stationary_distribution_stats
)

def main():
    p = Presets.p1_ls()
    try:
        # Try different filenames
        filenames = ["best_policy.pth", "best_policy_sswm.pth"]
        found = False
        for fname in filenames:
            if os.path.exists(os.path.join(project_root, "log/RL", fname)):
                tianshou_policy = load_best_policy(p, filename=fname, ppo=True)
                rl_wrapper = RLPolicyWrapper(tianshou_policy)
                print(f"RL Policy loaded successfully from {fname}.")
                found = True
                break
        
        if not found:
            raise FileNotFoundError("No policy file found in log/RL")
            
    except Exception as e:
        print(f"Could not load RL policy: {e}. Using a random sampler for demonstration.")
        class RandomSampler:
            def get_action(self, obs):
                return np.random.randint(0, 10)
        rl_wrapper = RandomSampler()

    # 2. Setup "External" MDP Policy (Placeholder)
    # Let's assume a simple greedy policy: always pick the drug that minimizes 
    # the fitness of the CURRENTLY dominant genotype.
    env = WrightFisherEnv(num_drugs=10)
    # Mock policy table: for each genotype (16), what's the best drug (10)?
    # we'll just use the drug that has the lowest fitness for that genotype in conc 2
    mock_policy_table = np.argmin(env.drug_seascapes[:, 2, :], axis=0) 
    mdp_wrapper = MDPPolicyWrapper(mock_policy_table)
    print("MDP Wrapper initialized with greedy mock policy.")

    # --- METRIC 1: Mean Fitness Comparison ---
    print("\nExecuting Metric 1: Mean Fitness Comparison...")
    rl_results = evaluate_policy(env, rl_wrapper, num_episodes=5, episode_length=20)
    mdp_results = evaluate_policy(env, mdp_wrapper, num_episodes=5, episode_length=20)
    
    print(f"RL Mean Fitness: {rl_results['avg_fitness'].mean():.4f}")
    print(f"MDP Mean Fitness: {mdp_results['avg_fitness'].mean():.4f}")

    # --- METRIC 2: State-Space Lattice Evaluation ---
    print("\nExecuting Metric 2: Lattice Evaluation...")
    # resolution 5 on a 16-genotype simplex is too large (combinatorial explosion)
    # Let's use a very low resolution for demo, or a subspace.
    # For 16 genotypes, resolution 2 results in 136 points? No, C(16+2-1, 2) = 136.
    lattice = generate_simplex_lattice(N_genotypes=16, resolution=2)
    print(f"Generated {len(lattice)} points on the simplex lattice.")
    
    a1, a2, agreement = compare_policies_on_lattice(lattice, rl_wrapper, mdp_wrapper)
    agreement_pct = np.mean(agreement) * 100
    print(f"Policy Agreement on Lattice: {agreement_pct:.2f}%")

    # --- METRIC 3: Stationary Distribution Analysis ---
    print("\nExecuting Metric 3: Stationary Distribution Analysis...")
    rl_means, rl_stds = get_stationary_distribution_stats(rl_results, env.seq_length)
    mdp_means, mdp_stds = get_stationary_distribution_stats(mdp_results, env.seq_length)
    
    print("\nRL Final State Means (Genotypes):")
    print(rl_means)
    print("\nMDP Final State Means (Genotypes):")
    print(mdp_means)

    # 4. Simple Visualization
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].bar(range(16), rl_means, yerr=rl_stds, alpha=0.7, label='RL')
    ax[0].set_title("RL Final State Dist.")
    ax[1].bar(range(16), mdp_means, yerr=mdp_stds, alpha=0.7, color='orange', label='MDP')
    ax[1].set_title("MDP Final State Dist.")
    
    plt.tight_layout()
    plt.savefig("policy_comparison.png")
    print("\nComparison plots saved to policy_comparison.png")

if __name__ == "__main__":
    main()
