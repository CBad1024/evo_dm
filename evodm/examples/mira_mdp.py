from evodm.dpsolve import dp_env, backwards_induction, value_iteration, policy_iteration
from evodm import define_mira_landscapes, evol_env
from evodm.learner import hyperparameters, DrugSelector
import numpy as np
import pandas as pd


def mira_env(compute_P = False):
    drugs = define_mira_landscapes()

    # The dp_env is for solving the MDP
    envdp = dp_env(N=4, num_drugs=15, drugs=drugs, sigma=0.5)
    # The evol_env is for simulating policies
    env = evol_env(N=4, drugs=drugs, num_drugs=15, normalize_drugs=False,
                   train_input='fitness')
    # The DrugSelector agent is for the RL algorithm. It requires an hp object.
    hp = hyperparameters()
    hp.N = 4  # Ensure N is set for hyperparameters
    hp.NUM_DRUGS = 15  # Ensure NUM_DRUGS is set for hyperparameters
    hp.EPISODES = 500
    hp.MIN_REPLAY_MEMORY_SIZE = 2000
    hp.MINIBATCH_SIZE = 256

    print("changed minibatch size: Minibatch = ", hp.MINIBATCH_SIZE)
    print("min replay memory size: ", hp.MIN_REPLAY_MEMORY_SIZE)
    print("num_episodes: ", hp.EPISODES)
    learner_env = DrugSelector(hp=hp, drugs=drugs)
    learner_env_naive = DrugSelector(hp=hp, drugs=drugs)
    return envdp, env, learner_env, learner_env_naive  # , naive_learner_env # Removed for simplicity, can be added back if needed




#generate drug sequences using policies from backwards induction,
#value iteration, or policy iteration
def get_sequences(policy, env, num_episodes=10, episode_length=20, finite_horizon=True):
    """
        Simulates the environment for a number of episodes using a given policy.

        Args:
            policy (np.array): The policy to follow.
            env (evol_env): The simulation environment.
            num_episodes (int): The number of simulation episodes.
            episode_length (int): The length of each episode.
            finite_horizon (bool): Whether the policy is for a finite horizon problem.
                                   If True, policy is indexed by time step.

        Returns:
            pd.DataFrame: A dataframe containing the simulation history.
        """
    ep_number_list = []
    opt_drug_list = []
    time_step_list = []
    fitness_list = []

    for i in range(num_episodes):
        env.reset()
        for j in range(episode_length):
            current_state_index = np.argmax(env.state_vector)
            if finite_horizon:
                # For FiniteHorizon, policy is shaped (time_step, state)
                action_opt = policy[j, current_state_index]
            else:
                # For Value/PolicyIteration, policy is shaped (state,)
                action_opt = policy[current_state_index]

            # evol_env now expects 0-indexed actions
            env.action = int(action_opt)
            env.step()

            # save the optimal drug, time step, and episode number
            opt_drug_list.append(env.action)
            time_step_list.append(j)
            ep_number_list.append(i)
            fitness_list.append(np.mean(env.fitness))

    results_df = pd.DataFrame({
        'episode': ep_number_list,
        'time_step': time_step_list,
        'drug': opt_drug_list,
        'fitness': fitness_list
    })
    return results_df

def main():
    """
    Main function to solve the MIRA MDP and evaluate the policies.
    """
    print("Initializing MIRA environments (DP and Simulation)...")
    envdp, env, learner_env, learner_env_naive = mira_env()  # Removed naive_learner_env from unpack

    # --- Solve the MDP using different algorithms ---
    print("\nSolving MDP with Backwards Induction (Finite Horizon)...")
    policy_bi, V_bi = backwards_induction(envdp, num_steps=16)
    print("Policy shape from Backwards Induction:", policy_bi.shape)

    print("\nSolving MDP with Value Iteration...")
    policy_vi, V_vi = value_iteration(envdp)
    print("Policy shape from Value Iteration:", policy_vi)

    print("\nSolving MDP with Policy Iteration...")
    policy_pi, V_pi = policy_iteration(envdp)
    print("Policy shape from Policy Iteration:", policy_pi)

    ## Print parameters
    print("Batch Size: ", learner_env.hp.MINIBATCH_SIZE)


    # --- Evaluate the policies by simulation ---
    print("\nSimulating policy from Backwards Induction...")
    bi_results = get_sequences(policy_bi, env, num_episodes=5, episode_length=envdp.nS, finite_horizon=True)
    print("Backwards Induction Results (first 5 rows):")
    print(bi_results.to_string())
    print("\nAverage fitness under BI policy:", bi_results['fitness'].mean())

    print("\nSimulating policy from Value Iteration...")
    vi_results = get_sequences(policy_vi, env, num_episodes=5, episode_length=envdp.nS, finite_horizon=False)
    print("Value Iteration Results (first 5 rows):")
    print(vi_results.to_string())
    print("\nAverage fitness under VI policy:", vi_results['fitness'].mean())

    print("\nSimulating policy from Policy Iteration...")
    pi_results = get_sequences(policy_pi, env, num_episodes=5, episode_length=envdp.nS, finite_horizon=False)
    print("Policy Iteration Results (first 5 rows):")
    print(pi_results.to_string())
    print("\nAverage fitness under PI policy:", pi_results['fitness'].mean())


if __name__ == "__main__":
    main()


