from mdptoolbox.mdp import FiniteHorizon
import sys

from evodm.dpsolve import dp_env, backwards_induction, value_iteration, policy_iteration
from evodm.evol_game import define_mira_landscapes, evol_env
from evodm.learner import DrugSelector, hyperparameters, practice, practice_WF
from evodm.exp import evol_deepmind
import numpy as np
import pandas as pd
import logging
import datetime as dt
import builtins
from evodm.landscapes import Seascape

# Set up logging
timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"mira_mdp_{timestamp}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Alias all prints as logger.info for consistency
def print(*args, **kwargs):
    logger.info(' '.join(str(arg) for arg in args))

builtins.print = print
def mira_env():
    """Initializes the MDP environment and a simulation environment."""
    drugs = define_mira_landscapes()

    # The dp_env is for solving the MDP
    envdp = dp_env(N=4, num_drugs=15, drugs=drugs, sigma=0.5)
    # The evol_env is for simulating policies
    env = evol_env(N=4, drugs=drugs, num_drugs=15, normalize_drugs=False,
                   train_input='fitness')
    # The DrugSelector agent is for the RL algorithm. It requires an hp object.
    # hp = hyperparameters()
    # hp.N = 4  # Ensure N is set for hyperparameters
    # hp.NUM_DRUGS = 15  # Ensure NUM_DRUGS is set for hyperparameters
    # hp.EPISODES = 500
    # hp.MIN_REPLAY_MEMORY_SIZE = 2000
    # hp.MINIBATCH_SIZE = 256

    # print("changed minibatch size: Minibatch = ", hp.MINIBATCH_SIZE)
    # print("min replay memory size: ", hp.MIN_REPLAY_MEMORY_SIZE)
    # print("num_episodes: ", hp.EPISODES)
    # learner_env = DrugSelector(hp=hp, drugs=drugs)
    # learner_env_naive = DrugSelector(hp=hp, drugs=drugs)
    return envdp, env #, learner_env, learner_env_naive  # , naive_learner_env # Removed for simplicity, can be added back if needed


# generate drug sequences using policies from backwards induction,
# value iteration, or policy iteration
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
    state_list = []

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

            # print("POLICY ", policy)
            # action_opt = policy[current_state_index]
            # evol_env now expects 0-indexed actions
            env.action = int(action_opt) if not env.SEASCAPES else action_opt
            env.step()

            # save the optimal drug, time step, and episode number
            opt_drug_list.append(env.action)
            time_step_list.append(j)
            ep_number_list.append(i)
            fitness_list.append(np.mean(env.fitness))
            state_list.append(np.argmax(env.state_vector))

    results_df = pd.DataFrame({
        'episode': ep_number_list,
        'time_step': time_step_list,
        'state': state_list,
        'drug': opt_drug_list,
        'fitness': fitness_list
    })
    return results_df


def main(mdp = False, rl = False, wf_test = False):
    """
    Main function to solve the MIRA MDP and evaluate the policies.
    """
    print("Initializing MIRA environments (DP and Simulation)...")
    envdp, env = mira_env()  # Removed naive_learner_env from unpack
    # result = mira_env()
    # print(":: MIRA ENV ", result)
    if mdp:
        run_mdp(envdp, env)
    if rl:
        run_rl(env, envdp)
    if wf_test:
        wf_run()



    ## Print parameters
    # print("Batch Size: ", learner_env.hp.MINIBATCH_SIZE)



    # --- RL Agent Training  ---


    # print("\nUsing naive RL agent to solve system...")
    # rewards_N, agent_N, _, __= practice(learner_env_naive, prev_action=False, standard_practice=True, compute_implied_policy_bool=True, train_freq = 5)
    # policy_N_one_hot = agent_N.compute_implied_policy(update = True)
    # policy_N = np.array([np.argmax(a) for a in policy_N_one_hot])
    # print("policy shape under naive RL: ", policy_N)



    # print("\nSimulating policy from Non-naive RL...")
    # RL_NN_results = get_sequences(policy_NN, env, num_episodes = 5, episode_length=envdp.nS, finite_horizon=False)
    # print("RL NN results:")
    # print(RL_NN_results.to_string())
    # print("\nAverage fitness under RL_NN policy:", RL_NN_results['fitness'].mean())

    # print("\nSimulating policy from naive RL...")
    # RL_N_results = get_sequences(policy_N, env, num_episodes = 5, episode_length=envdp.nS, finite_horizon=False)
    # print("RL NN results:")
    # print(RL_N_results.to_string())
    # print("\nAverage fitness under RL_N policy:", RL_N_results['fitness'].mean())



def run_sim_seascape(policy, drugs, num_episodes=10, episode_length=20):
    '''
    Currently only works for a SSWM problem
    Args:
        policy:
        drugs:
        num_episodes:
        episode_length:

    Returns:

    '''
    ss = [Seascape(N=4, ls_max=drug, sigma = 0.5) for drug in drugs]

    episode_numbers = []
    states = []
    actions = []
    fitnesses = []
    time_steps = []
    for i in range(num_episodes):

        state = 0  # Initial state vector
        action = None
        fitness = 0
        for j in range(episode_length):

            action = policy[state]
            fitness = ss[action[0]].ss[action[1]][state]

            states.append(state)
            actions.append(action)
            fitnesses.append(fitness)

            state = np.argmax(ss[action[0]].get_TM(action[1])[state])
            time_steps.append(j)

            episode_numbers.append(i)

    results_df = pd.DataFrame({"Episode": episode_numbers,  "Time Step": time_steps, "State": states, "Action": actions, "Fitness": fitnesses})
    return results_df





# def run_sim_wf(env, policy, drugs, num_episodes=10, episode_length=20):
#     """
#     Simulates the environment for a number of episodes using a given policy.
#
#     Args:
#         env: the evol_env_wf environment
#         policy (np.array): The policy to follow.
#         drugs (list): List of drug landscapes.
#         num_episodes (int): The number of simulation episodes.
#         episode_length (int): The length of each episode.
#
#     Returns:
#         pd.DataFrame: A dataframe containing the simulation history.
#     """
#
#     for i in range(num_episodes):
#         env.reset()
#         for j in range(episode_length):
#             current_state_index = np.argmax(env.state_vector)
#             action_opt = policy[current_state_index]
#
#             # evol_env now expects 0-indexed actions
#             env.action = int(action_opt) if not env.SEASCAPES else action_opt
#             env.step()
#
#
#     return results_df

def wf_run():
    practice_WF(hyperparameters())



def run_mdp(envdp, env):
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

    # --- Evaluate the policies by simulation ---
    print("\nSimulating policy from Backwards Induction...")
    bi_results = get_sequences(policy_bi, env, num_episodes=10, episode_length=envdp.nS, finite_horizon=True)
    print("Backwards Induction Results (first 5 rows):")
    print(bi_results.to_string())
    print("\nAverage fitness under BI policy:", bi_results['fitness'].mean())

    print("\nSimulating policy from Value Iteration...")
    vi_results = get_sequences(policy_vi, env, num_episodes=10, episode_length=envdp.nS, finite_horizon=False)
    print("Value Iteration Results (first 5 rows):")
    print(vi_results.to_string())
    print("\nAverage fitness under VI policy:", vi_results['fitness'].mean())

    print("\nSimulating policy from Policy Iteration...")
    pi_results = get_sequences(policy_pi, env, num_episodes=10, episode_length=envdp.nS, finite_horizon=False)
    print("Policy Iteration Results (first 5 rows):")
    print(pi_results.to_string())
    print("\nAverage fitness under PI policy:", pi_results['fitness'].mean())

def run_rl(env, envdp):
    v_N = 4
    v_mira = True
    v_drugs = 15
    num_episodes = 350
    batch_size = 256



    hp = hyperparameters()
    hp.SEASCAPES = False
    hp.N = v_N
    hp.mira = v_mira
    hp.num_episodes = num_episodes
    hp.batch_size = batch_size
    rewards, naive_rewards, agent, naive_agent, dp_agent, dp_rewards, dp_policy, naive_policy, policy, dp_V, rewards_ss, agent_ss, dosage_policy_raw, V_ss = evol_deepmind(savepath = None, num_evols = 1, N = v_N, episodes = num_episodes,
                  reset_every = 20, min_epsilon = 0.005,
                  train_input = "state_vector",  random_start = False,
                  noise = False, noise_modifier = 1, num_drugs = v_drugs,
                  sigma = 0.5, normalize_drugs = True,
                  player_wcutoff = -1, pop_wcutoff = 2, win_threshold = 200,
                  win_reward = 1, standard_practice = False, drugs = None,
                  average_outcomes = False, mira = v_mira, gamma = 0.99,
                  learning_rate = 0.0001, minibatch_size = batch_size,
                  pre_trained = False, wf = False,
                  mutation_rate = 1e-5,
                  gen_per_step = 500,
                  pop_size = 10000,
                  agent = "none",
                  update_target_every = 310, total_resistance = False,
                  starting_genotype = 0, train_freq = 1,
                  compute_implied_policy_bool = True,
                  dense = False, master_memory = True,
                  delay = 0, phenom = 1, min_replay_memory_size = 1000, seascapes = True)

    print(":: RETURNED POLICY ", np.array(policy))
    drug_policy=policy
    print("policy shape under non-naive RL: ", np.array(policy).shape)
    dosage_policy = np.array([np.argmax(s) for s in dosage_policy_raw])
    final_policy = [(int(drug_policy[i]), int(dosage_policy[i])) for i in range(len(drug_policy))]
    print("Final policy (drug, dosage): ", final_policy)
    print("\nDrug agent Q-table: ", agent.q_table())
    print("\nDose agent Q-table: ", agent_ss.q_table())

    print("\nSimulating policy from Non-naive RL...")
    RL_NN_results = run_sim_seascape(final_policy, np.array(define_mira_landscapes()))
    print("RL NN results:")
    print(RL_NN_results.to_string())
    print("\nAverage fitness under RL_NN policy:", RL_NN_results['Fitness'].mean())


if __name__ == "__main__":
    main(mdp = False, rl = False, wf_test = True)
