import argparse
import builtins
import datetime as dt
import logging
import sys
from pathlib import Path

# Add parent directory to path to import from local source instead of installed package
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Ensure log directory exists
(project_root / "log").mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tianshou.data import Batch
from tianshou.policy import BasePolicy
import json

from evodm.dpsolve import dp_env, backwards_induction, value_iteration, policy_iteration
from evodm.envs import define_mira_landscapes, evol_env, WrightFisherEnv, SSWMEnv
from evodm.exp import evol_deepmind
from evodm.core.hyperparameters import Presets, Hyperparameters
from evodm.core.landscapes import Seascape, SeascapeUtils
from evodm.agents.tianshou_agent import load_best_policy, load_random_policy, train_wf_landscapes, train_sswm_landscapes

# Set up logging
timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / "log" / f"mira_mdp_{timestamp}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Alias all prints as logger.info for consistency
def print(*args, **kwargs):
    logger.info(' '.join(str(arg) for arg in args))


builtins.print = print



def evaluate_best_single_drug(landscape = define_mira_landscapes(), env_type = "wf", n_episodes=20, seq_length=4):
    """
    Evaluate the best single drug policy through simulating each individually.

    Returns:
        best_drug: int - index of the best-performing drug
        best_fitness: float - fitness of the best-performing drug
        trajectories: list of pd.DataFrame - trajectories of each drug
    """
    if env_type == "wf":
        env = WrightFisherEnv(seq_length=seq_length, drugs=landscape, num_drugs=len(landscape), normalize_drugs=False,
                               train_input='fitness')
    elif env_type == "sswm":
        env = SSWMEnv(N=seq_length, landscapes=landscape, num_drugs=len(landscape))
    else:
        raise ValueError(f"Invalid env_type: {env_type}")
    
    trajectories = []
    best_drug = None
    best_fitness = None
    for i in range(len(landscape)):
        env.reset()
        env.action = i
        trajectory = get_sequences(env, n_episodes=n_episodes)
        trajectories.append(trajectory)
        if best_fitness is None or trajectory['fitness'].mean() < best_fitness:  # Lower is better
            best_drug = i
            best_fitness = trajectory['fitness'].mean()

    return best_drug, best_fitness, trajectories



def log_trajectory_step(signature, episode, step, genotype, fitness, drug):
    if not signature:
        return
    log_dir = project_root / "log" / "trajectories"
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = log_dir / f"{signature}_live.csv"
    
    # Write header if file doesn't exist
    if not filename.exists():
        with open(filename, "w") as f:
            f.write("episode,step,genotype,fitness,drug\n")
    
    with open(filename, "a") as f:
        f.write(f"{episode},{step},{genotype},{fitness},{drug}\n")

def log_policy_snapshot(signature, policy, env):
    if not signature:
        return
    log_dir = project_root / "log" / "policies"
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = log_dir / f"{signature}_live.json"
    
    # For DQN/Simple SSWM, we can get Q-values for all genotypes
    # For N=4, there are 16 genotypes
    n_states = 2**env.N
    state_tensor = torch.FloatTensor(np.identity(n_states))
    
    with torch.no_grad():
        if hasattr(policy, "model"): # DQN
            q_values = policy.model(state_tensor).cpu().numpy().tolist()
        else:
            return # Policy type not supported for snapshot yet
            
    snapshot = {
        "n_states": n_states,
        "q_values": q_values
    }
    
    with open(filename, "w") as f:
        json.dump(snapshot, f)



def mira_env(hp_args=None):
    """Initializes the MDP environment and a simulation environment."""
    drugs = define_mira_landscapes()

    v_sigma = hp_args.sigma if hp_args else 0.5
    # The dp_env is for solving the MDP
    envdp = dp_env(N=4, num_drugs=15, drugs=drugs, sigma=v_sigma)
    # The evol_env is for simulating policies
    env = evol_env(N=4, drugs=drugs, num_drugs=15, normalize_drugs=False,
                   train_input='fitness')
    return envdp, env  # , learner_env, learner_env_naive  # , naive_learner_env # Removed for simplicity, can be added back if needed


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


def main(mdp=False, rl=False, wf_test=False, wf_train=False, wf_seascapes=False, signature=None, filename=None, hp_args=None):
    """
    Main function to solve the MIRA MDP and evaluate the policies.
    """
    # print("Initializing MIRA environments (DP and Simulation)...")
    # envdp, env = mira_env(hp_args=hp_args)  # Removed naive_learner_env from unpack
    # result = mira_env()
    # print(":: MIRA ENV ", result)
    if mdp:
        run_mdp(envdp, env)
    if rl:
        run_rl(env, envdp, hp_args=hp_args)
    if wf_test:
        run_wright_fisher(train=wf_train, seascapes=wf_seascapes, signature=signature, filename=filename, hp_args=hp_args)


def run_sim_seascape(policy, drugs, num_episodes=50, episode_length=20):
    '''
    Currently only works for a SSWM problem
    Args:
        policy:
        drugs:
        num_episodes:
        episode_length:

    Returns:

    '''
    ss = [Seascape(N=4, ls_max=drug, sigma=0.5) for drug in drugs]

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
            if policy is not None:
                action = policy[state]
            else:
                action = (np.random.randint(15), np.random.randint(8))  # FIXME make this dynamic
            fitness = ss[action[0]].ss[action[1]][state]

            states.append(state)
            actions.append(action)
            fitnesses.append(fitness)

            state = np.argmax(ss[action[0]].get_TM(action[1])[state])
            time_steps.append(j)

            episode_numbers.append(i)

    results_df = pd.DataFrame(
        {"Episode": episode_numbers, "Time Step": time_steps, "State": states, "Action": actions, "Fitness": fitnesses})
    return results_df


def run_sim_tianshou(env, policy: BasePolicy, num_episodes=10, episode_length=20, signature=None):
    """
    Simulates the environment for a number of episodes using a given policy.

    Args:
        env: the evol_env_wf environment
        policy (np.array): The policy to follow.
        num_episodes (int): The number of simulation episodes.
        episode_length (int): The length of each episode.

    Returns:
        pd.DataFrame: A dataframe containing the simulation history.
    """
    print("Running simulation ...")
    states = []
    actions = []
    time_steps = []
    episodes = []
    fitnesses = []

    for i in range(num_episodes):
        env.reset()
        obs = env.get_obs()
        print(obs)

        for j in range(episode_length):
            states.append(obs)

            batch = Batch(obs=np.array([obs]), info=Batch())
            action = int(policy(batch).act)

            obs, rew, terminated, truncated, info = env.step(action)
            actions.append(int(action))
            fitnesses.append(env.get_fitness())
            time_steps.append(j)
            episodes.append(i)
            
            # Real-time trajectory logging
            log_trajectory_step(signature, i, j, int(np.argmax(obs)), env.get_fitness(), int(action))

    results_df = pd.DataFrame(
        {"Episode": episodes, "Time Step": time_steps, "State": states, "Action": actions, "Fitness": fitnesses})
    return results_df


def run_wright_fisher(train: bool, seascapes: bool = False, signature: str = None, filename: str = None, hp_args=None):
    if seascapes:
        p_base = Presets.p1_ss()
    else:
        p_base = Presets.p1_ls()
    
    # Determine correct action space size and state shape based on dataset
    v_dataset = hp_args.dataset if hp_args else p_base.dataset
    if v_dataset == "mira":
        v_num_drugs = 15
        v_N = 4
    elif v_dataset == "chen":
        v_num_drugs = 4
        v_N = 3
    else:
        v_num_drugs = 10  # synthetic default
        v_N = hp_args.n_mut if hp_args else 4
    
    v_state_shape = (2**v_N,)  # State shape matches genotype count
    v_num_actions = v_num_drugs * 8 if seascapes else v_num_drugs

    p = p_base
    if hp_args:
        p = Presets(
            state_shape=v_state_shape,
            num_actions=v_num_actions,
            lr=hp_args.lr or p_base.lr,
            # ... (batch_size, etc.)
            epochs=hp_args.epochs or p_base.epochs,
            train_steps_per_epoch=p_base.train_steps_per_epoch,
            test_episodes=p_base.test_episodes,
            batch_size=hp_args.batch_size or p_base.batch_size,
            buffer_size=p_base.buffer_size,
            activation=hp_args.activation or p_base.activation,
            reward_clip=hp_args.reward_clip,
            dataset=hp_args.dataset if hp_args else "mira",
            gen_per_step=hp_args.gen_per_step if hp_args else 500
        )
    else:
        # Even if no hp_args, we should ensure num_actions and state_shape are correct for the dataset
        p = Presets(
            state_shape=v_state_shape,
            num_actions=v_num_actions,
            lr=p_base.lr,
            epochs=p_base.epochs,
            train_steps_per_epoch=p_base.train_steps_per_epoch,
            test_episodes=p_base.test_episodes,
            batch_size=p_base.batch_size,
            buffer_size=p_base.buffer_size,
            activation=p_base.activation,
            reward_clip=p_base.reward_clip,
            dataset=p_base.dataset,
            gen_per_step=p_base.gen_per_step
        )

    if train:
        if seascapes:
            print("Seascapes enabled")

        print("Training Wright Fisher...")
        train_wf_landscapes(p=p, seascapes=seascapes, signature=signature)

    if filename is None:
        filename = "best_policy_ss.pth" if seascapes else "best_policy.pth"
        if signature:
            filename = f"{Path(filename).stem}_{signature}.pth"

    print(f"Using policy file: {filename}")
    
    # Load shared landscapes if available for synthetic evaluation
    active_landscapes = None
    if not ((hp_args and hp_args.dataset == "mira") or p_base.dataset == "mira"):
        landscape_file = os.path.join(project_root, "log", "RL", "active_landscapes.pkl")
        if os.path.exists(landscape_file):
            print(f"Loading shared landscapes from: {landscape_file}")
            with open(landscape_file, "rb") as f:
                active_landscapes = pickle.load(f)

    # WF uses PPO, so we must load as PPO
    best_policy = load_best_policy(p, filename=filename, env_type="wf", ppo=True)
    env = WrightFisherEnv(seascapes=seascapes, num_drugs=v_num_drugs, seq_length=hp_args.n_mut if hp_args else 4, seascape_list=active_landscapes, gen_per_step=hp_args.gen_per_step if hp_args else 500)
    # Update env with WF specific parameters
    if hp_args:
        env.pop_size = hp_args.pop_size
        env.mutation_rate = hp_args.mutation_rate
        env.gen_per_step = hp_args.gen_per_step

    if not seascapes:
        results_df = run_sim_tianshou(env=env, policy=best_policy, num_episodes=10, episode_length=20, signature=signature)
    else:

        results_df = run_sim_tianshou(env=env, policy=best_policy, num_episodes=10, episode_length=20, signature=signature)
    print(results_df.loc[:, ["Episode", "Time Step", "Action", "Fitness"]])

    print("\nAverage WF fitness: ", np.mean(results_df["Fitness"]))

    actions = results_df["Action"]

    action_freq = {i: 0 for i in range(env.num_drugs * env.num_concs)}
    print(action_freq.keys())

    for action in actions:
        action_freq[action] += 1

    # Get action frequencies sorted
    sorted_actions = np.array(list(action_freq.keys()))[np.argsort(np.array(list(action_freq.values())))][::-1]
    reformatted_actions = [f"{(action % v_num_drugs, int(action / v_num_drugs))}: {action_freq[action]}" for action in sorted_actions]
    print("Top actions: \n\n", reformatted_actions)

    # Print out the seascapes of the testing environment

    # for ss in env.seascape_list:
        # SeascapeUtils.visualize_concentration_effects(ss)

    random_policy = load_random_policy(p)
    if not seascapes:
        random_env = WrightFisherEnv(num_drugs=v_num_drugs, seq_length=hp_args.n_mut if hp_args else 4)
        random_results_df = run_sim_tianshou(env=random_env, policy=random_policy)
    else:
        env.reset()
        random_results_df = run_sim_tianshou(env=env, policy=random_policy)
    print("\nAverage Random WF fitness: ", np.mean(random_results_df["Fitness"]))
    
    # Evaluate best single-drug baseline (only if trained with signature)
    if signature and train:
        print("\nEvaluating best single-drug baseline...")
        try:
            # Get landscapes based on dataset type
            if hp_args and hp_args.dataset == "mira":
                from evodm.envs import define_mira_landscapes
                landscapes = define_mira_landscapes()
            elif hp_args and hp_args.dataset == "chen":
                from evodm.envs import define_chen_landscapes
                landscapes = define_chen_landscapes()
            else:
                landscapes = active_landscapes
            
            # Evaluate all single-drug policies
            best_drug_id, best_fitness, all_trajectories = evaluate_best_single_drug(
                landscape=landscapes, 
                env_type="wf", 
                n_episodes=20,
                seq_length=v_N
            )
            
            print(f"Best single drug: #{best_drug_id} with mean fitness: {best_fitness:.4f}")
            
            # Extract trajectories for best drug
            best_drug_traj = all_trajectories[best_drug_id]
            best_drug_episodes = []
            for i in range(20):  # 20 episodes
                episode_data = best_drug_traj[best_drug_traj['Episode'] == i]
                best_drug_episodes.append(episode_data['Fitness'].tolist())
            
            # Extract learned policy trajectories
            learned_episodes = []
            for i in range(20 if len(results_df) >= 400 else 10):  # Use available episodes
                episode_data = results_df[results_df['Episode'] == i]
                learned_episodes.append(episode_data['Fitness'].tolist())
            
            # Save baseline and learned results
            baseline_dir = os.path.join(project_root, "log", "baselines")
            os.makedirs(baseline_dir, exist_ok=True)
            
            baseline_file = os.path.join(baseline_dir, f"{signature}_baseline.json")
            with open(baseline_file, 'w') as f:
                json.dump({
                    'best_drug': int(best_drug_id),
                    'mean_fitness': float(best_fitness),
                    'trajectories': best_drug_episodes
                }, f)
            
            learned_file = os.path.join(baseline_dir, f"{signature}_learned.json")
            with open(learned_file, 'w') as f:
                json.dump({
                    'mean_fitness': float(np.mean([np.mean(ep) for ep in learned_episodes])),
                    'trajectories': learned_episodes
                }, f)
            
            print(f"Saved baseline comparison to: {baseline_dir}")
        except Exception as e:
            print(f"Warning: Could not evaluate baseline: {e}")

    # TODO compare to random policy
    states_unflattened = np.array(results_df.loc[:, "State"].values)

    states_flat = []
    for i in range(len(states_unflattened)):
        for e in states_unflattened[i]:
            states_flat.append(e)

    states = np.array(states_flat)

    states = np.reshape(states, (10, 20, 2**v_N))
    print(states.shape)

    for episode in states:
        fig, ax = plt.subplots()
        for i in range(2**v_N):
            gen = bin(i)[2:].zfill(4)
            ax.plot(episode[:, i], label=f'{gen}')

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Proportion')
        ax.set_title('Genotype WF Proportions over time')
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Genotype")
        plt.tight_layout()
        # plt.show()


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


def run_rl(env, envdp, hp_args=None):
    v_N = hp_args.n_mut if hp_args else 4
    v_mira = True
    v_drugs = 15
    num_episodes = hp_args.epochs if hp_args else 350
    batch_size = hp_args.batch_size if hp_args else 256

    hp = Hyperparameters()

    hp.N = v_N
    hp.mira = v_mira
    hp.num_episodes = num_episodes
    hp.batch_size = batch_size
    rewards, naive_rewards, agent, naive_agent, dp_agent, dp_rewards, dp_policy, naive_policy, policy, dp_V, rewards_ss, agent_ss, dosage_policy_raw, V_ss = evol_deepmind(
        savepath=None, num_evols=1, N=v_N, episodes=num_episodes,
        reset_every=20, min_epsilon=0.005,
        train_input="state_vector", random_start=False,
        noise=False, noise_modifier=1, num_drugs=v_drugs,
        sigma=hp_args.sigma if hp_args else 0.5, normalize_drugs=True,
        player_wcutoff=-1, pop_wcutoff=2, win_threshold=200,
        win_reward=1, standard_practice=False, drugs=None,
        average_outcomes=False, mira=v_mira, gamma=0.99,
        learning_rate=hp_args.lr if hp_args else 0.0001, minibatch_size=batch_size,
        pre_trained=False, wf=False,
        mutation_rate=hp_args.mutation_rate if hp_args else 1e-5,
        gen_per_step=hp_args.gen_per_step if hp_args else 500,
        pop_size=hp_args.pop_size if hp_args else 10000,
        agent="none",
        update_target_every=310, total_resistance=False,
        starting_genotype=0, train_freq=1,
        compute_implied_policy_bool=True,
        dense=False, master_memory=True,
        delay=0, phenom=1, min_replay_memory_size=1000, seascapes=True, skip_to_seascape_training=True,
        cycling_policy=[10, 10, 10, 10, 4, 4, 10, 10, 4, 4, 13, 13, 4, 4, 13, 13])

    print(":: RETURNED POLICY ", np.array(policy))
    drug_policy = policy
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



def main_validate_theoretical_model(fitness_matrix = None, num_drugs = 2, num_mutations = 1):
    from evodm.theoretical_model_compute import solve_pareto_frontier

    if fitness_matrix is not None:
        f_matrix = fitness_matrix
    else:
        f_matrix = np.random.random((num_drugs, 2 ** num_mutations))

    solve_pareto_frontier(fitness_matrix)

    main_simple_sswm()




    solve_pareto_frontier(f_matrix)


def main_mdp():
    main(mdp=True)


def main_sswm(hp_args=None):
    main(rl=True, hp_args=hp_args)


def main_wf_landscapes(train, signature=None, filename=None, hp_args=None):
    main(wf_test=True, wf_train=train, signature=signature, filename=filename, hp_args=hp_args)


def main_wf_seascapes(train, signature=None, filename=None, hp_args=None):
    main(wf_test=True, wf_train=train, wf_seascapes=True, signature=signature, filename=filename, hp_args=hp_args)


def main_simple_sswm(train=True, signature=None, filename=None, hp_args=None):
    p_base = Presets.p2_ls()
    
    # Override defaults with hp_args if provided
    p = p_base
    if hp_args:
        p = Presets(
            state_shape=(2**hp_args.n_mut,),
            num_actions=p_base.num_actions,
            lr=hp_args.lr or p_base.lr,
            epochs=hp_args.epochs or p_base.epochs,
            train_steps_per_epoch=p_base.train_steps_per_epoch,
            test_episodes=p_base.test_episodes,
            batch_size=hp_args.batch_size or p_base.batch_size,
            buffer_size=p_base.buffer_size,
            activation=hp_args.activation or p_base.activation,
            reward_clip=hp_args.reward_clip,
            dataset=hp_args.dataset if hp_args else "mira",
            gen_per_step=hp_args.gen_per_step if hp_args else 500
        )

    if train:
        train_sswm_landscapes(p, signature=signature)

    if filename is None:
        filename = "best_policy_sswm.pth"
        if signature:
            filename = f"best_policy_sswm_{signature}.pth"

    # Load shared landscapes if available for synthetic evaluation
    active_landscapes = None
    if not ((hp_args and hp_args.dataset == "mira") or p_base.dataset == "mira"):
        landscape_file = os.path.join(project_root, "log", "RL", "active_landscapes.pkl")
        if os.path.exists(landscape_file):
            print(f"Loading shared landscapes from: {landscape_file}")
            with open(landscape_file, "rb") as f:
                active_landscapes = pickle.load(f)

    best_policy : DQNPolicy = load_best_policy(p, filename=filename, env_type="sswm")
    env = SSWMEnv(N=hp_args.n_mut if hp_args else 2, landscapes=active_landscapes)
    results_df = run_sim_tianshou(env = env, policy=best_policy, num_episodes=10, episode_length=20, signature=signature)

    print(results_df.loc[:, ["Episode", "Time Step", "Action", "Fitness"]])

    print("\nAverage WF fitness: ", np.mean(results_df["Fitness"]))

    actions = results_df["Action"]

    action_freq = {i: 0 for i in range(env.num_drugs)}
    # print(action_freq.keys())

    for action in actions:
        action_freq[action] += 1

    # Get action frequencies sorted
    sorted_actions = np.array(list(action_freq.keys()))[np.argsort(np.array(list(action_freq.values())))][::-1]
    reformatted_actions = [f"{(action % env.num_drugs, int(action / env.num_drugs))}: {action_freq[action]}" for action in sorted_actions]
    print("Top actions: \n\n", reformatted_actions)

    print(np.identity(2**env.N))


    state_tensor = torch.FloatTensor(np.identity(2**env.N))



    with torch.no_grad():
        print(best_policy.model(state_tensor))
        q_table = np.array(best_policy.model(state_tensor)[0])

    print(q_table)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EvoDM Runner")
    parser.add_argument("--mode", type=str, choices=["mdp", "sswm", "wf_ls", "wf_ss", "simple_sswm"], default="simple_sswm")
    parser.add_argument("--train", action="store_true", help="Train before evaluation")
    parser.add_argument("--no-train", action="store_false", dest="train", help="Skip training (only evaluation)")
    parser.add_argument("--signature", type=str, default=None, help="Signature to append to the policy filename during training")
    parser.add_argument("--filename", type=str, default=None, help="Explicit policy filename to use during evaluation")
    
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs/episodes")
    parser.add_argument("--batch-size", type=int, default=None, help="Minibatch size")
    parser.add_argument("--n-mut", type=int, default=4, help="Number of mutations (N)")
    parser.add_argument("--sigma", type=float, default=0.5, help="Sigma for landscapes")
    parser.add_argument("--pop-size", type=int, default=10000, help="Population size (for WF)")
    parser.add_argument("--mutation-rate", type=float, default=1e-5, help="Mutation rate (for WF)")
    parser.add_argument("--gen-per-step", type=int, default=500, help="Generations per step (for WF)")
    parser.add_argument("--activation", type=str, default=None, help="Activation function (relu, tanh, swish, etc.)")
    parser.add_argument("--reward-clip", action="store_true", help="Enable reward clipping (default roughly [-5, 5])")
    parser.add_argument("--dataset", type=str, default="mira", choices=["mira", "chen", "synthetic"], help="Dataset to use (mira, chen, or synthetic)")
    
    parser.set_defaults(train=True)
    
    args = parser.parse_args()
    
    if args.mode == "mdp":
        main_mdp()
    elif args.mode == "sswm":
        main_sswm(hp_args=args)
    elif args.mode == "wf_ls":
        main_wf_landscapes(train=args.train, signature=args.signature, filename=args.filename, hp_args=args)
    elif args.mode == "wf_ss":
        main_wf_seascapes(train=args.train, signature=args.signature, filename=args.filename, hp_args=args)
    elif args.mode == "simple_sswm":
        main_simple_sswm(train=args.train, signature=args.signature, filename=args.filename, hp_args=args)
