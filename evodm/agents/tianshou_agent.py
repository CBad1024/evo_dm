import os
import pickle

import torch
from pathlib import Path
from keras.optimizers import Adam
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy, BasePolicy, DQNPolicy
from tianshou.trainer import OnpolicyTrainer, OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from ..envs import WrightFisherEnv, SSWMEnv
from ..core.hyperparameters import Presets as P

# Resolve PROJECT_ROOT relative to this file's location (evodm/agents/tianshou_agent.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Logger for tensorboard
log_path = os.path.join(PROJECT_ROOT, "log", "RL")
os.makedirs(log_path, exist_ok=True)
writer = SummaryWriter(log_path)
logger = TensorboardLogger(writer)

seascapes_run = False
non_seascape_policy = "best_policy.pth"
seascape_policy = "best_policy_ss.pth"


def load_testing_envs():
    envs = pickle.load(open(os.path.join(log_path, "testing_envs.pkl"), "rb"))
    return envs


def load_best_policy(p: P, filename: str = non_seascape_policy, ppo: bool = False, env_type: str = "wf"):
    if env_type == "wf":
        train_envs, test_envs = WrightFisherEnv.getEnv(4, 2, seascapes_run)
        current_log_path = log_path
    elif env_type == "sswm":
        train_envs, test_envs = SSWMEnv.getEnv(4, 2)
        current_log_path = os.path.join(PROJECT_ROOT, "log", "sswm_dqn")
    else:
        raise ValueError(f"Unknown env_type: {env_type}")

    if ppo:
        policy = get_ppo_policy(p, train_envs)
    else:
        policy = get_dqn_policy(p, train_envs)
    
    best_policy = load_best_fn(policy=policy, filename=filename, path=current_log_path)
    return best_policy


def load_random_policy(p: P):
    train_envs, test_envs = WrightFisherEnv.getEnv(4, 2, seascapes_run)
    policy = get_ppo_policy(p, train_envs)
    return policy



def train_sswm_landscapes(p: P, signature: str = None):
    train_envs, test_envs = SSWMEnv.getEnv(4, 2)
    policy = get_dqn_policy(p, train_envs)

    # Use a closure to capture the signature for the save callback
    def save_best_v2(policy: BasePolicy):
        filename = "best_policy_sswm.pth"
        if signature:
            filename = f"best_policy_sswm_{signature}.pth"
        
        log_path_sswm = os.path.join(PROJECT_ROOT, "log", "sswm_dqn")
        os.makedirs(log_path_sswm, exist_ok=True)
        torch.save(policy.state_dict(), os.path.join(log_path_sswm, filename))
        print(f"Best policy saved to: {os.path.join(log_path_sswm, filename)}")

    train_collector = Collector(policy, train_envs, VectorReplayBuffer(p.buffer_size, 4))
    test_collector = Collector(policy, test_envs)

    drug_trainer = OffpolicyTrainer(
        policy=policy,
        max_epoch=p.epochs,
        batch_size=p.batch_size,
        train_collector=train_collector,
        test_collector=test_collector,
        episode_per_test=p.test_episodes,
        step_per_collect= p.batch_size * 10,
        step_per_epoch=p.train_steps_per_epoch,
        save_best_fn=save_best_v2,
        logger=logger,
    )
    result = drug_trainer.run()

    print(f'Drug Cycling Training finished with result: {result}')

    # Testing
    test_result = test_collector.collect(n_episode=p.test_episodes)
    print(f'Final testing result: {test_result}')




def train_wf_landscapes(p: P, seascapes=False, signature: str = None):
    global seascapes_run
    seascapes_run = seascapes
    # Set up environment
    train_envs, test_envs = WrightFisherEnv.getEnv(4, 2, seascapes=seascapes)
    policy = get_ppo_policy(p, train_envs)

    def save_best_v2(policy: BasePolicy):
        global seascapes_run
        if seascapes_run:
            base_name = seascape_policy
        else:
            base_name = non_seascape_policy
        
        if signature:
            filename = f"{Path(base_name).stem}_{signature}.pth"
        else:
            filename = base_name

        os.makedirs(log_path, exist_ok=True)
        torch.save(policy.state_dict(), os.path.join(log_path, filename))
        print(f"Best policy saved to: {os.path.join(log_path, filename)}")

    # Replay buffer and collectors
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(p.buffer_size, 4))
    # train_collector = AsyncCollector(policy, train_envs, VectorReplayBuffer(p.buffer_size, 4))
    test_collector = Collector(policy, test_envs)
    # test_collector = AsyncCollector(policy, test_envs)

    # Warm-up collection
    train_collector.reset()
    train_collector.collect(n_step=p.batch_size * 10)

    print(f"Max epochs set to: {p.epochs}")
    # Training
    drug_trainer = OnpolicyTrainer(
        policy=policy,
        max_epoch=p.epochs,
        batch_size=p.batch_size,
        train_collector=train_collector,
        test_collector=test_collector,
        step_per_epoch=p.train_steps_per_epoch,
        repeat_per_collect=4,  # <- recommended value for PPO
        episode_per_test=p.test_episodes,
        step_per_collect=p.batch_size * 10,
        train_fn=lambda epoch, env_step: None,
        test_fn=lambda epoch, env_step: None,
        stop_fn=lambda mean_rewards: None,  # changed from mean rewards >= -1
        save_best_fn=save_best_v2,  # ✅ save best model
        logger=logger,

    )
    result = drug_trainer.run()

    print(f'Drug Cycling Training finished with result: {result}')

    # Testing
    test_result = test_collector.collect(n_episode=p.test_episodes)
    print(f'Final testing result: {test_result}')


def get_ppo_policy(p: P, train_envs: DummyVectorEnv):
    def init_ppo_agent():
        # Model and optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # action_space = train_envs.get_env_attr("action_space")[0]

        net = Net(
            state_shape=p.state_shape,
            hidden_sizes=[128, 128],
            device=device
        )
        actor = Actor(
            preprocess_net=net,
            action_shape=p.num_actions,
            hidden_sizes=[],
            device=device
        ).to(device)
        critic = Critic(
            preprocess_net=Net(state_shape=p.state_shape, hidden_sizes=[128, 128], device=device),
            hidden_sizes=[],
            device=device
        ).to(device)
        return actor, critic

    actor, critic = init_ppo_agent()
    actor_optim = Adam(actor.parameters(), lr=p.lr)
    # critic_optim = Adam(critic.parameters(), lr=p.lr)

    # lr_scheduler_actor = LambdaLR(actor_optim, lr_lambda=lambda epoch: p.lr * (0.995 ** epoch))

    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=actor_optim,
        dist_fn=torch.distributions.Categorical,
        action_space=train_envs.get_env_attr("action_space")[0],
        discount_factor=0.99,
        max_grad_norm=0.5,
        vf_coef=0.5,
        ent_coef=0.01,
        gae_lambda=0.95,
        reward_normalization=False,
        action_scaling=False,
        deterministic_eval=False,
        dual_clip=None,
        value_clip=True,
        eps_clip=0.2,
        advantage_normalization=True,
        recompute_advantage=False,
        # lr_scheduler=lr_scheduler_actor,
    )
    print("Policy Action Space: ", policy.action_space)
    return policy


def get_dqn_policy(p: P, train_envs: DummyVectorEnv):
    def init_dqn_agent():
        # Model and optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = Net(
            state_shape=p.state_shape,
            action_shape=p.num_actions,
            hidden_sizes=[128, 128],
            device=device
        )
        return net

    net = init_dqn_agent()
    optim = Adam(net.parameters(), lr=p.lr)

    policy = DQNPolicy(
        model=net,
        optim=optim,
        action_space=train_envs.get_env_attr("action_space")[0],
        discount_factor=0.99,
        estimation_step=2,
        target_update_freq=p.batch_size * 10,
    )
    print("Policy Action Space: ", policy.action_space)
    return policy


def save_best_fn_sswm(policy: BasePolicy):
    # Logger
    filename = "best_policy_sswm.pth"
    log_path_sswm = os.path.join(PROJECT_ROOT, "log", "sswm_dqn")
    os.makedirs(log_path_sswm, exist_ok=True)

    torch.save(policy.state_dict(), os.path.join(log_path_sswm, filename))

def save_best_fn(policy: BasePolicy):
    # Logger
    # if seascapes, then save to best_policy_ss.pth
    # if not seascapes, then save to best_policy.pth
    global seascapes_run
    if seascapes_run:
        filename = seascape_policy
    else:
        filename = non_seascape_policy

    os.makedirs(log_path, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(log_path, filename))


def load_best_fn(policy: BasePolicy, filename: str = "best_policy.pth", path: str = log_path):
    full_path = os.path.join(path, filename)
    print(f"Loading best policy from: {full_path}")
    policy.load_state_dict(torch.load(full_path))
    return policy

def load_best_fn_sswm(policy: BasePolicy, filename: str = "best_policy_sswm.pth"):

    policy.load_state_dict(torch.load(os.path.join(log_path, filename)))
    return policy
