import os
import pickle

import torch
from pathlib import Path
from keras.optimizers import Adam
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy, BasePolicy, DQNPolicy
from tianshou.trainer import OnpolicyTrainer, OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import json

from ..envs import WrightFisherEnv, SSWMEnv
from ..core.hyperparameters import Presets as P

import numpy as np
import torch.nn as nn

def get_activation(activation_name: str | None) -> type[nn.Module] | None:
    if not activation_name:
        return nn.ReLU
    
    activation_name = activation_name.lower()
    mapping = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "leakyrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "swish": nn.SiLU,
        "silu": nn.SiLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
    }
    return mapping.get(activation_name, nn.ReLU)

# Resolve PROJECT_ROOT relative to this file's location (evodm/agents/tianshou_agent.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Logger for tensorboard
log_path = os.path.join(PROJECT_ROOT, "log", "RL")
metrics_path = os.path.join(PROJECT_ROOT, "log", "metrics")
os.makedirs(log_path, exist_ok=True)
os.makedirs(metrics_path, exist_ok=True)
writer = SummaryWriter(log_path)
logger = TensorboardLogger(writer)

class MetricsLogger:
    def __init__(self, signature):
        self.signature = signature
        self.filename = os.path.join(metrics_path, f"{signature}.csv") if signature else None
        self.corr_filename = os.path.join(metrics_path, f"{signature}_correlation.csv") if signature else None
        if self.filename:
            with open(self.filename, "w") as f:
                f.write("epoch,mean_reward,std_reward,loss\n")
        if self.corr_filename:
            with open(self.corr_filename, "w") as f:
                f.write("epoch,correlation,p_value\n")

    def log(self, epoch, mean_reward, std_reward, loss=None):
        if self.filename:
            with open(self.filename, "a") as f:
                loss_str = f",{loss}" if loss is not None else ","
                f.write(f"{epoch},{mean_reward},{std_reward}{loss_str}\n")
    
    def log_correlation(self, epoch, correlation, p_value):
        if self.corr_filename:
            with open(self.corr_filename, "a") as f:
                f.write(f"{epoch},{correlation},{p_value}\n")


def log_policy_snapshot(signature, policy, n_states, epoch=None, metrics_logger=None, mira_data=None):
    if not signature:
        return
    log_dir = os.path.join(PROJECT_ROOT, "log", "policies")
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, f"{signature}_live.json")
    
    state_tensor = torch.FloatTensor(np.identity(n_states))
    
    with torch.no_grad():
        if hasattr(policy, "model"): # DQN
            model_out = policy.model(state_tensor)
            if isinstance(model_out, tuple): # Some models return (logits, state)
                model_out = model_out[0]
            q_values = model_out.cpu().numpy().tolist()
        elif hasattr(policy, "actor"): # PPO
            # For PPO, we can log the action probabilities
            # This is a bit more complex, but we can log the logits
            actor_out = policy.actor(state_tensor)
            if isinstance(actor_out, tuple):
                actor_out = actor_out[0]
            q_values = actor_out.cpu().numpy().tolist()
        else:
            return
    
    # If MIRA data is provided and epoch tracking is enabled, calculate correlation
    if epoch is not None and metrics_logger is not None and mira_data is not None:
        try:
            from scipy.stats import pearsonr
            q_array = np.array(q_values)
            mira_landscape = mira_data.T  # Transpose to match Q orientation
            correlation, p_value = pearsonr(q_array.flatten(), mira_landscape.flatten())
            metrics_logger.log_correlation(epoch, correlation, p_value)
        except Exception as e:
            pass  # Silently fail if correlation can't be computed
            
    snapshot = {
        "n_states": n_states,
        "q_values": q_values # Reusing key name for simplicity in UI
    }
    
    with open(filename, "w") as f:
        json.dump(snapshot, f)


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
        # Calculate N from p.state_shape if possible, else assume N=4 for load (MIRA logic)
        # However, for consistency we should pass it or derive it.
        # p.state_shape = 2**N
        v_N = int(np.log2(p.state_shape)) if p.state_shape > 0 else 4
        train_envs, test_envs = SSWMEnv.getEnv(4, 2, N=v_N)
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
    # Calculate N from Presets.state_shape
    v_N = int(np.log2(p.state_shape))
    # Train with random start (to learn Q-values globally), Test with fixed start (to evaluate target task)
    train_envs, _ = SSWMEnv.getEnv(4, 1, N=v_N, random_start=True)
    _, test_envs = SSWMEnv.getEnv(1, 2, N=v_N, random_start=False)
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

    metrics_logger = MetricsLogger(signature if signature else "last_sswm")
    
    # Track loss for visualization
    last_loss = [None]  # Use list to allow mutation in nested function

    def test_fn(epoch, env_step):
        if test_collector:
            stats = test_collector.collect(n_episode=p.test_episodes)
            if stats.returns_stat:
                metrics_logger.log(epoch, stats.returns_stat.mean, stats.returns_stat.std, last_loss[0])
            
            # Log policy snapshot with correlation tracking
            from ..envs import define_mira_landscapes
            mira_data = define_mira_landscapes()
            log_policy_snapshot(signature if signature else "last_sswm", policy, 2**v_N, epoch=epoch, metrics_logger=metrics_logger, mira_data=mira_data)

    # Dynamic Epsilon Schedule
    # Explore for first 40% of training, then decay linearly to 0.05
    start_decay = int(p.epochs * 0.4)
    end_decay = int(p.epochs * 0.9)
    
    def train_fn(epoch, env_step):
        # Tianshou epoch is 1-indexed
        if epoch <= start_decay:
            eps = 1.0 # Full exploration
        elif epoch > end_decay:
            eps = 0.05 # Minimum epsilon
        else:
            # Linear decay
            progress = (epoch - start_decay) / (end_decay - start_decay)
            eps = 1.0 - progress * (1.0 - 0.05)
            
        policy.set_eps(eps)
        
        # Capture loss if available from recent training
        if hasattr(policy, '_rew_mean'):
            # DQN doesn't expose loss directly, but we can get it from the logger
            # For now, we'll leave it as None and update after trainer provides it
            pass

    # Wrap the logger to capture loss
    class LossCapturingLogger:
        def __init__(self, base_logger, last_loss_ref):
            self.base_logger = base_logger
            self.last_loss = last_loss_ref
            
        def __getattr__(self, name):
            return getattr(self.base_logger, name)
        
        def log_update_data(self, update_result, step):
            # DQN/OffpolicyTrainer exposes loss through log_update_data
            # update_result can be a dict or an object depending on Tianshou version
            loss = None
            if isinstance(update_result, dict):
                 loss = update_result.get('loss')
            elif hasattr(update_result, 'loss'):
                 loss = update_result.loss
            
            if loss is not None:
                try:
                     self.last_loss[0] = float(loss)
                except:
                     pass

            return self.base_logger.log_update_data(update_result, step)
    
    wrapped_logger = LossCapturingLogger(logger, last_loss)

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
        logger=wrapped_logger,
        test_fn=test_fn,
        train_fn=train_fn
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
    metrics_logger = MetricsLogger(signature if signature else ("wf_ss" if seascapes else "wf_ls"))
    def test_fn(epoch, env_step):
        if test_collector:
            stats = test_collector.collect(n_episode=p.test_episodes)
            if stats.returns_stat:
                metrics_logger.log(epoch, stats.returns_stat.mean, stats.returns_stat.std)
            
            # Log policy snapshot
            log_policy_snapshot(signature if signature else ("wf_ss" if seascapes_run else "wf_ls"), policy, 16) # WF is fixed N=4 for now

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
        test_fn=test_fn,
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
        
        activation = get_activation(p.activation)

        net = Net(
            state_shape=p.state_shape,
            hidden_sizes=[128, 128],
            activation=activation,
            device=device
        )
        actor = Actor(
            preprocess_net=net,
            action_shape=p.num_actions,
            hidden_sizes=[],
            device=device
        ).to(device)
        critic = Critic(
            preprocess_net=Net(state_shape=p.state_shape, hidden_sizes=[128, 128], activation=activation, device=device),
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
        activation = get_activation(p.activation)
        net = Net(
            state_shape=p.state_shape,
            action_shape=p.num_actions,
            hidden_sizes=[128, 128],
            activation=activation,
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
        estimation_step=3,
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
