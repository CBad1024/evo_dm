import os

import torch
from keras.optimizers import Adam
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy, BasePolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from evodm.evol_game import WrightFisherEnv
from evodm.hyperparameters import Presets as P

# Logger
log_path = "./log/wf_ppo"
os.makedirs(log_path, exist_ok=True)
writer = SummaryWriter(log_path)
logger = TensorboardLogger(writer)


def load_best_policy(p: P):
    train_envs, test_envs = WrightFisherEnv.getEnv(4, 2)
    policy = get_ppo_policy(p, train_envs)
    best_policy: PPOPolicy = load_best_fn(policy)

    # // remove later
    import json

    # Get model architecture info
    model_info = {
        # "model_type": type(best_policy.state_dict().m).__name__,
        "state_dict_keys": list(best_policy.state_dict().keys()),
        "parameter_shapes": {k: list(v.shape) for k, v in best_policy.state_dict().items()},
        # "total_parameters": sum(p.numel() for p in best_policy.model.parameters()),
        "training_info": {
            "algorithm": type(best_policy).__name__,
            # Add other metadata you want to track
        }
    }

    # Save as JSON
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

    # // remove later
    return best_policy


def load_random_policy(p: P):
    train_envs, test_envs = WrightFisherEnv.getEnv(4, 2)
    policy = get_ppo_policy(p, train_envs)
    return policy


def train_ppo(p: P):
    # Set up environment
    train_envs, test_envs = WrightFisherEnv.getEnv(4, 2)

    policy = get_ppo_policy(p, train_envs)

    # Replay buffer and collectors
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(p.buffer_size, 4))
    # train_collector = AsyncCollector(policy, train_envs, VectorReplayBuffer(p.buffer_size, 4))
    test_collector = Collector(policy, test_envs)
    # test_collector = AsyncCollector(policy, test_envs)

    # Warm-up collection
    train_collector.reset()
    train_collector.collect(n_step=p.batch_size * 10)

    # Training
    trainer = OnpolicyTrainer(
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
        stop_fn=lambda mean_rewards: mean_rewards >= -1.0,
        save_best_fn=save_best_fn,  # ✅ save best model
        logger=logger,
    )
    result = trainer.run()
    print(f'Training finished with result: {result}')

    # Testing
    test_result = test_collector.collect(n_episode=p.test_episodes)
    print(f'Final testing result: {test_result}')


def get_ppo_policy(p: P, train_envs: DummyVectorEnv):
    def init_ppo_agent():
        # Model and optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    critic_optim = Adam(critic.parameters(), lr=p.lr)

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
    )
    return policy


def save_best_fn(policy: BasePolicy):
    # Logger
    log_path = "./log/wf_ppo"
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    torch.save(policy.state_dict(), os.path.join(log_path, 'best_policy.pth'))

    # Save the entire policy object
    torch.save(policy, os.path.join(log_path, 'best_complete_policy.pth'))


def load_best_fn(policy: BasePolicy):
    policy.load_state_dict(torch.load(os.path.join(log_path, 'best_policy.pth')))
    return policy
