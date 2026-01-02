import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tianshou.env import DummyVectorEnv
from .helpers import define_mira_landscapes

class SSWMEnv(gym.Env):
    def __init__(self, N = 2, switch_interval = 25, total_generations = 40, random_start = False):
        super(SSWMEnv, self).__init__()
        self.N = N
        self.total_generations = total_generations
        self.random_start = random_start

        # Slice landscapes for 2^N genotypes (e.g., 4 drugs, 2^N states)
        # Mira has 15 drugs and 16 genotypes (N=4)
        mira = define_mira_landscapes()
        self.landscapes = mira[:, :2**self.N] 

        self.num_drugs = len(self.landscapes)

        self.observation_space = spaces.Box(low=0, high=1, shape=(2**self.N,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_drugs)

        self.state = 0
        self.current_drug = 0
        self.generation = 0
        self.reset()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if hasattr(self, 'random_start') and self.random_start:
            self.state = np.random.randint(0, 2**self.N)
        else:
            self.state = 0
        self.current_drug = 0
        self.generation = 0
        obs = np.zeros(2 ** self.N)
        obs[self.state] = 1  # One-hot encoding of the current state

        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)


        return obs, {}

    def step(self, action):
        if action not in range(self.num_drugs):
            raise ValueError(f"Action must be within the range of available drugs {self.num_drugs}")

        self.current_drug = action

        # Get the fitness landscape for the current drug
        fitness_landscape = self.landscapes[self.current_drug]

        # Calculate the reward based on the current state and action
        reward = 1.5 - fitness_landscape[self.state]

        # Update the state based on the action
        self.state = self.get_next_state(fitness_landscape, self.state)

        self.generation += 1
        obs = np.zeros(2**self.N)
        obs[self.state] = 1  # One-hot encoding of the current state

        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)


        truncated = False
        terminated = self.generation >= self.total_generations

        info = {"fitness": fitness_landscape[self.state]}
        return obs, reward, terminated, truncated, info

    @staticmethod
    def getEnv(n_train, n_test, N=2, random_start=False):
        def make_env():
            return SSWMEnv(N=N, random_start=random_start)
        train_envs = DummyVectorEnv([make_env for _ in range(n_train)])
        test_envs = DummyVectorEnv([make_env for _ in range(n_test)])
        return train_envs, test_envs



    def get_next_state(self, fitness_landscape, state):
        mut = range(self.N)  # Creates a list (0, 1, ..., N) to call for bitshifting mutations.

        adjMut = [state ^ (1 << m) for m in
                  mut]  # For the current genotype i, creates list of genotypes that are 1 mutation away.

        adjFit = fitness_landscape[adjMut]  # Creates list of fitnesses for each corresponding genotype that is 1 mutation away.

        fittest = adjMut[np.argmax(adjFit)]  # Find the most fit mutation
        if fitness_landscape[state] > fitness_landscape[fittest]:  # If the most fit mutation is less fit than the current genotype, stay in the current genotype.
            next_state = state
        else:
            next_state = fittest

        return next_state


    def get_obs(self):
        obs = np.zeros(2 ** self.N)
        obs[self.state] = 1
        return obs

    def get_fitness(self):
        return self.landscapes[self.current_drug][self.state]
