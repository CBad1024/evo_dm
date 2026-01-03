import gymnasium as gym
from gymnasium import spaces
import numpy as np
import itertools
from tianshou.env import DummyVectorEnv
from ..core.landscapes import Seascape

class WrightFisherEnv(gym.Env):
    drug_data_set = False
    seascape_list = None
    drug_seascapes = None

    def __init__(self, pop_size=10000, seq_length=4, mutation_rate=1e-4, gen_per_step=500, total_generations=1000, seascapes = False, num_drugs = 10, seascape_list=None, random_start=False):
        super(WrightFisherEnv, self).__init__()
        self.pop_size = pop_size
        self.seq_length = seq_length
        self.mutation_rate = mutation_rate
        self.random_start = random_start
        self.switch_interval = gen_per_step # Use gen_per_step as the interval between actions
        self.total_generations = total_generations
        self.genotypes = [''.join(seq) for seq in itertools.product("01", repeat=self.seq_length)]
        self.seascapes = seascapes

        # Drug data (we want it to be same for all trials)
        if seascape_list is not None:
            self.seascape_list = seascape_list
            self.num_drugs = len(seascape_list)
        else:
            self.num_drugs = num_drugs
            self.seascape_list = [Seascape(self.seq_length, sigma=0.5, selectivity=0.05, drug_label = i) for i in range(self.num_drugs)]
        
        self.drug_seascapes = np.array([seas.ss for seas in self.seascape_list])  # (drug, conc, genotype)

        self.concentrations = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]

        if seascapes:
            self.num_concs = len(self.seascape_list[0].concentrations)
        else:
            self.num_concs = 1

        self.action_space = spaces.Discrete(self.num_drugs*self.num_concs)


        # Observation space: genotype frequencies
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.genotypes),), dtype=np.float32)

        # State initialization
        self.pop = {}
        self.current_drug = 0
        self.current_conc = 0
        self.generation = 0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_start:
            # Start at a random genotype
            idx = np.random.randint(len(self.genotypes))
            self.pop = {self.genotypes[idx]: self.pop_size}
        else:
            self.pop = {'0' * self.seq_length: self.pop_size}
        self.generation = 0
        self.current_drug = 0
        self.current_conc = 2 #we always rest the concentration to a medium value
        obs = self._get_obs()
        return obs, {}


    def step(self, action):
        if not self.seascapes:
            self.current_drug = action
            self.current_conc = 2
        else:
            self.current_drug = action % self.num_drugs #ones digit
            self.current_conc = int(action/self.num_drugs) #tens digit

        if self.current_drug >= self.num_drugs or self.current_conc >= 8:
           raise ValueError("Current Drug: ", self.current_drug, "\nCurrent Conc: ", self.current_conc)

        fitness = {geno: self.drug_seascapes[self.current_drug, self.current_conc, i] for i, geno in enumerate(self.genotypes)}

        for _ in range(self.switch_interval):
            self.time_step(fitness)
            self.generation += 1
            if self.generation >= self.total_generations:
                break

        obs = self._get_obs()
        avg_fit = sum((self.pop.get(g, 0) / self.pop_size) * fitness[g] for g in self.genotypes)

        if not self.seascapes:
            # Negative reward directly penalizes high fitness (thriving bacteria)
            reward = -avg_fit

        else:
            reward = 1-avg_fit

        if self.seascapes:
            a = self.concentrations[self.current_conc]
            if a <= 0:
                print(a)
            reward -= 0.06*np.log10(self.concentrations[self.current_conc])

        terminated = self.generation >= self.total_generations
        truncated = False  # Gymnasium requires this explicitly
        info = {'avg_fitness': avg_fit}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        freqs = np.array([self.pop.get(geno, 0) for geno in self.genotypes]) / self.pop_size
        return freqs.astype(np.float32)

    def get_obs(self):
        return self._get_obs()

    def time_step(self, fitness):
        self.mutation_step()
        self.offspring_step(fitness)

    def mutation_step(self):
        mutation_count = np.random.poisson(self.mutation_rate * self.pop_size * self.seq_length)
        for _ in range(mutation_count):
            haplotype = self.get_random_haplotype()
            if self.pop[haplotype] > 1:
                self.pop[haplotype] -= 1
                mutant = self.get_mutant(haplotype)
                self.pop[mutant] = self.pop.get(mutant, 0) + 1

    def get_random_haplotype(self):
        haplotypes, frequencies = zip(*self.pop.items())
        frequencies = np.array(frequencies) / self.pop_size
        return np.random.choice(haplotypes, p=frequencies)

    def get_mutant(self, haplotype):
        site = np.random.randint(0, self.seq_length)
        new_base = '1' if haplotype[site] == '0' else '0'
        return haplotype[:site] + new_base + haplotype[site + 1:]

    def offspring_step(self, fitness):
        haplotypes = list(self.pop.keys())
        frequencies = [self.pop[h] / self.pop_size for h in haplotypes]
        fit_values = [fitness[h] for h in haplotypes]
        weights = np.array(frequencies) * np.array(fit_values)
        weights /= weights.sum()

        counts = np.random.multinomial(self.pop_size, weights)
        self.pop.clear()
        for haplotype, count in zip(haplotypes, counts):
            if count > 0:
                self.pop[haplotype] = count

    @classmethod
    def getEnv(cls, n_train, n_test, seascapes = False, seascape_list = None, num_drugs = 10, gen_per_step=500, seq_length=4, random_start=False):
        def make_env_train():
            return WrightFisherEnv(seascapes=seascapes, seascape_list=seascape_list, num_drugs=num_drugs, gen_per_step=gen_per_step, seq_length=seq_length, random_start=random_start)
        def make_env_test():
            return WrightFisherEnv(seascapes=seascapes, seascape_list=seascape_list, num_drugs=num_drugs, gen_per_step=gen_per_step, seq_length=seq_length, random_start=False)
        train_envs = DummyVectorEnv([make_env_train for _ in range(n_train)])
        test_envs = DummyVectorEnv([make_env_test for _ in range(n_test)])
        return train_envs, test_envs

    def get_fitness(self):
        frequencies = np.array(list(self.pop.values())) / self.pop_size
        haplotypes = list(self.pop.keys())
        state_vector = np.zeros(2**self.seq_length)
        hap_inds = [int(hap, 2) for hap in haplotypes]
        for i, hap in enumerate(hap_inds):
            state_vector[hap] = frequencies[i]

        fitnesses = np.dot(state_vector, self.drug_seascapes[self.current_drug, self.current_conc])
        return np.mean(fitnesses)
