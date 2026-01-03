import math
import random
import itertools
import copy
import numpy as np
from ..core.landscapes import Landscape

def s_solve(y):
    x = -math.log(1 / y - 1)
    return x

def generate_landscapes(N=5, sigma=0.5, correl=np.linspace(-1.0, 1.0, 51),
                        dense=False, CS=False, num_drugs=4):
    A = Landscape(N, sigma, dense=dense)
    # give it two chances at this step because sometimes it doesn't converge
    try:
        Bs = A.generate_correlated_landscapes(correl)
    except:
        Bs = A.generate_correlated_landscapes(correl)

    if CS:
        # this code guarantees that high-level CS will be present
        split_index = np.array_split(range(len(Bs)), num_drugs)
        keep_index = [round(np.median(i)) for i in split_index]
    else:
        keep_index = np.random.randint(0, len(Bs) - 1, size=num_drugs)

    landscapes = [Bs[i] for i in keep_index]
    drugs = [i.ls for i in landscapes]

    return landscapes, drugs

def generate_landscapes2(N=4, sigma=0.5, num_drugs=4, CS=False, dense=False, correl=None):
    landscapes = []
    for i in range(num_drugs):
        landscapes.append(Landscape(N, sigma))

    drugs = [i.ls for i in landscapes]
    return landscapes, drugs

def normalize_landscapes(drugs, seascapes=False):
    if seascapes:
        for i in range(len(drugs)):
            for j in range(len(drugs[i])):
                drugs[i][j] = drugs[i][j] - np.min(drugs[i][j])
                drugs[i][j] = drugs[i][j] / np.max(drugs[i][j])
        drugs_normalized = drugs
    else:
        drugs_normalized = []
        for i in range(len(drugs)):
            drugs_i = drugs[i] - np.min(drugs[i])
            drugs_normalized.append(drugs_i / np.max(drugs_i))

    return drugs_normalized

def discretize_state(state_vector):
    S = [i for i in range(len(state_vector))]
    probs = state_vector.T.flatten()
    state = np.random.choice(S, size=1, p=probs)
    new_states = np.zeros((len(state_vector), 1))
    new_states[state] = 1
    return new_states

def run_sim_ss(evol_steps, ss, state_vector, average_outcomes=False, conc = 0, wf = False):
    reward = []
    for i in range(evol_steps):
        if not average_outcomes:
            if wf:
                state_vector = discretize_state(state_vector)

        reward.append(np.dot(ss.ss[conc], state_vector))
        state_vector = ss.evolve(1, curr_conc = conc, p0=state_vector)

    if not average_outcomes and wf:
        state_vector = discretize_state(state_vector)

    reward = np.squeeze(reward)
    return reward, state_vector

def run_sim(evol_steps, ls, state_vector, average_outcomes=False):
    reward = []
    for i in range(evol_steps):
        if not average_outcomes:
            state_vector = discretize_state(state_vector)

        reward.append(np.dot(ls.ls, state_vector))
        state_vector = ls.evolve(1, p0=state_vector)

    if not average_outcomes:
        state_vector = discretize_state(state_vector)

    reward = np.squeeze(reward)
    return reward, state_vector

def fast_choice(options, probs):
    x = random.random()
    cum = 0
    for i, p in enumerate(probs):
        cum += p
        if x < cum:
            return options[i]
    return options[-1]

def define_mira_landscapes(as_dict=False):
    if as_dict:
        drugs = {}
        drugs['AMP'] = [1.851, 2.082, 1.948, 2.434, 2.024, 2.198, 2.033, 0.034, 1.57, 2.165, 0.051, 0.083, 2.186, 2.322,
                        0.088, 2.821]
        drugs['AM'] = [1.778, 1.782, 2.042, 1.752, 1.448, 1.544, 1.184, 0.063, 1.72, 2.008, 1.799, 2.005, 1.557, 2.247,
                       1.768, 2.047]
        drugs['CEC'] = [2.258, 1.996, 2.151, 2.648, 2.396, 1.846, 2.23, 0.214, 0.234, 0.172, 2.242, 0.093, 2.15, 0.095,
                        2.64, 0.516]
        drugs['CTX'] = [0.16, 0.085, 1.936, 2.348, 1.653, 0.138, 2.295, 2.269, 0.185, 0.14, 1.969, 0.203, 0.225, 0.092,
                        0.119, 2.412]
        drugs['ZOX'] = [0.993, 0.5, 2.069, 2.683, 1.698, 2.01, 2.138, 2.688, 1.106, 1.171, 1.894, 0.681, 1.116, 1.105,
                        1.103, 2.591]
        drugs['CXM'] = [1.748, 1.7, 2.07, 1.938, 2.94, 2.173, 2.918, 3.272, 0.423, 1.578, 1.911, 2.754, 2.024, 1.678,
                        1.591, 2.923]
        drugs['CRO'] = [1.092, 0.287, 2.554, 3.042, 2.88, 0.656, 2.732, 0.436, 0.83, 0.54, 3.173, 1.153, 1.407, 0.751,
                        2.74, 3.227]
        drugs['AMC'] = [1.435, 1.573, 1.061, 1.457, 1.672, 1.625, 0.073, 0.068, 1.417, 1.351, 1.538, 1.59, 1.377, 1.914,
                        1.307, 1.728]
        drugs['CAZ'] = [2.134, 2.656, 2.618, 2.688, 2.042, 2.756, 2.924, 0.251, 0.288, 0.576, 1.604, 1.378, 2.63, 2.677,
                        2.893, 2.563]
        drugs['CTT'] = [2.125, 1.922, 2.804, 0.588, 3.291, 2.888, 3.082, 3.508, 3.238, 2.966, 2.883, 0.89, 0.546, 3.181,
                        3.193, 2.543]
        drugs['SAM'] = [1.879, 2.533, 0.133, 0.094, 2.456, 2.437, 0.083, 0.094, 2.198, 2.57, 2.308, 2.886, 2.504, 3.002,
                        2.528, 3.453]
        drugs['CPR'] = [1.743, 1.662, 1.763, 1.785, 2.018, 2.05, 2.042, 0.218, 1.553, 0.256, 0.165, 0.221, 0.223, 0.239,
                        1.811, 0.288]
        drugs['CPD'] = [0.595, 0.245, 2.604, 3.043, 1.761, 1.471, 2.91, 3.096, 0.432, 0.388, 2.651, 1.103, 0.638, 0.986,
                        0.963, 3.268]
        drugs['TZP'] = [2.679, 2.906, 2.427, 0.141, 3.038, 3.309, 2.528, 0.143, 2.709, 2.5, 0.172, 0.093, 2.453, 2.739,
                        0.609, 0.171]
        drugs['FEP'] = [2.59, 2.572, 2.393, 2.832, 2.44, 2.808, 2.652, 0.611, 2.067, 2.446, 2.957, 2.633, 2.735, 2.863,
                        2.796, 3.203]
    else:
        drugs = []
        drugs.append([1.851, 2.082, 1.948, 2.434, 2.024, 2.198, 2.033, 0.034, 1.57, 2.165, 0.051, 0.083, 2.186, 2.322, 0.088, 2.821])
        drugs.append([1.778, 1.782, 2.042, 1.752, 1.448, 1.544, 1.184, 0.063, 1.72, 2.008, 1.799, 2.005, 1.557, 2.247, 1.768, 2.047])
        drugs.append([2.258, 1.996, 2.151, 2.648, 2.396, 1.846, 2.23, 0.214, 0.234, 0.172, 2.242, 0.093, 2.15, 0.095, 2.64, 0.516])
        drugs.append([0.16, 0.085, 1.936, 2.348, 1.653, 0.138, 2.295, 2.269, 0.185, 0.14, 1.969, 0.203, 0.225, 0.092, 0.119, 2.412])
        drugs.append([0.993, 0.805, 2.069, 2.683, 1.698, 2.01, 2.138, 2.688, 1.106, 1.171, 1.894, 0.681, 1.116, 1.105, 1.103, 2.591])
        drugs.append([1.748, 1.7, 2.07, 1.938, 2.94, 2.173, 2.918, 3.272, 0.423, 1.578, 1.911, 2.754, 2.024, 1.678, 1.591, 2.923])
        drugs.append([1.092, 0.287, 2.554, 3.042, 2.88, 0.656, 2.732, 0.436, 0.83, 0.54, 3.173, 1.153, 1.407, 0.751, 2.74, 3.227])
        drugs.append([1.435, 1.573, 1.061, 1.457, 1.672, 1.625, 0.073, 0.068, 1.417, 1.351, 1.538, 1.59, 1.377, 1.914, 1.307, 1.728])
        drugs.append([2.134, 2.656, 2.618, 2.688, 2.042, 2.756, 2.924, 0.251, 0.288, 0.576, 1.604, 1.378, 2.63, 2.677, 2.893, 2.563])
        drugs.append([2.125, 1.922, 2.804, 0.588, 3.291, 2.888, 3.082, 3.508, 3.238, 2.966, 2.883, 0.89, 0.546, 3.181, 3.193, 2.543])
        drugs.append([1.879, 2.533, 0.133, 0.094, 2.456, 2.437, 0.083, 0.094, 2.198, 2.57, 2.308, 2.886, 2.504, 3.002, 2.528, 3.453])
        drugs.append([1.743, 1.662, 1.763, 1.785, 2.018, 2.05, 2.042, 0.218, 1.553, 0.256, 0.165, 0.221, 0.223, 0.239, 1.811, 0.288])
        drugs.append([0.595, 0.245, 2.604, 3.043, 1.761, 1.471, 2.91, 3.096, 0.432, 0.388, 2.651, 1.103, 0.638, 0.986, 0.963, 3.268])
        drugs.append([2.679, 2.906, 2.427, 0.141, 3.038, 3.309, 2.528, 0.143, 2.709, 2.5, 0.172, 0.093, 2.453, 2.739, 0.609, 0.171])
        drugs.append([2.59, 2.572, 2.393, 2.832, 2.44, 2.808, 2.652, 0.611, 2.067, 2.446, 2.957, 2.633, 2.735, 2.863, 2.796, 3.203])
    return np.array(drugs)

def define_chen_landscapes(as_dict=False):
    """
    Chen et al. fitness landscapes for 8 genotypes (N=3) and 4 drugs.
    Source: Chen et al. evolutionary dynamics paper
    """
    if as_dict:
        drugs = {}
        drugs['Drug_A'] = [0.993, 0.998, 1.009, 1.003, 1.007, 1.001, 0.992, 0.997]
        drugs['Drug_B'] = [0.995, 1.005, 1.002, 0.999, 1.005, 0.994, 0.999, 1.001]
        drugs['Drug_C'] = [0.997, 1.001, 0.989, 1.003, 1.003, 0.998, 1.010, 0.997]
        drugs['Drug_D'] = [1.005, 0.988, 0.999, 1.001, 0.995, 1.011, 1.000, 0.999]
    else:
        drugs = []
        drugs.append([0.993, 0.998, 1.009, 1.003, 1.007, 1.001, 0.992, 0.997])
        drugs.append([0.995, 1.005, 1.002, 0.999, 1.005, 0.994, 0.999, 1.001])
        drugs.append([0.997, 1.001, 0.989, 1.003, 1.003, 0.998, 1.010, 0.997])
        drugs.append([1.005, 0.988, 0.999, 1.001, 0.995, 1.011, 1.000, 0.999])
    return np.array(drugs)

def define_successful_landscapes():
    return np.array([[1.20506869, 3.35382954, 0.32273345, 0.29391833],
     [3.99748972, 0.5007246, 1.94397629, 0.66432379],
     [2.13112861, 0.59541999, 1.32083556, 1.48314829],
     [1.0306329, 3.99639049, 0.70993479, 0.74922469],
     [2.73580711, 0.43070899, 3.9541098, 3.44483566],
     [3.00333663, 0.24205831, 2.57842129, 0.91931369],
     [1.23278399, 2.31578297, 2.80822274, 2.46058061],
     [2.80129708, 0.23133693, 0.33508322, 3.83364791]])
