import numpy as np
from .envs import define_mira_landscapes

# This code is used for understanding selectivity but not used in the main program

def calculate_selectivity(landscape):
    selection_coeffs = []
    for drug in landscape:
        drug_norm = np.array(drug) / np.max(drug)
        selection_coeffs.append(np.mean(1 - drug_norm))

    return np.mean(selection_coeffs)

print(calculate_selectivity(define_mira_landscapes()))