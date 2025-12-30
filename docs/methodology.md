# Evolutionary Methodology

`evodm` supports two primary evolutionary dynamics to model how populations (like cancer cells or bacteria) respond to treatment.

## Strong Selection Weak Mutation (SSWM)

In the SSWM regime, the population is assumed to be mostly monomorphic (consisting of one genotype). Mutations are rare enough that when a beneficial mutation occurs, it either goes extinct quickly due to drift or fixes in the population before the next mutation arises.

### Implementation in `SSWMEnv`
- The state is represented as a single integer (genotype index).
- Transition: When a drug is applied, the framework identifies all 1-mutation neighbors of the current genotype.
- Fixation: The population "jumps" to the most fit neighboring genotype if it is fitter than the current one (Greedy ascent).
- **Reward**: Typically based on minimizing the fitness of the population in the current drug environment.

## Wright-Fisher Dynamics

The Wright-Fisher model is a more granular simulation of evolution in a population of fixed size $N$. It accounts for:
- **Selection**: Genotypes with higher fitness produce more offspring on average.
- **Mutation**: Individuals can mutate to neighboring genotypes with a probability $\mu$.
- **Genetic Drift**: Random sampling of the next generation from the current offspring pool (modeled using a Multinomial distribution).

### Implementation in `WrightFisherEnv`
- The state is a frequency vector where each element represents the proportion of the population with a specific genotype.
- Each step in the environment can represent multiple generations of Wright-Fisher evolution.
- **Pharmacodynamics (Seascapes)**: Fitness is not just a function of the genotype and drug, but also the concentration (dosage). This is modeled using Hill equations or similar Dose-Response curves.

## Seascape Modeling
A Seascape is a matrix where each row is a fitness landscape for a specific drug concentration. While a standard fitness landscape is a vector of fitness values for different genotypes, the Seascape accounts for concentration-dependent effects (Pharmacodynamics). This matrix defines the transition dynamics for concentration-aware simulations.

$$f(g, c) = f_{null}(g) + \frac{(f_{max}(g) - f_{null}(g)) \cdot c^h}{IC_{50}(g)^h + c^h}$$

Where:
- $f(g, c)$ is the fitness of genotype $g$ at concentration $c$.
- $IC_{50}(g)$ is the half-maximal inhibitory concentration for genotype $g$.
- $h$ is the Hill coefficient.
