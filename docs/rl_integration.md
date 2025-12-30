# Reinforcement Learning Integration

`evodm` utilizes Reinforcement Learning (RL) to discover treatment scheduling policies that minimize population fitness over time.

## Frameworks Used

### Tianshou Integration (`evodm/agents/tianshou_agent.py`)
The primary modern interface for training agents. It leverages the Tianshou library for efficient, vectorized environment handling and standard algorithm implementations.

- **DQN (Deep Q-Network)**: Often used for `SSWMEnv` where the state and action spaces are discrete.
- **PPO (Proximal Policy Optimization)**: Used for `WrightFisherEnv` and Seascape environments, handling larger state spaces (frequency vectors) and more complex dynamics.

### Custom Learner (`evodm/agents/legacy_learner.py`)
Contains specific implementations of Deep Q-Learning tailored for the evolutionary game. This includes:
- **`DrugSelector`**: A class that wraps a Keras-based neural network to predict Q-values for drug selections.
- Replay buffers and epsilon-greedy exploration strategies.

## Observation Spaces

| Environment | Observation Type | Description |
| :--- | :--- | :--- |
| `SSWMEnv` | One-hot Genotype | A vector with 1 at the current genotype index and 0 elsewhere. |
| `WrightFisherEnv` | Frequency Vector | A vector of length $2^N$ where each entry is the proportion of the population with that genotype. |

## Action Spaces
- **Drug Selection**: Discrete actions choosing which drug to apply.
- **Dosage Selection**: When operating in a Seascape regime, actions can include selecting specific concentration levels from the fitness matrix. 

## Training Workflow

1. **Pre-training**: Some agents are pre-trained on standardized landscapes (like the Mira landscapes) to learn baseline resistance-reversal strategies.
2. **Fine-tuning**: Agents can be fine-tuned on specific, noisy, or concentration-dependent Seascapes.
3. **Evaluation**: Policies are evaluated by running simulations over many episodes and comparing result metrics (mean fitness, time to extinction) against random cycling or standard-of-care heuristics.
