# Refactoring Proposals for evodm

Based on a review of the current source code, several architectural and organizational improvements are recommended to enhance maintainability, scalability, and clarity.

## 1. Project Structure Reorganization

The current structure groups most logic into a single flat `evodm/` directory. A more modular approach is recommended:

```text
evodm/
├── core/
│   ├── landscapes.py       # Landscape and Seascape classes
│   ├── dynamics.py         # SSWM and Wright-Fisher theoretical logic
│   └── hyperparameters.py  # Configuration and Presets
├── envs/
│   ├── base.py             # Abstract base classes for environments
│   ├── sswm_env.py         # Refactored SSWMEnv
│   └── wright_fisher_env.py # Refactored WrightFisherEnv
├── agents/
│   ├── tianshou_agent.py   # Tianshou-based training logic
│   ├── legacy_learner.py   # Old learner.py logic
│   └── networks.py         # Neural network architectures (Torch/Keras)
├── utils/
│   ├── data.py             # Data loading and Mira datasets
│   ├── visualization.py    # Plotting and landscape viz
│   └── misc.py             # Helper functions
└── ...
```

## 2. Modularization of `evol_game.py`

`evol_game.py` is currently a "God object" file exceeding 1,200 lines, containing multiple environment classes, landscape generation helpers, and simulation runners.
- **Proposal**: Split this file into `evodm/envs/` as described above. Each environment should reside in its own file.
- **Benefit**: Easier debugging and prevents circular imports.

## 3. Naming Conventions and Type Hinting

- **Consistency**: Transition from `camelCase` and inconsistent underscores to PEP 8 `snake_case` for all functions and variables.
- **Class Naming**: Ensure all classes use `PascalCase` (e.g., `hyperparameters` should be `Hyperparameters`).
- **Type Hinting**: Systematically add Python type hints (`List[float]`, `np.ndarray`, etc.) to improve IDE support and catch bugs early.

## 4. Decoupling RL from Environments

Currently, some environment methods (like `get_ppo_policy` inside `tianshou_learner.py` being tightly coupled with environment presets) make it harder to swap RL frameworks.
- **Proposal**: Define a clear "Observation/Action" interface. The RL training script should only interact with the environment through the Gymnasium API.

## 5. Configuration Management

Transition from the `hyperparameters` class to a more standard configuration approach:
- Use YAML or JSON files for experiment configurations.
- Use `Pydantic` for configuration validation.

## 6. Cleanup of Legacy Code

Identified several flags (e.g., `TODO`, `FIXME`, and commented-out code blocks) in `evol_game.py` and `exp.py`.
- **Proposal**: Conduct a dedicated cleanup pass to remove dead code and resolve long-standing TODOs (e.g., the `discretize_state` behavior in different environments).
