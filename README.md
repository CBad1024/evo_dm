# evodm V2

An RL-Based Framework for Controlling Bacterial and Carcinomic Populations under Strong Selection Weak Mutation (SSWM) and Wright-Fisher evolutionary dynamics.

<!---badges-->
[![unit tests](https://github.com/DavisWeaver/evo_dm/actions/workflows/tests.yml/badge.svg)](https://github.com/DavisWeaver/evo_dm/actions)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![codecov](https://codecov.io/gh/DavisWeaver/evo_dm/branch/main/graph/badge.svg?token=ET8DJP3FI7)](https://codecov.io/gh/DavisWeaver/evo_dm)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FDavisWeaver%2Fevo_dm&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
<!---badges end-->

## Overview

`evodm` is a reinforcement learning (RL) framework designed to discover optimal dosing strategies to control a cancer or bacterial population. It models the population using two different evolutionary dynamics (SSWM and Wright-Fisher) and uses RL agents (PPO, DQN) to learn treatment policies that minimize population fitness and prevent resistance.

## Key Features

- **Dual Evolutionary Models**: 
  - **SSWM**: Focuses on fixation events in genotype space.
  - **Wright-Fisher**: Simulates population frequencies, drift, and selection.
- **Seascape Modeling**: Accounts for concentration-dependent fitness (Pharmacodynamics).
- **RL Integrated**: Built-in support for Tianshou (PPO, DQN) and custom Deep Q-Learning.
- **Experimental Data**: Includes empirical fitness landscapes from Mira et al. (2015).

## Getting Started

### Installation
```bash
# Clone the repository
git clone https://github.com/DavisWeaver/evo_dm.git
cd evo_dm

# Install dependencies (using uv or pip)
uv sync
```

### Quick Run
Check out `examples/run.py` to see how to train and evaluate an agent:
```bash
uv run examples/run.py
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:
- [**Architecture Overview**](docs/architecture.md): How the system is built.
- [**Evolutionary Methodology**](docs/methodology.md): Details on SSWM and Wright-Fisher dynamics.
- [**RL Integration**](docs/rl_integration.md): How to train and use reinforcement learning agents.
- [**Refactoring Proposals**](docs/refactoring_proposals.md): Recommendations for project organization.

## Architecture

```mermaid
graph LR
    Agent[RL Agent] -->|Action: Drug/Dose| Env[Evolutionary Env]
    Env -->|State: Genotypes/Freqs| Agent
    Env -->|Reward: -Fitness| Agent
    LS[Landscape/Seascape] -.->|Fitness Mapping| Env
```

## Authors

- **Original Authors**: Davis Weaver and Jeff Maltas
- **V2 Author**: Chaaranath Badrinath
