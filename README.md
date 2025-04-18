# Active Inference Bot

A Python implementation of active inference agents in various environments, demonstrating the principles of active inference and free energy minimization.

## Overview

This project implements active inference agents that can operate in different environments:
- A 1D demo world with linear/quadratic functions
- A 2D maze environment with discrete states

The implementation focuses on demonstrating active inference principles, including:
- Variational Free Energy (VFE) minimization <-- almost done
- Expected Free Energy (EFE) calculations <-- TODO
- Bayesian inference in action selection <-- TODO

## Project Structure

```
.
├── agents/          # Agent implementations
├── applications/    # Application-specific code
│   ├── demo/       # 1D demo environment
│   └── maze/       # 2D maze environment
├── core/           # Core active inference components
├── environments/   # Environment implementations
├── worlds/         # World abstractions
└── main.py         # Main entry point
```

## Requirements

- Python 3.x
- Dependencies:
  - numpy >= 1.24.0
  - matplotlib >= 3.7.0
  - tensorflow >= 2.15.0
  - pygame >= 2.5.0

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd Active-Inference-Bot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To run the different demos:

1. 1D Demo Environment:
```bash
python -m applications.demo.main
```

2. 2D Maze Environment:
```bash
python -m applications.maze.main
```

## Features

- Implementation of active inference principles
- Multiple environment types (1D and 2D)
- Visual representation of agent behavior
- Configurable agent parameters
- Modular architecture for easy extension

## Development

The project is structured to allow easy addition of new environments and agent types. Key components:

- `World`: Base class for different environments
- `Agent`: Base class for different agent implementations
- `Environment`: Interface for environment-specific implementations