# Active Inference Bot

A Python implementation of an Active Inference agent navigating a T-shaped maze environment. This project demonstrates the principles of Active Inference, a theoretical framework for understanding brain function and artificial intelligence. I have created this project mostly to understand the core concepts from the Active Inference book, and I am publishing it in case it is useful for anyone else trying to understand the topic.s

## Overview

This project implements Active Inference agents that demonstrate the core principles of the framework:
1. Maintaining internal models of the environment
2. Updating beliefs based on observations
3. Taking actions that minimize expected free energy
4. Learning from experience through Bayesian updating

## Screenshots

### Function Learning Demo
![Function Learning Demo](screenshots/demo.png)

**Function Learning Demo Screenshot Explanation**

This screenshot shows the interface for the Function Learning Demo, where an Active Inference agent learns to model and predict a simple function. The interface is divided into two main plots and a set of interactive controls:

- **VFE and Its Components (Left Plot):**  
  This plot visualizes the agent's internal calculations as it tries to learn about the world.  
  - The black curve shows the Variational Free Energy (VFE), a measure the agent tries to minimize to improve its understanding.
  - The red dashed line is the "complexity" term, representing how much the agent's beliefs differ from its prior expectations.
  - The blue dashed line is the "negative accuracy," showing how well the agent's beliefs explain what it observes.
  - Vertical lines and points indicate the real state, the agent's current belief, and where the VFE is minimized.

- **Machina Functions (Right Plot):**  
  This plot compares the true function of the world (blue line) with the agent's learned model (red line).  
  - The agent's goal is to make its red line match the blue line as closely as possible.
  - Dots show the current predictions for a specific input.

- **Interactive Controls (Bottom):**  
  Buttons allow you to move between states, adjust the agent's beliefs, and trigger learning for different parts of the model.  
  - For example, you can adjust the agent's belief about the current state, or tell it to learn more about the prior or the generative model.

**For those new to active inference:**  
The agent is trying to figure out how the world works by constantly updating its beliefs and minimizing its "free energy." The left plot shows how confident the agent is in its beliefs, and the right plot shows how well it has learned the true function. The controls let you experiment with how the agent learns and adapts.

**For those familiar with active inference:**  
The demo visualizes the decomposition of VFE into complexity and negative accuracy, and shows the agent's belief updating in a continuous state space. The right plot provides a direct comparison between the agent's generative model and the true world model, with real-time feedback as beliefs and parameters are updated.

### Maze Navigation
![Maze Navigation](screenshots/maze.png)

**Maze Navigation Screenshot Explanation**

This screenshot shows the interface of the Active Inference agent navigating a T-shaped maze. The display is split into several sections:

- **Maze Visualization (Center):**  
  The T-shaped maze is shown in the middle. The blue circle represents the agent's current position, and the yellow question mark indicates a special tile that reveals information about the reward's location.

- **Belief Tables (Left):**  
  - **Q(x) Distribution:** Shows the agent's current beliefs about where it might be in the maze and where the reward could be. Higher numbers mean the agent is more confident about being in that state.
  - **P(x) Distribution:** Represents the agent's prior beliefs before seeing any new information.
  - **P(y|x) Distribution:** Shows how likely the agent thinks it is to observe certain things (like seeing the reward) in each state.

- **Action and Free Energy Analysis (Top Right):**  
  The buttons let you choose which direction the agent should move. The "Expected Free Energy (EFE)" value helps the agent decide which action is best, balancing the need to explore (learn more) and exploit (go for the reward).

- **Detailed Calculations (Right):**  
  These tables break down the agent's predictions and calculations for each possible action, including:
  - Where it expects to end up (s_pi_t)
  - How uncertain it is (entropy)
  - What it expects to observe (o_pi_t)
  - How good or bad those outcomes are (zeta)

**For those new to active inference:**  
The agent is constantly updating its beliefs about the maze and making decisions to both learn more about its environment and to find the reward efficiently. The interface visualizes this process in real time, showing how the agent "thinks" and adapts as it moves.

**For those familiar with active inference:**  
The screenshot illustrates the agent's belief updating (Q(x)), prior (P(x)), and generative model (P(y|x)), as well as the decomposition of expected free energy for each policy. The real-time visualization of these distributions and EFE components provides insight into the agent's epistemic and pragmatic drives during navigation.

**Note**
The "Alt" button on the top right changes the right panel to an alternative view, which is still in development (its goal will be to show how the Active Inference model thinks over a chain of time steps, not just one step ahead)

## Core Components

### 1. Core Framework
- `distributions.py`: Implements probability distributions (Discrete and Normal) with methods for sampling, probability calculation, and KL divergence
- `machinas.py`: Provides linear and quadratic machinas for modeling relationships between variables
- `optimizers.py`: Contains optimization algorithms for updating the agent's beliefs
- `conditional_distributions.py`: Implements conditional probability distributions

### 2. Agent Implementation
- `discrete_agent.py`: Implements a discrete-state Active Inference agent
- `base.py`: Contains the base Agent class with common functionality
- The agent uses:
  - Prior beliefs (px)
  - Approximate posterior (qx)
  - Generative model (py_x)
  - Expected Free Energy calculations

## Active Inference Principles

The implementation follows these key principles:

1. **Generative Model**: The agent maintains a model of how observations depend on hidden states
2. **Belief Updating**: The agent updates its beliefs about hidden states based on observations
3. **Expected Free Energy**: Actions are chosen to minimize expected free energy, balancing:
   - Epistemic value (reducing uncertainty)
   - Pragmatic value (achieving preferred outcomes)

## Applications

The project includes two distinct applications that demonstrate Active Inference principles in different contexts:

### Application 1. Function Learning Demo

The demo application provides an interactive visualization of Active Inference in a simpler, continuous state space. It demonstrates how an agent learns to model and predict linear or quadratic functions.

#### Key Features
- Interactive visualization of Variational Free Energy (VFE) and its components
- Real-time plotting of the agent's generative model vs. the true world model
- Interactive controls for:
  - Adjusting the agent's beliefs (q(x))
  - Learning prior beliefs (p(x))
  - Learning the generative model (p(y|x))
  - Exploring different states
  - Modifying uncertainty (variance)

#### Visualization Components

1. **VFE Analysis Plot**
   - Shows Variational Free Energy (VFE) as a function of the agent's beliefs
   - Displays components:
     - Complexity (red dashed line): KL divergence between q(x) and p(x)
     - Negative Accuracy (blue dashed line): Expected log likelihood
     - Total VFE (black line): Sum of complexity and negative accuracy
   - Vertical lines showing:
     - Real state (green)
     - Current belief (orange)
     - Minimum VFE point (purple)
     - Prior mean (cyan)

2. **Machina Function Plot**
   - Compares the agent's learned model with the true world model
   - Shows:
     - World function (blue line)
     - Agent's learned function (red dashed line)
     - Current state and predictions
   - Displays the equations for both functions

#### Interactive Controls
- **State Navigation**: Move between different states
- **Variance Adjustment**: Modify the agent's uncertainty
- **Learning Controls**:
  - Adjust q(x): Update the agent's beliefs
  - Learn p(x): Update prior beliefs
  - Learn p(y|x): Update the generative model

#### Why This Matters
The demo application is particularly valuable because it:
1. Shows Active Inference in a continuous state space
2. Visualizes the relationship between beliefs and observations
3. Demonstrates how the agent learns to model its environment
4. Makes the mathematical concepts of VFE tangible
5. Shows how uncertainty affects learning and prediction

### Application 2: Maze Navigation

The maze application demonstrates Active Inference in a discrete state space through a T-shaped maze environment. The agent learns to navigate the maze to find a reward while maintaining and updating its beliefs about the environment.

#### Key Features
- T-shaped maze with discrete states
- Question mark tile that reveals reward location
- Reward placement in either top-left or top-right corner
- Real-time visualization of agent's beliefs and decision-making

#### Visualization Components

1. **Maze Visualization (Left Panel)**
   - Shows the T-shaped maze with the agent's current position
   - Displays the question mark tile that reveals the reward location
   - Visualizes the reward once discovered
   - Provides immediate feedback on agent movement and state changes

2. **Belief Visualization (Right Panel)**
   The right panel displays several key components of Active Inference:

   - **Q(x) Distribution Table**
     - Shows the agent's current beliefs about its state (approximate posterior)
     - Color-coded changes: green for increasing probabilities, red for decreasing
     - Organized by player position and reward location
     - Demonstrates how the agent updates its beliefs based on observations

   - **P(x) Distribution Table**
     - Displays the agent's prior beliefs about states
     - Shows the agent's initial assumptions about the environment
     - Helps understand how priors influence decision-making

   - **P(y|x) Distribution Table**
     - Shows the likelihood of observations given states
     - Demonstrates the agent's generative model
     - Updates based on current observation (y)
     - Helps understand how observations influence belief updates

3. **Expected Free Energy (EFE) Analysis**
   The display includes two modes for analyzing EFE:

   - **Standard Mode**
     - Shows EFE calculations for single actions (Up, Down, Left, Right)
     - Displays:
       - s_pi_t: Predicted future states
       - entropy: Uncertainty in observations
       - o_pi_t: Predicted observations
       - zeta: Difference between predicted and preferred outcomes

   - **Alternative Mode**
     - Analyzes sequences of actions (up to 4 moves)
     - Shows how EFE changes with action sequences
     - Helps understand long-term planning and its impact on free energy

4. **Interactive Controls**
   - Policy selection buttons for testing different actions
   - Mode switch between standard and alternative analysis
   - Real-time updates of all distributions and calculations

#### Why This Matters
The maze application demonstrates key Active Inference concepts:

1. **Belief Updating**
   - Real-time visualization of how Q(x) updates based on observations
   - Shows the Bayesian nature of belief updating
   - Demonstrates how uncertainty is reduced through experience

2. **Generative Modeling**
   - P(y|x) table shows how the agent models observations
   - Demonstrates the relationship between states and observations
   - Shows how the model learns from experience

3. **Expected Free Energy**
   - Visualizes how the agent evaluates actions
   - Shows the balance between exploration (reducing uncertainty) and exploitation (achieving goals)
   - Demonstrates how the agent plans sequences of actions

4. **Active Learning**
   - Shows how the agent actively seeks information
   - Demonstrates the relationship between actions and information gain
   - Visualizes how the agent balances exploration and exploitation

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run either application:
```bash
# For maze navigation
python main.py

# For function learning demo
python -m applications.demo.main
```

## Project Structure

```
.
├── agents/             # Agent implementations
├── applications/       # Specific applications
│   ├── demo/         # Function learning demo
│   └── maze/         # Maze navigation
├── core/              # Core Active Inference framework
├── environments/      # Environment implementations
├── worlds/           # World state management
└── main.py           # Entry point
```

## Dependencies

- Python 3.x
- NumPy
- Pygame
- Matplotlib (for demo application)

## Future Work

- Model more difficult problems
- Maze:
-- Better state space (e.g. including information about state visibility to preserve long-term memory)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
