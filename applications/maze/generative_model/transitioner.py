import numpy as np
from core.distributions import DiscreteDistribution
from applications.maze.generative_model.mapping import state_to_index, index_to_state

# Define position indices
TL, TC, TR, C, CD = 0, 1, 2, 3, 4

# Create transition matrices for each action
# Each matrix is 10x10 where entry (i,j) represents probability of transitioning to state i from state j

'''
(i, j) <-- TO STATE I, FROM STATE J
'''

# Up action (0)
UP_MATRIX = np.zeros((10, 10))
for reward in range(2):
    # C -> TC
    UP_MATRIX[state_to_index(TC, reward), state_to_index(C, reward)] = 1.0
    # CD -> C
    UP_MATRIX[state_to_index(C, reward), state_to_index(CD, reward)] = 1.0
# Add self-loops for states that don't transition with this action
for j in range(10):
    # Check if column j is all zeros (no transitions FROM state j)
    if not np.any(UP_MATRIX[:, j]):
        UP_MATRIX[j, j] = 1.0

# Down action (1)
DOWN_MATRIX = np.zeros((10, 10))
for reward in range(2):
    # TC -> C
    DOWN_MATRIX[state_to_index(C, reward), state_to_index(TC, reward)] = 1.0
    # C -> CD
    DOWN_MATRIX[state_to_index(CD, reward), state_to_index(C, reward)] = 1.0
# Add self-loops for states that don't transition with this action
for j in range(10):
    # Check if column j is all zeros (no transitions FROM state j)
    if not np.any(DOWN_MATRIX[:, j]):
        DOWN_MATRIX[j, j] = 1.0

# Left action (2)
LEFT_MATRIX = np.zeros((10, 10))
for reward in range(2):
    # TC -> TL
    LEFT_MATRIX[state_to_index(TL, reward), state_to_index(TC, reward)] = 1.0
    # TR -> TC
    LEFT_MATRIX[state_to_index(TC, reward), state_to_index(TR, reward)] = 1.0
# Add self-loops for states that don't transition with this action
for j in range(10):
    # Check if column j is all zeros (no transitions FROM state j)
    if not np.any(LEFT_MATRIX[:, j]):
        LEFT_MATRIX[j, j] = 1.0

# Right action (3)
RIGHT_MATRIX = np.zeros((10, 10))
for reward in range(2):
    # TL -> TC
    RIGHT_MATRIX[state_to_index(TC, reward), state_to_index(TL, reward)] = 1.0
    # TC -> TR
    RIGHT_MATRIX[state_to_index(TR, reward), state_to_index(TC, reward)] = 1.0
# Add self-loops for states that don't transition with this action
for j in range(10):
    # Check if column j is all zeros (no transitions FROM state j)
    if not np.any(RIGHT_MATRIX[:, j]):
        RIGHT_MATRIX[j, j] = 1.0

# List of all transition matrices
TRANSITION_MATRICES = [UP_MATRIX, DOWN_MATRIX, LEFT_MATRIX, RIGHT_MATRIX]

def transitioner(state: DiscreteDistribution, action: np.ndarray) -> DiscreteDistribution:
    """
    Transitions the state based on the given action probabilities in the maze environment.
    
    Args:
        state: A DiscreteDistribution representing the current state (10-length vector)
        action: A 4-length probability vector for [up, down, left, right] actions
        
    Returns:
        A DiscreteDistribution representing the next state
    """
    # The maze layout is:
    # [TL]---[TC]---[TR]
    #         |
    #        [C]
    #         |
    #        [CD]
    
    # Get current state probabilities
    state_probs = state.get_probabilities()


    # Combine transition matrices based on action probabilities
    combined_matrix = sum(a * m for a, m in zip(action, TRANSITION_MATRICES))
    
    # Calculate next state probabilities
    next_state_probs = combined_matrix @ state_probs
        
    return DiscreteDistribution(logits=np.log(next_state_probs + 1e-10))  # Add small constant for numerical stability 