import numpy as np

class Policy():
    def __init__(self, policy_matrix, actions=None):
        if actions is not None:
            self.policy_matrix = actions_to_matrix(actions)

        else:
            self.policy_matrix = policy_matrix
        
    def __call__(self, tau):
        return self.policy_matrix[tau]


def actions_to_matrix(actions):
    """
    Convert a list of action indices to a one-hot encoded matrix.
    Example: [0,2,1] -> [[1,0,0,0], [0,0,1,0], [0,1,0,0]]
    
    Args:
        actions: List of action indices (0-3 for up, down, left, right)
        
    Returns:
        One-hot encoded numpy array with shape (len(actions), 4)
    """
    
    # Create empty matrix of zeros
    matrix = np.zeros((len(actions), 4), dtype=int)
    
    # Set the appropriate positions to 1
    for i, action in enumerate(actions):
        matrix[i, action] = 1
        
    return matrix