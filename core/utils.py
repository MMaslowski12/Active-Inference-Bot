import numpy as np

def logits2p(logits):
    """
    Convert logits to probabilities using softmax.
    
    Args:
        logits: Array of logits
        
    Returns:
        Array of probabilities
    """
    return np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

def p2logits(p):
    """
    Convert probabilities to logits using log.
    
    Args:
        p: Array of probabilities
        
    Returns:
        Array of logits
    """
    return np.log(p + 1e-10)  # Add small epsilon to avoid log(0) 