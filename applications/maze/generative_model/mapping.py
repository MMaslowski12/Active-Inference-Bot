"""
State index mapping utility for T-maze.
Ensures consistent mapping between (position, reward) and flat index.

Canonical mapping: index = reward * 5 + position

Positions: 0=Top Left, 1=Top Mid, 2=Top Right, 3=Center, 4=Center Down
Rewards: 0=Left, 1=Right

Usage:
    idx = state_to_index(position, reward)
    pos, rew = index_to_state(idx)
"""

NUM_POSITIONS = 5
NUM_REWARDS = 2
NUM_PLAYER_OBS = 6
NUM_STIMULI = 3

POSITION_NAMES = ["Top Left", "Top Mid", "Top Right", "Center", "Center Down"]
REWARD_NAMES = ["Left", "Right"]

def state_to_index(position: int, reward: int) -> int:
    """Map (position, reward) to canonical flat index."""
    return reward * NUM_POSITIONS + position

def index_to_state(index: int) -> tuple:
    """Map canonical flat index to (position, reward)."""
    reward = index // NUM_POSITIONS
    position = index % NUM_POSITIONS
    return position, reward

def get_observation_idx(player_obs: int, stimulus: int) -> int:
    """Convert player observation and internal stimulus into an observation index.
    
    Args:
        player_obs: int (0-5) representing player position
        stimulus: int (0-2) representing internal stimulus (positive, neutral, negative)
        
    Returns:
        int: observation index (0-17)
    """
    return player_obs + NUM_PLAYER_OBS * stimulus 