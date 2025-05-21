'''

STATE:
- Where am I? [5]
- Where is the reward? [2]

TL
TC
TR
C
CD

OBSERVATION:
- Player position (inc. info on reward) [6]
- Internal stimulus (nothing, good, bad) [3]

A = player position simple; reward = dependent on whats going on. 
B = transition
C = preference to the good one, anti-preference to the bad one
D = uniform over reward
'''

from .mapping import state_to_index, get_observation_idx

def determine_observation(state):
    """Determine the observation for a given state.
    
    Args:
        state: int (0-9) representing the state
        
    Returns:
        tuple: (player_obs, stimulus) where:
        - player_obs: int (0-5) representing player position
        - stimulus: int (0-2) representing internal stimulus
    """
    position = state % 5  # 0-4: TL, TC, TR, C, CD
    reward = state // 5   # 0-1: left, right
    
    # Determine player position observation
    if position == 4 and reward == 1:  # CD with right reward
        player_obs = 5
    else:
        player_obs = position
    
    # Determine internal stimulus
    if position == 0:  # TL
        if reward == 0:  # left reward
            stimulus = 0  # positive
        else:  # right reward
            stimulus = 2  # negative
    elif position == 2:  # TR
        if reward == 0:  # left reward
            stimulus = 2  # negative
        else:  # right reward
            stimulus = 0  # positive
    else:  # TC, C, CD
        stimulus = 1  # neutral
    
    return player_obs, stimulus

# Generate observation matrix
observation_matrix = [[0]*10 for _ in range(18)]
for state in range(10):
    player_obs, stimulus = determine_observation(state)
    observation_matrix[get_observation_idx(player_obs, stimulus)][state] = 1

# Uniform prior over all 10 states
priors_vector = [0]*10

c_vector = [0]*18
for obs in range(6):
    # Positive preference for good stimulus (stimulus=0)
    c_vector[get_observation_idx(player_obs=obs, stimulus=0)] = 6
    # Negative preference for bad stimulus (stimulus=2)
    c_vector[get_observation_idx(player_obs=obs, stimulus=2)] = -6
    # Neutral stimulus (stimulus=1) remains at 0