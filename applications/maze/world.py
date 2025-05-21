from worlds.base import World
from applications.maze.environment import MazeGame

#same as the generative model, but doesnt have to be
from applications.maze.generative_model.matrices import determine_observation, get_observation_idx

class MazeWorld(World):
    def __init__(self, environment: MazeGame = None, machina_type='discrete', **machina_params):
        """Initialize the maze world with a maze environment"""
        super().__init__(environment or MazeGame(), machina_type, **machina_params)
    
    def step(self, action):
        """Apply an action to move in the maze"""
        return self._environment.apply(action)
    
    def observe(self):
        """Return the current observation (position) in the maze as a single index"""
        state = self._environment.get_state()
        player_obs, stimulus = determine_observation(state=state)

        return get_observation_idx(player_obs, stimulus)
    
    def _get_state(self):
        """Return the current state (position) in the maze"""
        return self._environment.get_state() 

    def display(self):
        """Display the current state of the maze"""
        self._environment.display()