from worlds.base import World
from applications.maze.environment import MazeGame

class MazeWorld(World):
    def __init__(self, environment: MazeGame = None, machina_type='discrete', **machina_params):
        """Initialize the maze world with a maze environment"""
        super().__init__(environment or MazeGame(), machina_type, **machina_params)
    
    def step(self, action):
        """Apply an action to move in the maze"""
        return self._environment.apply(action)
    
    def observe(self):
        """Return the current observation (position) in the maze"""
        return self._machina(self._environment.get_state())
    
    def _get_state(self):
        """Return the current state (position) in the maze"""
        return self._environment.get_state() 