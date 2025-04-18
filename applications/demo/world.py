from worlds.base import World
from applications.demo.environment import SingletonEnvironment

class DemoWorld(World):
    def __init__(self, environment: SingletonEnvironment = None, machina_type='linear', **machina_params):
        super().__init__(environment or SingletonEnvironment(), machina_type, **machina_params)
    
    def step(self, action=1):
        """Apply an action to the environment"""
        return self._environment.apply(action)
    
    def observe(self):
        """Return the observation based on the machina type applied to environment state"""
        return self._machina(self._environment.get_state())
    
    def _get_state(self):
        """Return the current state of the environment"""
        return self._environment.get_state() 