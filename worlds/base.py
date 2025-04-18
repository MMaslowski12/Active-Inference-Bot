from core.machinas import MachinaGenerator
from environments.base import Environment

class World:
    def __init__(self, environment: Environment, machina_type='linear', **machina_params):
        self._environment = environment
        self._machina = MachinaGenerator.create(machina_type, **machina_params)
    
    def step(self, action=1):
        """Apply an action to the environment"""
        return self._environment.apply(action)
    
    def observe(self):
        """Return the observation based on the machina type applied to environment state"""
        return self._machina(self._environment.get_state())
    
    def _get_state(self):
        """Return the current state of the environment"""
        return self._environment.get_state()
