from environments.base import Environment

class SingletonEnvironment(Environment):
    def __init__(self, initial_state=0):
        self._state = initial_state
    
    def get_state(self):
        return self._state
    
    def apply(self, action):
        """Increment the state by the action value"""
        self._state += action
        return self._state 