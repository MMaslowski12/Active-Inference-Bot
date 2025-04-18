from abc import ABC, abstractmethod

class Environment(ABC):
    @abstractmethod
    def get_state(self):
        """Return the current state of the environment"""
        pass
    
    @abstractmethod
    def apply(self, action):
        """Apply an action to the environment and update its state"""
        pass
