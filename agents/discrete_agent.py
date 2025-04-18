from core.distributions import Discrete, ConditionalDiscrete
from agents.base import Agent
from core.utils import logits2p

class DiscreteAgent(Agent):
    def __init__(self, px_vector, machina_type='matrix', **machina_params):
        super().__init__()
        
        # Initialize distributions
        self.px = Discrete(probs=logits2p(px_vector))  # Prior over x
        self.qx = Discrete(probs=logits2p(px_vector))  # Approximate posterior over x
        self.py_x = ConditionalDiscrete(machina_type=machina_type, machina_params=machina_params)