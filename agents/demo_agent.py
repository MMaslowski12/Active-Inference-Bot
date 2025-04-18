from core.distributions import Normal
from core.conditional_distributions import ConditionalNormal
from core.machinas import LinearMachina as Linear, QuadraticMachina as Quadratic
from Agents.base import Agent

class DemoAgent(Agent):
    def __init__(self, machina_type='linear', obs_noise=1.0, **machina_params):
        super().__init__()
        
        # Initialize distributions
        self.px = Normal(mean=0.0, std=1)  # Prior over x
        self.qx = Normal(mean=0.0, std=1)  # Approximate posterior over x
        self.py_x = ConditionalNormal(machina_type=machina_type, machina_params=machina_params, std=obs_noise)