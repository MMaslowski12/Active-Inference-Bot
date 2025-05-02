from core.distributions import DiscreteDistribution
from core.conditional_distributions import ConditionalDiscrete
from agents.base import Agent
from core.utils import logits2p
import numpy as np
eps=1e-16

class DiscreteAgent(Agent):
    def __init__(self, px_vector, c_vector, transitioner, machina_type='matrix', q_learning_rate = 0.1, **machina_params):
        super().__init__(q_learning_rate)
        
        # Initialize distributions
        self.px = DiscreteDistribution(logits=px_vector)  # Prior over x
        self.qx = DiscreteDistribution(logits=px_vector)  # Approximate posterior over x
        self.py_x = ConditionalDiscrete(machina_type=machina_type, machina_params=machina_params)
        self.c = DiscreteDistribution(logits=c_vector)
        self.A = machina_params
        self.transitioner = None #Sth sth

    def calculate_efe(self, state, pi, tau=1):
        if tau > 1:
            raise ValueError(f"You wish. Max time_step is 1, given: {tau}")
    
        entropy = self.calculate_entropy() #Do I do this in here, in machinas, or in discrete agents? Do this in the machina.....
        s_pi_t = self.get_s_pi_t(state, pi, tau) #Ah, thats just probabilities on whats gonna happen. simple easy-peasy, transitions
        o_pi_t = self.get_o_pi_t(s_pi_t) #Thats simple. py_x(s_pi_t)
        c_t = self.c.get_probabilities() #and thats what I want. 50/50 between "im left, reward's left" and "i'm right, reward's right"
        zeta = np.log(o_pi_t+eps)-np.log(c_t+eps)

        ambiguity = entropy @ s_pi_t
        risk = o_pi_t @ zeta

        return ambiguity + risk

    def _get_s_pi_t(self, pi, tau, current_state):
        return self.transitioner(current_state, pi).get_probabilities()

    def _get_o_pi_t(self, s_pi_t):
        return self.py_x(s_pi_t).get_probabilities()

    def calculate_entropy(self):
        return -np.sum(self.A*np.log(self.A+eps), axis=0)
    

'''
TODO

C -- 50/50 0/7
Transition -- just a simple AF function. I mean not as simple, it'd have to be something close to simulating the whole game to understand it

'''