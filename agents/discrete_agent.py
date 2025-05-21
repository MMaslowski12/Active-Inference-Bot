from core.distributions import DiscreteDistribution
from core.conditional_distributions import ConditionalDiscrete
from agents.base import Agent
from core.utils import logits2p
from applications.maze.generative_model.mapping import state_to_index, index_to_state
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
        self.A = np.array(machina_params['A'])
        self.transitioner = transitioner

    def adjust_q(self, y):
        super().adjust_q(y)
        # Handle probability constraints for Discrete distributions
        probs = self.qx.get_probabilities()
        
        # Ensure minimum probability of 1% for each action
        min_prob = 0.005
        n = len(probs)
        
        # Calculate how much probability mass we need to add to reach minimum
        below_min = probs < min_prob
        if np.any(below_min):
            # Calculate total deficit
            deficit = np.sum(min_prob - probs[below_min])
            # Calculate how much to take from above-min probabilities
            above_min = ~below_min
            if np.any(above_min):
                # Take proportionally from above-min probabilities
                excess = probs[above_min] - min_prob
                total_excess = np.sum(excess)
                if total_excess > 0:
                    # Scale down above-min probabilities to compensate
                    probs[above_min] -= (excess * deficit / total_excess)
                    # Set below-min probabilities to minimum
                    probs[below_min] = min_prob
                else:
                    # If we can't take enough from above-min, distribute evenly
                    probs = np.ones(n) / n
            else:
                # If all probabilities are below min, distribute evenly
                probs = np.ones(n) / n
        
        # Convert back to logits
        self.qx.logits = np.log(probs)

    def calculate_efe(self, state, pi, tau=1):    
        entropy = self.calculate_entropy() #Done in the machina. Entropy of observations for each state
        # Convert integer state to DiscreteDistribution
        s_pi_t = self._get_s_pi_t(state, pi, tau) #Ah, thats just probabilities on whats gonna happen. simple easy-peasy, transitions
        o_pi_t = self._get_o_pi_t(s_pi_t) #Thats simple. py_x(s_pi_t)
        c_t = self.c.get_probabilities() #and thats what I want. 50/50 between "im left, reward's left" and "i'm right, reward's right"
        zeta = np.log(o_pi_t+eps)-np.log(c_t+eps)

        ambiguity = entropy @ s_pi_t
        risk = o_pi_t @ zeta

        return ambiguity + risk

    def _get_s_pi_t(self, state, pi, tau):
        for _ in range (tau):
            state = self.transitioner(state=state, action=pi(tau))

        return state.get_probabilities()

    def _get_o_pi_t(self, s_pi_t):
        return self.py_x(s_pi_t, vector_input=True).get_probabilities()

    def calculate_entropy(self):
        return -np.sum(self.A*np.log(self.A+eps), axis=0)
    
    
