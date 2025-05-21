from core.distributions import Normal
from core.optimizers import SGD
from abc import ABC, abstractmethod
import numpy as np

class Agent(ABC):
    def __init__(self, q_learning_rate=0.1):
        # Distributions to be initialized by subclasses
        self.px = None  # Prior over x
        self.qx = None  # Approximate posterior over x
        self.transition = None 
        self.py_x = None  # Observation model
        
        # Create optimizers
        self.q_optimizer = SGD(learning_rate=q_learning_rate)  # For q_mu and q_var
        self.px_optimizer = SGD(learning_rate=0.01)  # For p(x) mean
        self.py_x_optimizer = SGD(learning_rate=0.01)  # For p(y|x) parameters
    
    def calculate_complexity(self):
        """Placeholder for complexity calculation"""
        return self.qx.kl_divergence(self.px)
    
    def calculate_accuracy(self, y):
        """Placeholder for accuracy calculation"""
        return self.qx.negative_expected_log(self.py_x, y)
    
    def calculate_vfe(self, y, q_mu=None):
        """Calculate the total variational free energy: complexity + accuracy"""
        return self.calculate_complexity() + self.calculate_accuracy(y)
    
    def adjust_q(self, y):
        """
        Adjust the approximate posterior q(x) to minimize VFE.
        This should update both q_mu and q_sigma.
        """
        # Compute gradients
        loss_fn = lambda: self.calculate_vfe(y)
        grads_and_vars = self.q_optimizer.compute_gradients(loss_fn, self.qx)

        # Apply gradients
        self.q_optimizer.apply_gradients(grads_and_vars)
        
        # Ensure std stays positive for Normal distributions
        if hasattr(self.qx, 'std'):
            self.qx.std = max(0.1, self.qx.std)
    
    def learn_px(self, y):
        """
        Learn the prior p(x) by updating its parameters.
        This should update the prior distribution's parameters.
        """
        # Learn p(x)
        loss_fn = lambda: self.calculate_complexity()  # Only use complexity for p(x) learning
        grads_and_vars = self.px_optimizer.compute_gradients(loss_fn, self.px)
        self.px_optimizer.apply_gradients(grads_and_vars)
    
    def learn(self, y):
        """
        Learn from observations by updating p(x) and p(y|x).
        This should update both the prior and the observation model.
        """
        # Learn p(x)
        self.learn_px(y)
        
        # Learn p(y|x)
        self.learn_py_x(y)
    
    def learn_py_x(self, y):
        """Update the observation model p(y|x) based on the observation y"""
        # Compute gradients using the SGD optimizer
        loss_fn = lambda: self.calculate_accuracy(y)  # Only use accuracy for p(y|x) learning
        grads_and_vars = self.py_x_optimizer.compute_gradients(loss_fn, self.py_x)
        
        # Apply gradients
        self.py_x_optimizer.apply_gradients(grads_and_vars)
    

