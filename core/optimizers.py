import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def _get_nested_attr(self, obj, attr_path):
        """Get a nested attribute using dot notation"""
        for attr in attr_path.split('.'):
            obj = getattr(obj, attr)
        return obj
    
    def _set_nested_attr(self, obj, attr_path, value):
        """Set a nested attribute using dot notation"""
        attrs = attr_path.split('.')
        for attr in attrs[:-1]:
            obj = getattr(obj, attr)
        setattr(obj, attrs[-1], value)
    
    def apply_gradients(self, grads_and_vars):
        """
        Apply gradients to variables.
        Args:
            grads_and_vars: List of (gradient, (distribution, var_idx)) pairs.
        """
        for grad, (dist, var_idx) in grads_and_vars:
            if grad is not None:
                var_path = dist.variables[var_idx]
                current_value = self._get_nested_attr(dist, var_path)
                self._set_nested_attr(dist, var_path, current_value - self.learning_rate * grad)
    
    def compute_gradients(self, loss_fn, distribution):
        """
        Compute numerical gradients for all variables in the distribution.
        Args:
            loss_fn: Function that returns the loss value.
            distribution: Distribution object with variables list of variable paths.
        Returns:
            List of (gradient, (distribution, var_idx)) pairs.
        """
        grads_and_vars = []
        eps = 1e-5
        
        # Compute base loss
        base_loss = loss_fn()
        
        # Compute gradient for each variable
        for var_idx in range(len(distribution.variables)):
            var_path = distribution.variables[var_idx]
            original_value = self._get_nested_attr(distribution, var_path)
            
            # Compute loss with positive perturbation
            self._set_nested_attr(distribution, var_path, original_value + eps)
            loss_plus = loss_fn()
            
            # Compute loss with negative perturbation
            self._set_nested_attr(distribution, var_path, original_value - eps)
            loss_minus = loss_fn()
            
            # Restore original value
            self._set_nested_attr(distribution, var_path, original_value)
            
            # Compute central difference gradient
            grad = (loss_plus - loss_minus) / (2 * eps)
            grads_and_vars.append((grad, (distribution, var_idx)))
            
            # Print debug info
            print(f"Variable {var_path}:")
            print(f"  Original: {original_value}")
            print(f"  Loss plus: {loss_plus}")
            print(f"  Loss minus: {loss_minus}")
            print(f"  Gradient: {grad}")
        
        return grads_and_vars 