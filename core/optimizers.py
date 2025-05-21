import numpy as np

class SGD:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
    
    def _get_nested_attr(self, obj, attr_path):
        """Get a nested attribute using dot notation or array indexing"""
        for attr in attr_path.split('.'):
            if '[' in attr and ']' in attr:
                # Handle array indexing
                attr_name = attr.split('[')[0]
                index = int(attr.split('[')[1].split(']')[0])
                obj = getattr(obj, attr_name)[index]
            else:
                obj = getattr(obj, attr)
        return obj
    
    def _set_nested_attr(self, obj, attr_path, value):
        """Set a nested attribute using dot notation or array indexing"""
        attrs = attr_path.split('.')
        for attr in attrs[:-1]:
            if '[' in attr and ']' in attr:
                # Handle array indexing
                attr_name = attr.split('[')[0]
                index = int(attr.split('[')[1].split(']')[0])
                obj = getattr(obj, attr_name)[index]
            else:
                obj = getattr(obj, attr)
        
        last_attr = attrs[-1]
        if '[' in last_attr and ']' in last_attr:
            # Handle array indexing for the last attribute
            attr_name = last_attr.split('[')[0]
            index = int(last_attr.split('[')[1].split(']')[0])
            array = getattr(obj, attr_name)
            # Ensure we're setting a scalar value
            if isinstance(value, (list, np.ndarray)):
                value = value[0] if len(value) > 0 else 0.0
            array[index] = value
        else:
            setattr(obj, last_attr, value)
    
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

        # Compute gradient for each variable
        for var_idx in range(len(distribution.variables)):
            var_path = distribution.variables[var_idx]
            original_value = self._get_nested_attr(distribution, var_path)
            og_loss = loss_fn()
            
            # Get original probabilities
            if hasattr(distribution, 'get_probabilities'):
                og_probs = distribution.get_probabilities()

            # Compute loss with positive perturbation
            self._set_nested_attr(distribution, var_path, original_value + eps)
            loss_plus = loss_fn()
            if hasattr(distribution, 'get_probabilities'):
                plus_probs = distribution.get_probabilities()
            
            # Compute loss with negative perturbation
            self._set_nested_attr(distribution, var_path, original_value - eps)
            loss_minus = loss_fn()
            if hasattr(distribution, 'get_probabilities'):
                minus_probs = distribution.get_probabilities()
            
            # Restore original value
            self._set_nested_attr(distribution, var_path, original_value)
            
            # Compute central difference gradient
            grad = (loss_plus - loss_minus) / (2 * eps)
            
            grads_and_vars.append((grad, (distribution, var_idx)))
        
        return grads_and_vars 