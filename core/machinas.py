from abc import ABC, abstractmethod
import numpy as np

class Machina(ABC):
    @abstractmethod
    def __call__(self, x):
        """Compute the output of the machina for a given input x"""
        pass

class LinearMachina(Machina):
    def __init__(self, b1, b0):
        self.b1 = b1
        self.b0 = b0
        self.variables = ['b1', 'b0']
    
    def __call__(self, x):
        return self.b1 * x + self.b0

class QuadraticMachina(Machina):
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        self.variables = ['a', 'b', 'c']
    
    def __call__(self, x):
        return self.a * x**2 + self.b * x + self.c

class MatrixMachina(Machina):
    def __init__(self, A):
        """
        Initialize a matrix machina that computes y = Ax
        Args:
            A: numpy array representing the transformation matrix
        """
        self.A = np.array(A)
        # Flatten the matrix and create variables for each element
        self.A_flat = self.A.flatten()
        self.variables = [f'A_flat[{i}]' for i in range(len(self.A_flat))]
    
    def __call__(self, x, vector_input=False):
        """Compute y = Ax for the given input x"""
        # For discrete observations, x should be a one-hot vector
        if not vector_input:
            x_onehot = np.zeros(self.A.shape[1])
            x_onehot[int(x)] = 1.0
        
        # Reshape A_flat back to matrix before multiplication
        A = self.A_flat.reshape(self.A.shape)
        
        # Return raw matrix multiplication result
        return np.dot(A, x_onehot)

class MachinaGenerator:
    @staticmethod
    def create(machina_type, **params):
        """Create a machina of the specified type with given parameters"""
        if machina_type == 'linear':
            return LinearMachina(**params)
        elif machina_type == 'quadratic':
            return QuadraticMachina(**params)
        elif machina_type == 'matrix':
            return MatrixMachina(**params)
        else:
            raise ValueError(f"Unsupported machina type: {machina_type}") 