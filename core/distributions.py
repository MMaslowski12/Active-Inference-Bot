import numpy as np
from abc import ABC, abstractmethod
from .machinas import LinearMachina, QuadraticMachina, MachinaGenerator
EPS=1e-10

class Distribution(ABC):
    @abstractmethod
    def sample(self):
        """Generate a random sample from the distribution"""
        pass
    
    @abstractmethod
    def probability(self, x):
        """Compute the probability density of x under this distribution"""
        pass
    
    def kl_divergence(self, other, num_samples=1000):
        """
        Compute KL divergence using Monte Carlo estimation
        KL(p||q) = E_p[log p(x) - log q(x)]
        """
        # Generate samples from this distribution (p)
        samples = [self.sample() for _ in range(num_samples)]
        
        # Compute the Monte Carlo estimate
        kl_estimate = 0.0
        epsilon = 1e-10  # Small constant for numerical stability
        for x in samples:
            kl_estimate += np.log(self.probability(x) + EPS) - np.log(other.probability(x) + EPS)
        
        return kl_estimate / num_samples
    
    def negative_expected_log(self, other, y, num_samples=1000):
        """
        Calculate -E_Q(x)[ln P(y|x)] where:
        - self represents Q(x)
        - other(x) returns P(y|x) for a given x
        - y is the observed output
        """
        # Generate samples from Q(x)
        x_samples = [self.sample() for _ in range(num_samples)]
        
        # Compute the Monte Carlo estimate
        neg_log_estimate = 0.0
        for x in x_samples:
            # Get P(y|x) distribution for this x
            p_y_given_x = other(x)
            # Calculate probability of y under P(y|x)
            neg_log_estimate -= np.log(p_y_given_x.probability(y) + EPS)
        
        return neg_log_estimate / num_samples

class DiscreteDistribution(Distribution):
    def __init__(self, logits):
        """
        Initialize a discrete distribution with given logits
        Args:
            logits: numpy array of logits (unconstrained values)
        """
        self.logits = np.array(logits)
        self.n = len(logits)
        self.variables = [f'logits[{i}]' for i in range(self.n)]  # Each logit is independently optimizable
    
    def get_probabilities(self):
        """Convert logits to probabilities using softmax"""
        # Numerically stable softmax that preserves gradients
        # Subtract max for numerical stability, but store the max value
        max_logit = np.max(self.logits)
        # Ensure logits don't get too negative to prevent underflow
        min_threshold = -100  # exp(-100) is still representable
        shifted_logits = np.maximum(self.logits - max_logit, min_threshold)
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / np.sum(exp_logits)
        
        return probs
    
    def sample(self):
        """Generate a random sample from the discrete distribution"""
        return np.random.choice(self.n, p=self.get_probabilities())
    
    def probability(self, x):
        """Compute the probability of x under this distribution"""
        if not isinstance(x, (int, np.integer)) or x < 0 or x >= self.n:
            return 0.0
        return self.get_probabilities()[x]
    
    def kl_divergence(self, other):
        """
        Compute KL divergence between two discrete distributions
        KL(p||q) = Σ p(x) * (log p(x) - log q(x))
        """
        if not isinstance(other, DiscreteDistribution) or other.n != self.n:
            raise ValueError("KL divergence can only be computed between two Discrete distributions of the same size")
        
        # Get probabilities directly
        p = self.get_probabilities()
        q = other.get_probabilities()
        
        # Add small epsilon to avoid log(0)
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        
        # Compute KL divergence directly
        kl = np.sum(p * (np.log(p) - np.log(q)))
        
        return kl
    
    def negative_expected_log(self, conditional_dist, y):
        """
        Calculate -E_Q(x)[ln P(y|x)] where:
        - self (Q(x)) is a Discrete distribution
        - other(x) returns P(y|x) which is assumed to be a Discrete distribution
        - y is the observed output value
        """
        # Get probabilities directly
        q = self.get_probabilities()
        q = np.clip(q, 1e-10, 1.0)
        
        neg_log_estimate = 0.0
        for x in range(self.n):
            p_y_given_x = conditional_dist(x)
            if not isinstance(p_y_given_x, DiscreteDistribution):
                raise ValueError("other(x) must return a Discrete distribution")
            
            # Get probability of y under P(y|x)
            p_y = p_y_given_x.probability(y)
            p_y = np.clip(p_y, 1e-10, 1.0)
            
            # Add contribution: -Q(x) * log P(y|x)
            neg_log_estimate -= q[x] * np.log(p_y)
        
        return neg_log_estimate

class Normal(Distribution):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
        self.variables = ['mean', 'std']  # Define optimizable parameters
    
    def sample(self):
        """Generate a random sample from the normal distribution"""
        return np.random.normal(self.mean, self.std)
    
    def probability(self, x):
        """Compute the probability density of x under this normal distribution"""
        return (1.0 / (self.std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - self.mean) / self.std)**2)
    
    def kl_divergence(self, other):
        """
        Compute KL divergence between two normal distributions
        KL(p||q) = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2
        """
        if not isinstance(other, Normal):
            raise ValueError("KL divergence can only be computed between two Normal distributions")
        
        var1 = self.std ** 2
        var2 = other.std ** 2
        return (np.log(other.std/self.std) + 
                (var1 + (self.mean - other.mean)**2)/(2*var2) - 0.5)
    
    def negative_expected_log(self, conditional_dist, y, num_samples=None):
        """
        Calculate -E_Q(x)[ln P(y|x)] analytically where:
        - self (Q(x)) is a Normal distribution
        - other(x) returns P(y|x) which is assumed to be a Normal distribution
        - y is the observed output value
        
        For normal distributions, this has an analytical solution:
        -E_Q(x)[ln P(y|x)] = 0.5 * ln(2πσ₂²) + 0.5 * (σ₁² + (μ₁-μ₂)² + σ₂²)/σ₂²
        where:
        - Q(x) ~ N(μ₁, σ₁)
        - P(y|x) ~ N(μ₂(x), σ₂)
        """
        # Get P(y|x) distribution for the current mean of Q(x)
        p_y_given_x = conditional_dist(self.mean)
        if not isinstance(p_y_given_x, Normal):
            raise ValueError("other(x) must return a Normal distribution")
            
        # Extract parameters
        var1 = self.std ** 2  # σ₁²
        var2 = p_y_given_x.std ** 2  # σ₂²
        mu1 = self.mean  # μ₁
        mu2 = p_y_given_x.mean  # μ₂
        
        # For linear machina: μ₂(x) = b1*x + b0
        # For quadratic machina: μ₂(x) = a*x² + b*x + c
        # We need to compute E_Q(x)[(y - μ₂(x))²]
        
        # For linear machina:
        if isinstance(conditional_dist.machina, LinearMachina):
            b1 = conditional_dist.machina.b1
            b0 = conditional_dist.machina.b0
            # E[(y - (b1*x + b0))²] = (y - b1*μ₁ - b0)² + b1²*σ₁²
            expected_squared_error = (y - b1*mu1 - b0)**2 + (b1**2)*var1
        # For quadratic machina:
        elif isinstance(conditional_dist.machina, QuadraticMachina):
            a = conditional_dist.machina.a
            b = conditional_dist.machina.b
            c = conditional_dist.machina.c
            # E[(y - (a*x² + b*x + c))²] = (y - a*E[x²] - b*μ₁ - c)² + a²*Var[x²] + b²*σ₁²
            # E[x²] = μ₁² + σ₁²
            # Var[x²] = 4*μ₁²*σ₁² + 2*σ₁⁴
            ex2 = mu1**2 + var1
            varx2 = 4*mu1**2*var1 + 2*var1**2
            expected_squared_error = (y - a*ex2 - b*mu1 - c)**2 + a**2*varx2 + b**2*var1
        else:
            raise ValueError("Unsupported machina type for analytical solution")
        
        # Compute the analytical solution
        return 0.5 * (np.log(2 * np.pi * var2) + expected_squared_error / var2) 