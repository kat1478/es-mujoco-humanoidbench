"""
Utility functions for Evolution Strategies.

Includes:
- Fitness shaping (rank transformation)
- Observation normalization
- Seed management
"""

import numpy as np
from typing import List, Tuple


def compute_centered_ranks(fitness: np.ndarray) -> np.ndarray:
    """
    Compute centered ranks for fitness shaping.
    
    Rank transformation from Wierstra et al. (2014):
    - Ranks fitness values from 0 to n-1
    - Centers them to have mean 0
    - Scales to range [-0.5, 0.5]
    
    This reduces sensitivity to outliers and improves optimization stability.
    
    Args:
        fitness: Array of fitness values (rewards)
        
    Returns:
        Centered and normalized ranks
    """
    n = len(fitness)
    ranks = np.zeros(n, dtype=np.float32)
    
    # Get ranking (argsort of argsort gives ranks)
    sorted_indices = np.argsort(fitness)
    for rank, idx in enumerate(sorted_indices):
        ranks[idx] = rank
    
    # Center and normalize to [-0.5, 0.5]
    ranks = (ranks / (n - 1)) - 0.5
    
    return ranks


def compute_weight_decay(params: np.ndarray, l2_coef: float = 0.005) -> float:
    """
    Compute L2 weight decay penalty.
    
    Args:
        params: Flat parameter vector
        l2_coef: L2 regularization coefficient
        
    Returns:
        Weight decay penalty value
    """
    return l2_coef * np.sum(params ** 2)


class ObservationNormalizer:
    """
    Online observation normalizer using Welford's algorithm.
    
    Maintains running mean and variance of observations
    for stable training.
    """
    
    def __init__(self, obs_dim: int, clip: float = 5.0):
        """
        Args:
            obs_dim: Dimension of observations
            clip: Clip normalized observations to [-clip, clip]
        """
        self.obs_dim = obs_dim
        self.clip = clip
        
        # Running statistics
        self.mean = np.zeros(obs_dim, dtype=np.float64)
        self.var = np.ones(obs_dim, dtype=np.float64)
        self.count = 1e-4  # Small value to avoid division by zero
        
    def update(self, obs: np.ndarray):
        """Update running statistics with new observation(s)."""
        obs = np.asarray(obs)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
            
        batch_mean = np.mean(obs, axis=0)
        batch_var = np.var(obs, axis=0)
        batch_count = obs.shape[0]
        
        self._update_from_moments(batch_mean, batch_var, batch_count)
        
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """Update from batch moments using parallel algorithm."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        
        new_var = m2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
        
    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running statistics."""
        normalized = (obs - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)
    
    def get_state(self) -> dict:
        """Get normalizer state for saving."""
        return {
            'mean': self.mean.copy(),
            'var': self.var.copy(),
            'count': self.count
        }
    
    def set_state(self, state: dict):
        """Set normalizer state from saved data."""
        self.mean = state['mean']
        self.var = state['var']
        self.count = state['count']


class SharedNoiseTable:
    """
    Shared noise table for efficient ES implementation.
    
    Instead of communicating full perturbation vectors,
    workers share seeds that index into a common noise table.
    This dramatically reduces communication overhead.
    """
    
    def __init__(self, size: int = 250_000_000, seed: int = 42):
        """
        Args:
            size: Size of noise table (in floats)
            seed: Random seed for reproducibility
        """
        print(f"Creating shared noise table with {size:,} floats...")
        rng = np.random.RandomState(seed)
        self.noise = rng.randn(size).astype(np.float32)
        print(f"  Memory usage: {self.noise.nbytes / 1e6:.1f} MB")
        
    def get(self, idx: int, size: int) -> np.ndarray:
        """Get noise vector starting at index."""
        return self.noise[idx:idx + size]
    
    def sample_index(self, rng: np.random.RandomState, size: int) -> int:
        """Sample a valid starting index for given size."""
        return rng.randint(0, len(self.noise) - size + 1)


def test_utils():
    """Test utility functions."""
    print("Testing utility functions...")
    
    # Test centered ranks
    fitness = np.array([10, 5, 20, 15, 0])
    ranks = compute_centered_ranks(fitness)
    print(f"  Fitness: {fitness}")
    print(f"  Centered ranks: {ranks}")
    assert np.isclose(ranks.mean(), 0, atol=1e-6), "Ranks should be centered"
    print("  Centered ranks: OK")
    
    # Test observation normalizer
    normalizer = ObservationNormalizer(obs_dim=4)
    for _ in range(100):
        obs = np.random.randn(4) * 10 + 5  # Mean 5, std 10
        normalizer.update(obs)
    
    normalized = normalizer.normalize(np.array([5, 5, 5, 5]))
    print(f"  Normalized [5,5,5,5]: {normalized}")
    assert np.abs(normalized).max() < 2, "Should be close to 0 after normalization"
    print("  Observation normalizer: OK")
    
    print("Utils test PASSED!\n")


if __name__ == "__main__":
    test_utils()
