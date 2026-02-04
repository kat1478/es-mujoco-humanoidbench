"""
Policy network for Evolution Strategies.
Based on: Salimans et al. (2017) "Evolution Strategies as a Scalable Alternative to RL"

Architecture: MLP with 2 hidden layers of 64 units, tanh activation (for MuJoCo tasks)
"""

import numpy as np


class MLPPolicy:
    """
    Simple MLP policy for continuous control.
    
    Architecture matches the original ES paper:
    - 2 hidden layers with 64 units each
    - tanh activation functions
    - Linear output layer
    """
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: tuple = (64, 64)):
        """
        Initialize policy network.
        
        Args:
            obs_dim: Dimension of observation space
            act_dim: Dimension of action space  
            hidden_sizes: Tuple with sizes of hidden layers
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_sizes = hidden_sizes
        
        # Initialize weights and biases
        self.params = self._init_params()
        self.num_params = self._count_params()
        
    def _init_params(self) -> dict:
        """Initialize network parameters with Xavier initialization."""
        params = {}
        
        # Layer sizes: input -> hidden1 -> hidden2 -> output
        layer_sizes = [self.obs_dim] + list(self.hidden_sizes) + [self.act_dim]
        
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            
            # Xavier initialization
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            
            params[f'W{i}'] = np.random.randn(fan_in, fan_out).astype(np.float32) * scale
            params[f'b{i}'] = np.zeros(fan_out, dtype=np.float32)
            
        return params
    
    def _count_params(self) -> int:
        """Count total number of parameters."""
        total = 0
        for key, value in self.params.items():
            total += value.size
        return total
    
    def get_flat_params(self) -> np.ndarray:
        """Flatten all parameters into a single vector."""
        flat = []
        for i in range(len(self.hidden_sizes) + 1):
            flat.append(self.params[f'W{i}'].flatten())
            flat.append(self.params[f'b{i}'].flatten())
        return np.concatenate(flat).astype(np.float32)
    
    def set_flat_params(self, flat_params: np.ndarray):
        """Set parameters from a flat vector."""
        idx = 0
        layer_sizes = [self.obs_dim] + list(self.hidden_sizes) + [self.act_dim]
        
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            
            # Weights
            w_size = fan_in * fan_out
            self.params[f'W{i}'] = flat_params[idx:idx + w_size].reshape(fan_in, fan_out)
            idx += w_size
            
            # Biases
            b_size = fan_out
            self.params[f'b{i}'] = flat_params[idx:idx + b_size]
            idx += b_size
    
    def forward(self, obs: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            obs: Observation array of shape (obs_dim,) or (batch, obs_dim)
            
        Returns:
            Action array of shape (act_dim,) or (batch, act_dim)
        """
        x = obs.astype(np.float32)
        
        # Hidden layers with tanh activation
        for i in range(len(self.hidden_sizes)):
            x = x @ self.params[f'W{i}'] + self.params[f'b{i}']
            x = np.tanh(x)
        
        # Output layer (linear)
        last_idx = len(self.hidden_sizes)
        x = x @ self.params[f'W{last_idx}'] + self.params[f'b{last_idx}']
        
        return x
    
    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Select action for given observation.
        
        Args:
            obs: Single observation
            
        Returns:
            Action array
        """
        return self.forward(obs)
    
    def copy(self) -> 'MLPPolicy':
        """Create a copy of this policy."""
        new_policy = MLPPolicy(self.obs_dim, self.act_dim, self.hidden_sizes)
        new_policy.set_flat_params(self.get_flat_params().copy())
        return new_policy


def test_policy():
    """Simple test to verify policy works correctly."""
    print("Testing MLPPolicy...")
    
    # Create policy for HalfCheetah-like environment
    obs_dim = 17  # HalfCheetah observation dimension
    act_dim = 6   # HalfCheetah action dimension
    
    policy = MLPPolicy(obs_dim, act_dim)
    
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {act_dim}")
    print(f"  Total parameters: {policy.num_params}")
    
    # Test forward pass
    obs = np.random.randn(obs_dim).astype(np.float32)
    action = policy.act(obs)
    
    print(f"  Input shape: {obs.shape}")
    print(f"  Output shape: {action.shape}")
    print(f"  Sample action: {action[:3]}...")
    
    # Test parameter get/set
    flat = policy.get_flat_params()
    print(f"  Flat params shape: {flat.shape}")
    
    policy.set_flat_params(flat + 0.01)
    action2 = policy.act(obs)
    
    assert not np.allclose(action, action2), "Parameters should have changed!"
    print("  Parameter update: OK")
    
    print("Policy test PASSED!\n")


if __name__ == "__main__":
    test_policy()
