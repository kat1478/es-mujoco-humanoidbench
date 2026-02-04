"""
Evolution Strategies Algorithm Implementation.

Based on: Salimans et al. (2017) "Evolution Strategies as a Scalable Alternative to RL"
https://arxiv.org/abs/1703.03864

This is a simplified single-threaded version for testing and debugging.
Key features implemented:
- Antithetic sampling (mirrored sampling)
- Fitness shaping (rank transformation)
- Weight decay regularization
- Observation normalization
"""

import numpy as np
import gymnasium as gym
from typing import Tuple, List, Optional, Callable
from dataclasses import dataclass
import time

from policy import MLPPolicy
from utils import compute_centered_ranks, ObservationNormalizer


@dataclass
class ESConfig:
    """Configuration for Evolution Strategies."""
    
    # Population
    population_size: int = 40          # Number of perturbations (n in paper)
    
    # Noise
    sigma: float = 0.02                # Noise standard deviation
    
    # Optimization
    learning_rate: float = 0.01        # Step size (alpha)
    weight_decay: float = 0.005        # L2 regularization
    
    # Training
    max_iterations: int = 1000         # Number of ES iterations
    timesteps_per_iteration: int = 4800  # Approx timesteps per iteration
    
    # Episode
    max_episode_steps: int = 1000      # Max steps per episode
    
    # Normalization
    normalize_observations: bool = True
    normalize_rewards: bool = True     # Use fitness shaping
    
    # Logging
    log_interval: int = 10             # Print stats every N iterations
    eval_interval: int = 50            # Evaluate policy every N iterations
    
    # Reproducibility
    seed: int = 42


class EvolutionStrategies:
    """
    Evolution Strategies optimizer for reinforcement learning.
    
    Algorithm (simplified):
    1. Sample n perturbations ε_i from N(0, I)
    2. Evaluate perturbed parameters θ + σε_i and θ - σε_i (antithetic)
    3. Compute fitness-shaped rewards
    4. Update: θ ← θ + α/(nσ) * Σ F_i * ε_i
    """
    
    def __init__(
        self,
        env_name: str,
        config: Optional[ESConfig] = None,
        policy_hidden_sizes: Tuple[int, ...] = (64, 64)
    ):
        """
        Initialize ES optimizer.
        
        Args:
            env_name: Gymnasium environment name
            config: ES configuration
            policy_hidden_sizes: Hidden layer sizes for policy network
        """
        self.env_name = env_name
        self.config = config or ESConfig()
        
        # Set random seed
        np.random.seed(self.config.seed)
        
        # Create environment to get dimensions
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        
        # Action bounds for clipping
        self.act_low = self.env.action_space.low
        self.act_high = self.env.action_space.high
        
        print(f"Environment: {env_name}")
        print(f"  Observation dim: {self.obs_dim}")
        print(f"  Action dim: {self.act_dim}")
        print(f"  Action bounds: [{self.act_low[0]:.1f}, {self.act_high[0]:.1f}]")
        
        # Create policy
        self.policy = MLPPolicy(self.obs_dim, self.act_dim, policy_hidden_sizes)
        self.num_params = self.policy.num_params
        print(f"  Policy parameters: {self.num_params:,}")
        
        # Current parameters
        self.theta = self.policy.get_flat_params()
        
        # Observation normalizer
        if self.config.normalize_observations:
            self.obs_normalizer = ObservationNormalizer(self.obs_dim)
        else:
            self.obs_normalizer = None
            
        # Training statistics
        self.total_timesteps = 0
        self.iteration = 0
        self.best_reward = -np.inf
        self.reward_history = []
        
    def _evaluate_policy(
        self,
        params: np.ndarray,
        render: bool = False,
        update_normalizer: bool = True
    ) -> Tuple[float, int]:
        """
        Evaluate policy with given parameters.
        
        Args:
            params: Flat parameter vector
            render: Whether to render environment
            update_normalizer: Whether to update observation normalizer
            
        Returns:
            Tuple of (total_reward, episode_length)
        """
        # Set policy parameters
        self.policy.set_flat_params(params)
        
        # Reset environment
        obs, _ = self.env.reset()
        
        total_reward = 0.0
        steps = 0
        
        for _ in range(self.config.max_episode_steps):
            # Normalize observation
            if self.obs_normalizer is not None:
                if update_normalizer:
                    self.obs_normalizer.update(obs)
                obs_normalized = self.obs_normalizer.normalize(obs)
            else:
                obs_normalized = obs
            
            # Get action
            action = self.policy.act(obs_normalized)
            
            # Clip action to valid range
            action = np.clip(action, self.act_low, self.act_high)
            
            # Step environment
            obs, reward, terminated, truncated, _ = self.env.step(action)
            
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
                
            if render:
                self.env.render()
                
        return total_reward, steps
    
    def _compute_gradient(
        self,
        noise: np.ndarray,
        rewards_pos: np.ndarray,
        rewards_neg: np.ndarray
    ) -> np.ndarray:
        """
        Compute ES gradient estimate.
        
        Uses antithetic sampling: for each noise ε, we evaluate both
        θ + σε and θ - σε, then combine the results.
        
        Args:
            noise: Array of noise vectors, shape (n, num_params)
            rewards_pos: Rewards for θ + σε
            rewards_neg: Rewards for θ - σε
            
        Returns:
            Gradient estimate
        """
        n = len(rewards_pos)
        
        # Combine rewards from antithetic pairs
        # The gradient contribution from pair (ε, -ε) is:
        # F(θ+σε)ε + F(θ-σε)(-ε) = (F(θ+σε) - F(θ-σε))ε
        reward_diff = rewards_pos - rewards_neg
        
        # Apply fitness shaping (rank transformation)
        if self.config.normalize_rewards:
            # Combine all rewards for ranking
            all_rewards = np.concatenate([rewards_pos, rewards_neg])
            ranks = compute_centered_ranks(all_rewards)
            ranks_pos = ranks[:n]
            ranks_neg = ranks[n:]
            reward_diff = ranks_pos - ranks_neg
        
        # Compute gradient: (1/nσ) * Σ (F+ - F-) * ε
        gradient = np.zeros(self.num_params, dtype=np.float32)
        for i in range(n):
            gradient += reward_diff[i] * noise[i]
        
        gradient /= (n * self.config.sigma)
        
        # Add weight decay
        gradient -= self.config.weight_decay * self.theta
        
        return gradient
    
    def train_iteration(self) -> dict:
        """
        Perform one ES iteration.
        
        Returns:
            Dictionary with training statistics
        """
        n = self.config.population_size // 2  # Half for antithetic pairs
        
        # Sample noise vectors
        noise = np.random.randn(n, self.num_params).astype(np.float32)
        
        # Evaluate perturbed parameters
        rewards_pos = np.zeros(n, dtype=np.float32)
        rewards_neg = np.zeros(n, dtype=np.float32)
        total_steps = 0
        
        for i in range(n):
            # Positive perturbation: θ + σε
            params_pos = self.theta + self.config.sigma * noise[i]
            rewards_pos[i], steps = self._evaluate_policy(params_pos)
            total_steps += steps
            
            # Negative perturbation: θ - σε (antithetic)
            params_neg = self.theta - self.config.sigma * noise[i]
            rewards_neg[i], steps = self._evaluate_policy(params_neg)
            total_steps += steps
        
        # Compute gradient and update parameters
        gradient = self._compute_gradient(noise, rewards_pos, rewards_neg)
        self.theta += self.config.learning_rate * gradient
        
        # Update statistics
        self.total_timesteps += total_steps
        self.iteration += 1
        
        # Compute stats
        all_rewards = np.concatenate([rewards_pos, rewards_neg])
        mean_reward = np.mean(all_rewards)
        max_reward = np.max(all_rewards)
        min_reward = np.min(all_rewards)
        std_reward = np.std(all_rewards)
        
        self.reward_history.append(mean_reward)
        
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
        
        return {
            'iteration': self.iteration,
            'timesteps': self.total_timesteps,
            'mean_reward': mean_reward,
            'max_reward': max_reward,
            'min_reward': min_reward,
            'std_reward': std_reward,
            'best_reward': self.best_reward,
            'episodes': 2 * n,
            'steps_this_iter': total_steps
        }
    
    def evaluate(self, num_episodes: int = 5) -> float:
        """
        Evaluate current policy without noise.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Mean reward over episodes
        """
        rewards = []
        for _ in range(num_episodes):
            reward, _ = self._evaluate_policy(
                self.theta,
                update_normalizer=False
            )
            rewards.append(reward)
        return np.mean(rewards)
    
    def train(self, callback: Optional[Callable] = None) -> List[dict]:
        """
        Run full ES training.
        
        Args:
            callback: Optional callback function called after each iteration
            
        Returns:
            List of training statistics for each iteration
        """
        print(f"\nStarting ES training for {self.config.max_iterations} iterations...")
        print(f"  Population size: {self.config.population_size}")
        print(f"  Sigma: {self.config.sigma}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print()
        
        history = []
        start_time = time.time()
        
        for _ in range(self.config.max_iterations):
            # Training iteration
            stats = self.train_iteration()
            history.append(stats)
            
            # Logging
            if self.iteration % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                fps = self.total_timesteps / elapsed
                
                print(f"Iter {stats['iteration']:4d} | "
                      f"Timesteps: {stats['timesteps']:8,} | "
                      f"Reward: {stats['mean_reward']:8.1f} ± {stats['std_reward']:6.1f} | "
                      f"Best: {stats['best_reward']:8.1f} | "
                      f"FPS: {fps:6.0f}")
            
            # Evaluation
            if self.iteration % self.config.eval_interval == 0:
                eval_reward = self.evaluate()
                print(f"  >>> Evaluation reward: {eval_reward:.1f}")
            
            # Callback
            if callback is not None:
                callback(stats)
        
        total_time = time.time() - start_time
        print(f"\nTraining complete!")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Total timesteps: {self.total_timesteps:,}")
        print(f"  Best reward: {self.best_reward:.1f}")
        
        return history
    
    def save(self, filepath: str):
        """Save policy parameters."""
        np.savez(
            filepath,
            theta=self.theta,
            obs_mean=self.obs_normalizer.mean if self.obs_normalizer else None,
            obs_var=self.obs_normalizer.var if self.obs_normalizer else None,
            obs_count=self.obs_normalizer.count if self.obs_normalizer else None
        )
        print(f"Saved policy to {filepath}")
    
    def load(self, filepath: str):
        """Load policy parameters."""
        data = np.load(filepath)
        self.theta = data['theta']
        if self.obs_normalizer and data['obs_mean'] is not None:
            self.obs_normalizer.mean = data['obs_mean']
            self.obs_normalizer.var = data['obs_var']
            self.obs_normalizer.count = float(data['obs_count'])
        print(f"Loaded policy from {filepath}")


def test_es():
    """Quick test of ES on a simple environment."""
    print("=" * 60)
    print("Testing Evolution Strategies on Pendulum-v1")
    print("=" * 60)
    
    config = ESConfig(
        population_size=20,
        sigma=0.1,
        learning_rate=0.03,
        max_iterations=30,
        max_episode_steps=200,
        log_interval=5,
        eval_interval=10
    )
    
    es = EvolutionStrategies(
        env_name="Pendulum-v1",
        config=config
    )
    
    history = es.train()
    
    print("\nFinal evaluation:")
    final_reward = es.evaluate(num_episodes=10)
    print(f"Mean reward over 10 episodes: {final_reward:.1f}")
    
    return history


if __name__ == "__main__":
    test_es()
