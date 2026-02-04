"""
Experiment: Replication of ES results on HalfCheetah-v4

Based on: Salimans et al. (2017) "Evolution Strategies as a Scalable Alternative to RL"

This script runs ES on HalfCheetah to replicate the learning curve from the paper.
Due to limited compute (8 threads vs 1440 CPUs in paper), we use fewer timesteps
but aim to verify that the algorithm converges correctly.

Expected results from paper (Table 1):
- ES matches TRPO performance at ~2.88M timesteps
- Final TRPO score at 5M timesteps: ~2386

Our goal: Show ES learns and improves on HalfCheetah with limited compute.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../src')

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time

from es_algorithm import EvolutionStrategies, ESConfig


def run_halfcheetah_experiment(
    max_timesteps: int = 500_000,
    population_size: int = 40,
    sigma: float = 0.02,
    learning_rate: float = 0.01,
    seed: int = 42,
    save_results: bool = True
):
    """
    Run ES experiment on HalfCheetah-v4.
    
    Args:
        max_timesteps: Maximum training timesteps
        population_size: ES population size
        sigma: Noise standard deviation
        learning_rate: ES learning rate
        seed: Random seed
        save_results: Whether to save results to files
    """
    print("=" * 70)
    print("Evolution Strategies on HalfCheetah-v4")
    print("Replication experiment from Salimans et al. (2017)")
    print("=" * 70)
    
    # Configuration based on paper
    # Paper used: sigma=0.02, lr=0.01 for MuJoCo tasks
    config = ESConfig(
        population_size=population_size,
        sigma=sigma,
        learning_rate=learning_rate,
        weight_decay=0.005,
        max_iterations=10000,  # Will stop early based on timesteps
        max_episode_steps=1000,
        normalize_observations=True,
        normalize_rewards=True,
        log_interval=10,
        eval_interval=50,
        seed=seed
    )
    
    # Create ES optimizer
    es = EvolutionStrategies(
        env_name="HalfCheetah-v4",
        config=config,
        policy_hidden_sizes=(64, 64)  # Same as paper
    )
    
    print(f"\nTarget timesteps: {max_timesteps:,}")
    print(f"Population size: {population_size}")
    print(f"Sigma: {sigma}")
    print(f"Learning rate: {learning_rate}")
    
    # Estimate iterations needed
    steps_per_iter = population_size * config.max_episode_steps  # Worst case
    estimated_iters = max_timesteps // (population_size * 500)  # Average episode ~500 steps
    print(f"Estimated iterations: ~{estimated_iters}")
    print()
    
    # Training loop with timestep limit
    history = []
    eval_history = []
    start_time = time.time()
    
    print("Starting training...")
    print("-" * 70)
    
    while es.total_timesteps < max_timesteps:
        stats = es.train_iteration()
        history.append(stats)
        
        # Log progress every iteration for first 10, then every 10
        should_log = (es.iteration <= 10) or (es.iteration % 10 == 0)
        
        if should_log:
            elapsed = time.time() - start_time
            fps = es.total_timesteps / elapsed if elapsed > 0 else 0
            progress = 100 * es.total_timesteps / max_timesteps
            
            print(f"Iter {stats['iteration']:4d} | "
                  f"Steps: {stats['timesteps']:8,} ({progress:5.1f}%) | "
                  f"Reward: {stats['mean_reward']:8.1f} ± {stats['std_reward']:6.1f} | "
                  f"Best: {stats['best_reward']:8.1f} | "
                  f"FPS: {fps:5.0f}")
        
        # Periodic evaluation
        if es.iteration % 50 == 0:
            eval_reward = es.evaluate(num_episodes=5)
            eval_history.append({
                'iteration': es.iteration,
                'timesteps': es.total_timesteps,
                'eval_reward': eval_reward
            })
            print(f"  >>> Evaluation (5 episodes): {eval_reward:.1f}")
    
    # Final evaluation
    print("-" * 70)
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    final_rewards = [es.evaluate(num_episodes=1) for _ in range(10)]
    final_mean = np.mean(final_rewards)
    final_std = np.std(final_rewards)
    
    total_time = time.time() - start_time
    
    print(f"Final reward (10 episodes): {final_mean:.1f} ± {final_std:.1f}")
    print(f"Total timesteps: {es.total_timesteps:,}")
    print(f"Total iterations: {es.iteration}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Average FPS: {es.total_timesteps/total_time:.0f}")
    
    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.dirname(os.path.abspath(__file__)) + '/../results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save training history
        results = {
            'config': {
                'env_name': 'HalfCheetah-v4',
                'population_size': population_size,
                'sigma': sigma,
                'learning_rate': learning_rate,
                'seed': seed
            },
            'final_reward_mean': float(final_mean),
            'final_reward_std': float(final_std),
            'total_timesteps': es.total_timesteps,
            'total_time_seconds': total_time,
            'history': history,
            'eval_history': eval_history
        }
        
        results_file = f"{results_dir}/halfcheetah_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        print(f"\nResults saved to: {results_file}")
        
        # Save policy
        policy_file = f"{results_dir}/halfcheetah_policy_{timestamp}.npz"
        es.save(policy_file)
        
        # Plot learning curve
        plot_file = f"{results_dir}/halfcheetah_curve_{timestamp}.png"
        plot_learning_curve(history, eval_history, plot_file)
        print(f"Learning curve saved to: {plot_file}")
    
    return es, history, eval_history


def plot_learning_curve(history, eval_history, save_path=None):
    """Plot and optionally save the learning curve."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract data
    timesteps = [h['timesteps'] for h in history]
    mean_rewards = [h['mean_reward'] for h in history]
    
    # Smooth rewards with moving average
    window = min(20, max(1, len(mean_rewards) // 5))
    if window > 1 and len(mean_rewards) > window:
        smoothed = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
        smooth_timesteps = timesteps[window-1:]
    else:
        smoothed = mean_rewards
        smooth_timesteps = timesteps
    
    # Plot 1: Reward vs Timesteps
    ax1.plot(timesteps, mean_rewards, alpha=0.3, color='blue', label='Raw')
    ax1.plot(smooth_timesteps, smoothed, color='blue', linewidth=2, label='Smoothed')
    
    # Add evaluation points
    if eval_history:
        eval_ts = [e['timesteps'] for e in eval_history]
        eval_rewards = [e['eval_reward'] for e in eval_history]
        ax1.scatter(eval_ts, eval_rewards, color='red', s=50, zorder=5, label='Evaluation')
    
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('ES on HalfCheetah-v4: Learning Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reward distribution over time
    iterations = [h['iteration'] for h in history]
    max_rewards = [h['max_reward'] for h in history]
    min_rewards = [h['min_reward'] for h in history]
    
    ax2.fill_between(iterations, min_rewards, max_rewards, alpha=0.3, color='blue')
    ax2.plot(iterations, mean_rewards, color='blue', linewidth=1)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Reward')
    ax2.set_title('Reward Distribution (min/mean/max)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def quick_test():
    """Quick test to verify everything works (~5-10 min)."""
    print("=" * 70)
    print("QUICK TEST - Running ES on HalfCheetah for 100k timesteps")
    print("This should take about 5-10 minutes on 4 CPU cores")
    print("=" * 70)
    
    run_halfcheetah_experiment(
        max_timesteps=100_000,
        population_size=40,  # Standard size
        sigma=0.02,
        learning_rate=0.01,
        save_results=True
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ES experiment on HalfCheetah')
    parser.add_argument('--timesteps', type=int, default=500_000,
                        help='Maximum training timesteps')
    parser.add_argument('--population', type=int, default=40,
                        help='Population size')
    parser.add_argument('--sigma', type=float, default=0.02,
                        help='Noise standard deviation')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test (100k timesteps)')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    else:
        run_halfcheetah_experiment(
            max_timesteps=args.timesteps,
            population_size=args.population,
            sigma=args.sigma,
            learning_rate=args.lr,
            seed=args.seed
        )
