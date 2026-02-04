"""
Experiments on HumanoidBench environments.
Implementation of tasks 2-4 from the preliminary documentation.

"""

import sys
import os
import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../src')

# Internal imports
from es_algorithm import EvolutionStrategies, ESConfig
from utils import ObservationNormalizer

# Try importing HumanoidBench
try:
    import humanoid_bench
    HAS_HUMANOID = True
except ImportError:
    HAS_HUMANOID = False
    print("WARNING: humanoid_bench module not found. Please ensure it is installed.")


def run_humanoid_experiment(env_name, timesteps=1_000_000, seed=42, quick=False):
    """
    Run ES algorithm on a specific HumanoidBench environment.
    """
    print("=" * 70)
    print(f"STARTING EXPERIMENT: {env_name}")
    print("=" * 70)

    # Configuration
    # Humanoid tasks are harder, keeping slightly larger network and specific steps
    population_size = 40
    
    if quick:
        timesteps = 50_000
        max_episode_steps = 200
        eval_interval = 20
    else:
        max_episode_steps = 1000  # Standard for humanoid tasks
        eval_interval = 50

    config = ESConfig(
        population_size=population_size,
        sigma=0.02,
        learning_rate=0.01,
        weight_decay=0.005,
        max_iterations=100000,
        max_episode_steps=max_episode_steps,
        normalize_observations=True,
        normalize_rewards=True,
        log_interval=10,
        eval_interval=eval_interval,
        seed=seed
    )
    
    # Attach env_name to config for convenience (Fixes the AttributeError)
    config.env_name = env_name

    # Initialize ES
    try:
        es = EvolutionStrategies(
            env_name=env_name,
            config=config,
            policy_hidden_sizes=(128, 128)  # Larger network for complex humanoid control
        )
    except Exception as e:
        print(f"ERROR creating environment {env_name}: {e}")
        return None

    # Calculate expected iterations
    expected_steps_per_iter = population_size * max_episode_steps
    
    print(f"\n{'Configuration':=^70}")
    print(f"  Target timesteps:     {timesteps:,}")
    print(f"  Population size:      {population_size}")
    print(f"  Max episode steps:    {max_episode_steps}")
    print(f"  Sigma (noise):        {config.sigma}")
    print(f"  Learning rate:        {config.learning_rate}")
    print(f"  Seed:                 {seed}")
    print("=" * 70)

    # Training loop
    history = []
    eval_history = []
    start_time = time.time()
    
    print("\nTraining started...")
    print("-" * 70)
    print(f"{'Iter':>5} | {'Steps':>10} | {'Progress':>7} | {'Reward':>10} | {'Std':>8} | {'Best':>10} | {'FPS':>6}")
    print("-" * 70)
    
    while es.total_timesteps < timesteps:
        stats = es.train_iteration()
        history.append(stats)
        
        # Adaptive logging logic
        if es.iteration <= 20:
            log_every = 5
        elif es.iteration <= 100:
            log_every = 10
        else:
            log_every = 20

        should_log = (es.iteration % log_every == 0) or (es.iteration == 1)
        
        if should_log:
            elapsed = time.time() - start_time
            fps = es.total_timesteps / elapsed if elapsed > 0 else 0
            progress = 100 * es.total_timesteps / timesteps
            
            print(f"{stats['iteration']:5d} | "
                  f"{stats['timesteps']:10,} | "
                  f"{progress:6.1f}% | "
                  f"{stats['mean_reward']:10.1f} | "
                  f"{stats['std_reward']:8.1f} | "
                  f"{stats['best_reward']:10.1f} | "
                  f"{fps:6.0f}")
            
        # Evaluation
        if es.iteration % config.eval_interval == 0:
            # Evaluate on 3 episodes during training to save time
            eval_reward = es.evaluate(num_episodes=3)
            eval_history.append({
                'iteration': es.iteration,
                'timesteps': es.total_timesteps,
                'eval_reward': eval_reward
            })
            print(f"  >>> EVAL (3 episodes): {eval_reward:.1f}")

    # Final Evaluation (more rigorous)
    print("-" * 70)
    total_time = time.time() - start_time
    
    print(f"\n{'FINAL EVALUATION':=^70}")
    
    final_rewards = []
    for i in range(10):
        reward = es.evaluate(num_episodes=1)
        final_rewards.append(reward)
        print(f"  Episode {i+1}: {reward:.1f}")
    
    final_mean = np.mean(final_rewards)
    final_std = np.std(final_rewards)

    print("-" * 70)
    print(f"  MEAN:  {final_mean:.1f} ± {final_std:.1f}")
    print(f"\n{'SUMMARY':=^70}")
    print(f"  Total timesteps:  {es.total_timesteps:,}")
    print(f"  Total iterations: {es.iteration}")
    print(f"  Total time:       {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best reward:      {es.best_reward:.1f}")
    print(f"  Final reward:     {final_mean:.1f}")
    print("=" * 70)

    # Prepare results dictionary
    results_data = {
        'experiment': env_name,
        'config': {
            'env_name': env_name,
            'population_size': population_size,
            'sigma': config.sigma,
            'learning_rate': config.learning_rate,
            'max_episode_steps': max_episode_steps,
            'seed': seed
        },
        'results': {
            'final_reward_mean': float(final_mean),
            'final_reward_std': float(final_std),
            'best_reward': float(es.best_reward),
            'total_timesteps': es.total_timesteps,
            'total_iterations': es.iteration,
            'total_time_seconds': total_time
        },
        'history': history,
        'eval_history': eval_history
    }
    
    save_results(es, results_data, env_name, seed, config)
    return results_data


def save_results(es, results_data, env_name, seed, config):
    """Save results, policy and plots."""
    results_dir = os.path.dirname(os.path.abspath(__file__)) + '/../results'
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save JSON results
    json_path = f"{results_dir}/{env_name}_seed{seed}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2, 
                 default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nResults saved to: {json_path}")
    
    # 2. Save Policy
    policy_path = f"{results_dir}/{env_name}_seed{seed}_policy.npz"
    es.save(policy_path)
    print(f"Policy saved to: {policy_path}")
    
    # 3. Create Detailed Plot
    plot_path = f"{results_dir}/{env_name}_seed{seed}_curve.png"
    plot_learning_curve_detailed(results_data['history'], results_data['eval_history'], 
                                config, results_data, plot_path)
    print(f"Plot saved to: {plot_path}")


def plot_learning_curve_detailed(history, eval_history, config, results, save_path=None):
    """Create detailed learning curve plot (consistent with HalfCheetah)."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    timesteps = [h['timesteps'] for h in history]
    iterations = [h['iteration'] for h in history]
    mean_rewards = [h['mean_reward'] for h in history]
    max_rewards = [h['max_reward'] for h in history]
    min_rewards = [h['min_reward'] for h in history]
    std_rewards = [h['std_reward'] for h in history]
    
    # Retrieve env name safely
    env_name = results['config'].get('env_name', 'Unknown Environment')

    # Plot 1: Reward vs Timesteps
    ax1 = axes[0, 0]
    ax1.plot(timesteps, mean_rewards, 'b-', alpha=0.5, linewidth=1, label='Mean')
    
    # Smoothing
    window = min(20, max(1, len(mean_rewards) // 10))
    if window > 1 and len(mean_rewards) > window:
        smoothed = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
        smooth_ts = timesteps[window-1:]
        ax1.plot(smooth_ts, smoothed, 'b-', linewidth=2, label=f'Smoothed (w={window})')
    
    if eval_history:
        eval_ts = [e['timesteps'] for e in eval_history]
        eval_rewards = [e['eval_reward'] for e in eval_history]
        ax1.scatter(eval_ts, eval_rewards, color='red', s=60, zorder=5, label='Evaluation')
    
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title(f'{env_name}: Reward vs Timesteps')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reward distribution
    ax2 = axes[0, 1]
    ax2.fill_between(iterations, min_rewards, max_rewards, alpha=0.3, color='blue', label='Min-Max')
    ax2.plot(iterations, mean_rewards, 'b-', linewidth=1.5, label='Mean')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Reward')
    ax2.set_title('Reward Distribution per Iteration')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Standard deviation over time
    ax3 = axes[1, 0]
    ax3.plot(iterations, std_rewards, 'g-', linewidth=1.5)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Std Dev')
    ax3.set_title('Reward Variance (exploration indicator)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    EXPERIMENT SUMMARY
    ══════════════════════════════════════
    
    Environment:        {env_name}
    Algorithm:          Evolution Strategies
    
    HYPERPARAMETERS
    ─────────────────────────────────────
    Population size:    {results['config']['population_size']}
    Sigma (noise):      {results['config']['sigma']}
    Learning rate:      {results['config']['learning_rate']}
    Max episode steps:  {results['config']['max_episode_steps']}
    
    RESULTS
    ─────────────────────────────────────
    Total timesteps:    {results['results']['total_timesteps']:,}
    Total iterations:   {results['results']['total_iterations']}
    Training time:      {results['results']['total_time_seconds']:.1f}s
    
    Best reward:        {results['results']['best_reward']:.1f}
    Final reward:       {results['results']['final_reward_mean']:.1f} ± {results['results']['final_reward_std']:.1f}
    
    ══════════════════════════════════════
    Based on: Salimans et al. (2017)
    "Evolution Strategies as a Scalable
     Alternative to Reinforcement Learning"
    """
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ES on HumanoidBench environments')
    parser.add_argument('--env', type=str, default='all', 
                       choices=['all', 'h1hand-walk-v0', 'h1hand-reach-v0', 'h1hand-push-v0'],
                       help='Environment to run or "all"')
    parser.add_argument('--steps', type=int, default=1_000_000, help='Total timesteps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick', action='store_true', help='Run quick test')
    
    args = parser.parse_args()

    if not HAS_HUMANOID:
        print("ERROR: humanoid_bench not installed.")
        sys.exit(1)

    envs_to_run = []
    if args.env == 'all':
        envs_to_run = ['h1hand-walk-v0', 'h1hand-reach-v0', 'h1hand-push-v0']
    else:
        envs_to_run = [args.env]

    for env_name in envs_to_run:
        run_humanoid_experiment(env_name, timesteps=args.steps, seed=args.seed, quick=args.quick)