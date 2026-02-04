"""
Experiment: Replication of ES results on HalfCheetah-v4 (OPTIMIZED VERSION)

Based on: Salimans et al. (2017) "Evolution Strategies as a Scalable Alternative to RL"

Changes from basic version:
- Shorter max episode steps for more iterations
- Optimized hyperparameters based on paper
- Better logging and progress tracking
- Multiple seeds support for stability analysis
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


def run_halfcheetah_optimized(
    max_timesteps: int = 1_000_000,
    population_size: int = 40,
    sigma: float = 0.02,
    learning_rate: float = 0.01,
    max_episode_steps: int = 400,  # Shorter episodes = more iterations
    seed: int = 42,
    save_results: bool = True,
    experiment_name: str = "halfcheetah_optimized"
):
    """
    Run optimized ES experiment on HalfCheetah-v4.
    
    Key insight: Shorter episodes mean more gradient updates per timestep,
    which leads to faster learning (at the cost of not seeing full episodes).
    """
    print("=" * 70)
    print("Evolution Strategies on HalfCheetah-v4 (OPTIMIZED)")
    print("Replication experiment from Salimans et al. (2017)")
    print("=" * 70)
    
    # Optimized configuration
    config = ESConfig(
        population_size=population_size,
        sigma=sigma,
        learning_rate=learning_rate,
        weight_decay=0.005,
        max_iterations=100000,  # High limit, will stop by timesteps
        max_episode_steps=max_episode_steps,  # KEY: shorter episodes
        normalize_observations=True,
        normalize_rewards=True,  # Fitness shaping
        log_interval=10,
        eval_interval=50,
        seed=seed
    )
    
    # Create ES optimizer
    es = EvolutionStrategies(
        env_name="HalfCheetah-v4",
        config=config,
        policy_hidden_sizes=(64, 64)
    )
    
    # Calculate expected iterations
    expected_steps_per_iter = population_size * max_episode_steps
    expected_iters = max_timesteps // expected_steps_per_iter
    
    print(f"\n{'Configuration':=^70}")
    print(f"  Target timesteps:     {max_timesteps:,}")
    print(f"  Population size:      {population_size}")
    print(f"  Max episode steps:    {max_episode_steps}")
    print(f"  Sigma (noise):        {sigma}")
    print(f"  Learning rate:        {learning_rate}")
    print(f"  Expected iterations:  ~{expected_iters}")
    print(f"  Seed:                 {seed}")
    print("=" * 70)
    
    # Training
    history = []
    eval_history = []
    start_time = time.time()
    last_log_time = start_time
    
    print("\nTraining started...")
    print("-" * 70)
    print(f"{'Iter':>5} | {'Steps':>10} | {'Progress':>7} | {'Reward':>10} | {'Std':>8} | {'Best':>10} | {'FPS':>6}")
    print("-" * 70)
    
    while es.total_timesteps < max_timesteps:
        stats = es.train_iteration()
        history.append(stats)
        
        # Adaptive logging: more frequent at start, less frequent later
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
            progress = 100 * es.total_timesteps / max_timesteps
            
            print(f"{stats['iteration']:5d} | "
                  f"{stats['timesteps']:10,} | "
                  f"{progress:6.1f}% | "
                  f"{stats['mean_reward']:10.1f} | "
                  f"{stats['std_reward']:8.1f} | "
                  f"{stats['best_reward']:10.1f} | "
                  f"{fps:6.0f}")
        
        # Evaluation every 50 iterations
        if es.iteration % 50 == 0:
            eval_reward = es.evaluate(num_episodes=10)
            eval_history.append({
                'iteration': es.iteration,
                'timesteps': es.total_timesteps,
                'eval_reward': eval_reward
            })
            print(f"  >>> EVAL (10 episodes): {eval_reward:.1f}")
    
    # Final evaluation
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
    print(f"  Average FPS:      {es.total_timesteps/total_time:.0f}")
    print(f"  Best reward:      {es.best_reward:.1f}")
    print(f"  Final reward:     {final_mean:.1f} ± {final_std:.1f}")
    print("=" * 70)
    
    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.dirname(os.path.abspath(__file__)) + '/../results'
        os.makedirs(results_dir, exist_ok=True)
        
        results = {
            'experiment': experiment_name,
            'config': {
                'env_name': 'HalfCheetah-v4',
                'population_size': population_size,
                'sigma': sigma,
                'learning_rate': learning_rate,
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
        
        # Save JSON
        results_file = f"{results_dir}/{experiment_name}_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, 
                     default=lambda x: float(x) if isinstance(x, np.floating) else x)
        print(f"\nResults: {results_file}")
        
        # Save policy
        policy_file = f"{results_dir}/{experiment_name}_policy_{timestamp}.npz"
        es.save(policy_file)
        
        # Plot
        plot_file = f"{results_dir}/{experiment_name}_curve_{timestamp}.png"
        plot_learning_curve_detailed(history, eval_history, config, results, plot_file)
        print(f"Plot: {plot_file}")
    
    return es, history, eval_history, results


def plot_learning_curve_detailed(history, eval_history, config, results, save_path=None):
    """Create detailed learning curve plot."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    timesteps = [h['timesteps'] for h in history]
    iterations = [h['iteration'] for h in history]
    mean_rewards = [h['mean_reward'] for h in history]
    max_rewards = [h['max_reward'] for h in history]
    min_rewards = [h['min_reward'] for h in history]
    std_rewards = [h['std_reward'] for h in history]
    
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
    ax1.set_title('Learning Curve: Reward vs Timesteps')
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
    
    Environment:        HalfCheetah-v4
    Algorithm:          Evolution Strategies
    
    HYPERPARAMETERS
    ─────────────────────────────────────
    Population size:    {config.population_size}
    Sigma (noise):      {config.sigma}
    Learning rate:      {config.learning_rate}
    Max episode steps:  {config.max_episode_steps}
    
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
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    """Run the optimized HalfCheetah experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Optimized ES experiment on HalfCheetah-v4',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick test (~5 min):
    python run_halfcheetah_optimized.py --quick
    
  Standard experiment (~15-20 min):
    python run_halfcheetah_optimized.py --timesteps 1000000
    
  Long experiment (~1 hour):
    python run_halfcheetah_optimized.py --timesteps 5000000
        """
    )
    
    parser.add_argument('--timesteps', type=int, default=1_000_000,
                        help='Maximum training timesteps (default: 1M)')
    parser.add_argument('--population', type=int, default=40,
                        help='Population size (default: 40)')
    parser.add_argument('--episode-steps', type=int, default=400,
                        help='Max steps per episode (default: 400)')
    parser.add_argument('--sigma', type=float, default=0.02,
                        help='Noise standard deviation (default: 0.02)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with 200k timesteps')
    
    args = parser.parse_args()
    
    if args.quick:
        print("\n" + "="*70)
        print("QUICK TEST MODE - 200k timesteps (~3-5 min)")
        print("="*70 + "\n")
        run_halfcheetah_optimized(
            max_timesteps=200_000,
            population_size=40,
            max_episode_steps=300,
            save_results=True,
            experiment_name="halfcheetah_quick"
        )
    else:
        run_halfcheetah_optimized(
            max_timesteps=args.timesteps,
            population_size=args.population,
            max_episode_steps=args.episode_steps,
            sigma=args.sigma,
            learning_rate=args.lr,
            seed=args.seed,
            save_results=True
        )


if __name__ == "__main__":
    main()