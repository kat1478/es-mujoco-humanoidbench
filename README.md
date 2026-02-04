# Evolution Strategies for MuJoCo & HumanoidBench

This repository implements Evolution Strategies (ES) following Salimans et al. (2017) and contains experiments reproducing and adapting the method for MuJoCo HalfCheetah and several HumanoidBench tasks.

Author: Katarzyna Kadyszewska

Overview

- **Goal**: replicate ES results on HalfCheetah-v4 and evaluate/optimize ES on HumanoidBench tasks: `h1hand-walk-v0`, `h1hand-reach-v0`, `h1hand-push-v0`.
- **Key techniques**: antithetic sampling, fitness shaping (rank transforms), observation normalization, weight decay.

Quick results summary

- **HalfCheetah-v4**: best reward ~483.6 (partial replication, 1M timesteps budget)
- **h1hand-walk-v0**: final mean reward ~7.7 (requires VBN for better locomotion)
- **h1hand-reach-v0**: high variance, unstable convergence with 25 iterations
- **h1hand-push-v0**: clear improvement trend, final reward improvements observed

Installation

Prerequisites:

- Python 3.10
- Mamba or Conda

Create environment and install dependencies:

```bash
# From repository root
mamba env create -f environment.yml
# or
conda env create -f environment.yml

# Activate the environment
mamba activate es-rl
# or
conda activate es-rl
```

Additional setup

```bash
# Clone and install HumanoidBench (required for humanoid experiments)
git clone https://github.com/carlosferrazza/humanoid-bench.git
cd humanoid-bench && pip install -e . && cd ..

# If you run into NumPy compatibility issues
pip install "numpy<2.0"

# For headless render on WSL2 or servers
export MUJOCO_GL=egl
```

Project structure

```text
es-mujoco-humanoidbench/
├── src/                 # core implementation: policy, ES algorithm, utilities
├── experiments/         # experiment scripts (HalfCheetah, HumanoidBench)
├── results/             # training logs, checkpoints, plots
├── configs/             # experiment configurations
├── environment.yml      # conda/mamba dependencies
└── README.md            # this file
```

Running experiments

Quick test (verify installation):

```bash
python experiments/run_halfcheetah_optimized.py --quick
```

Full runs:

```bash
# HalfCheetah (replication)
python experiments/run_halfcheetah_optimized.py --timesteps 1000000

# All HumanoidBench experiments
python experiments/run_humanoid.py --env all --steps 1000000

# Single HumanoidBench environment
python experiments/run_humanoid.py --env h1hand-walk-v0 --steps 1000000
```

Common command-line flags

- `--timesteps` / `--steps`: total environment timesteps to run
- `--population`: population size (default 40)
- `--sigma`: noise std (default 0.02)
- `--lr`: learning rate (default 0.01)
- `--seed`: random seed (default 42)

Reproducibility notes

- Experiments were run on a workstation with 8 logical threads and 16 GB RAM (WSL2/Ubuntu 24.04). Results vary with CPU count and randomness.
- For HumanoidBench locomotion tasks, Virtual Batch Normalization (VBN) and larger computational budgets significantly improve performance.

Citation

If you use this code or results, please cite:

Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017). Evolution Strategies as a Scalable Alternative to Reinforcement Learning. arXiv:1703.03864

Additional references

- HumanoidBench: https://humanoid-bench.github.io
