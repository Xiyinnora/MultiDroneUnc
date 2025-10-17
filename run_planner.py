#!/usr/bin/env python3
"""
Run planner for MultiDroneUnc environment.

Now supports default config (example_config.yaml) and MCTSPlanner with
Manhattan-distance greedy rollout.

Usage:
    python run_planner.py
or
    python run_planner.py --config example_config.yaml
"""

import argparse
import time
import yaml
import numpy as np
from multi_drone import MultiDroneUnc
from mcts_planner import MCTSPlanner

def main():
    # -------- argument parser --------
    parser = argparse.ArgumentParser(description="Run MCTS planner for MultiDroneUnc environment")
    parser.add_argument("--config", default="example_config.yaml", help="Path to configuration YAML file")
    parser.add_argument("--time", type=float, default=1.0, help="Planning time per step (seconds)")
    parser.add_argument("--rollout", choices=["random", "greedy"], default="greedy",
                        help="Rollout type for MCTS (random or Manhattan greedy)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    # -------- instantiate environment using the path string --------
    env = MultiDroneUnc(args.config)

    # -------- instantiate MCTS planner --------
    planner = MCTSPlanner(
        env,
        c=1.0,
        rollout_depth=30,
        rng_seed=args.seed,
        rollout_mode=args.rollout
    )

    # -------- run episode --------
    state = env.reset()
    total_discounted_reward = 0.0
    gamma = env.get_config().discount_factor
    step = 0

    print(f"\n[INFO] Starting planning run with config={args.config}, rollout={args.rollout}, "
          f"planning_time={args.time}s per step\n")

    while True:
        # select action via MCTS
        action = planner.plan(state, planning_time_per_step=args.time)

        # apply to environment
        next_state, reward, done, info = env.step(action)

        total_discounted_reward += (gamma ** step) * reward
        state = next_state
        step += 1

        print(f"Step {step:03d} | reward={reward:.2f} | total discounted={total_discounted_reward:.2f}")

        if done:
            print("\n[INFO] Episode finished.")
            break

    print(f"\n[RESULT] Total discounted reward after {step} steps: {total_discounted_reward:.3f}")

if __name__ == "__main__":
    main()
