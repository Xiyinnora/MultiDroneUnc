import argparse
from multi_drone import MultiDroneUnc
# Import the V7.0 MCTSPlanner
from mcts_planner import MCTSPlanner

# --- Argument Parser ---
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='example_config.yaml', help="Path to the yaml configuration file")
args = parser.parse_args()


def run(env, planner, planning_time_per_step=10.0):  # Increased default time for the powerful A*
    """Main execution loop"""
    print(f"Starting run with planning time per step: {planning_time_per_step}s")

    current_state = env.reset()
    num_steps = 0
    total_discounted_reward = 0.0
    history = []

    while True:
        print(f"\n--- Step {num_steps} ---")
        print(f"Current state (positions): \n{current_state[:, :3]}")

        # Use the MCTS planner to decide the next action
        action = planner.plan(current_state, planning_time_per_step)

        # Execute the action in the environment
        next_state, reward, done, info = env.step(action)
        print(f"Action taken: {action}, Reward received: {reward:.2f}")
        if info.get('num_collisions') or info.get('num_vehicle_collisions'):
            print("COLLISION DETECTED!")

        # Accumulate discounted reward
        total_discounted_reward += (env.get_config().discount_factor ** num_steps) * reward

        # Log history
        history.append((current_state, action, reward, next_state, done, info))

        # Transition to the next state
        current_state = next_state
        num_steps += 1

        if done or num_steps >= env.get_config().max_num_steps:
            break

    return total_discounted_reward, history


# --- Main Program Entry ---
if __name__ == "__main__":
    # 1. Instantiate the environment
    env = MultiDroneUnc(args.config)

    # 2. Instantiate your V7.0 MCTS planner (without rollout_depth)
    planner = MCTSPlanner(env, exploration_constant=2.0)

    # 3. Run the main loop
    # NOTE: The A* heuristic is powerful but computationally intensive.
    # Give it a reasonable amount of time per step.
    total_reward, history = run(env, planner, planning_time_per_step=10.0)

    # 4. Print final results
    print("\n--- MISSION COMPLETE ---")
    if history:
        final_info = history[-1][5]
        success_status = final_info.get('success', False)
        print(f"Success: {success_status}")
        print(f"Total steps: {len(history)}")
        print(f"Final Total Discounted Reward: {total_reward:.4f}")
    else:
        print("Mission could not start.")

    # Show the final trajectory visualization
    print("Press 'q' to close the visualization window.")
    env.show()