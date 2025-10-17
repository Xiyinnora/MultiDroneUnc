import argparse
import time
from multi_drone import MultiDroneUnc
from mcts_planner import MCTSPlanner

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='example_config.yaml', help="Path to the yaml configuration file")
args = parser.parse_args()


def run(env, planner, planning_time_per_step=10.0, total_mission_time_limit=120.0):
    print(f"开始运行，每步规划时间: {planning_time_per_step}s, 总任务时限: {total_mission_time_limit}s")
    mission_start_time = time.time()
    current_state = env.reset()
    num_steps, total_discounted_reward, history = 0, 0.0, []

    while True:
        elapsed_time = time.time() - mission_start_time
        if elapsed_time > total_mission_time_limit:
            print(f"\n--- 任务超时 ({total_mission_time_limit:.0f}秒)！强制终止。 ---")
            break

        print(f"\n--- 第 {num_steps} 步 (总耗时: {elapsed_time:.1f}s) ---")

        action = planner.plan(current_state, planning_time_per_step)
        next_state, reward, done, info = env.step(action)
        print(f"选择的动作: {action}, 获得的奖励: {reward:.2f}")

        total_discounted_reward += (env.get_config().discount_factor ** num_steps) * reward
        history.append((current_state, action, reward, next_state, done, info))
        current_state, num_steps = next_state, num_steps + 1

        if done or num_steps >= env.get_config().max_num_steps:
            break

    return total_discounted_reward, history


if __name__ == "__main__":
    env = MultiDroneUnc(args.config)
    # In run_UID.py
    planner = MCTSPlanner(env,
                          exploration_constant=2.5,
                          rollout_depth=20,
                          risk_aversion_weight=0.5)  # 使用新的 risk_aversion_weight 参数

    total_reward, history = run(env, planner,
                                planning_time_per_step=3.0,
                                total_mission_time_limit=120.0)

    print("\n--- 任务结束 ---")
    if history:
        success_status = history[-1][5].get('success', False)
        print(f"任务成功: {success_status}")
        print(f"总步数: {len(history)}")
        print(f"最终总折扣奖励: {total_reward:.4f}")
    else:
        print("任务未能开始或中途强制终止。")

    print("按 'q' 键关闭可视化窗口。")
    env.show()