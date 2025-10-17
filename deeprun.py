import argparse
import numpy as np
from multi_drone import MultiDroneUnc
import time


class ImprovedAStarPlanner:
    def __init__(self, env: MultiDroneUnc, heuristic_weight: float = 1.0, exploration_factor: float = 0.1):
        self.env = env
        self.cfg = env.get_config()
        self.heuristic_weight = heuristic_weight
        self.exploration_factor = exploration_factor

    def _get_action_vectors(self):
        """获取动作向量"""
        if self.cfg.change_altitude:
            vectors = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if not (dx == 0 and dy == 0 and dz == 0):
                            vectors.append([dx, dy, dz])
            return np.array(vectors)
        else:
            vectors = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if not (dx == 0 and dy == 0):
                        vectors.append([dx, dy, 0])
            return np.array(vectors)

    def _manhattan_distance(self, pos1, pos2):
        return np.sum(np.abs(pos1 - pos2))

    def plan(self, current_state: np.ndarray, planning_time_per_step: float) -> int:
        start_time = time.time()
        goals = np.array(self.cfg.goal_positions)
        num_drones = len(goals)

        # 如果所有无人机都已到达目标
        if np.all(current_state[:, 3] == 1):
            return 0

        action_vectors = self._get_action_vectors()
        num_single_actions = len(action_vectors)

        best_value = -float('inf')
        best_action = 0

        # 评估候选动作
        positions = current_state[:, :3]
        reached = current_state[:, 3].astype(bool)

        # 为每个无人机生成候选动作
        candidate_actions_per_drone = []
        for i in range(num_drones):
            if reached[i]:
                candidate_actions_per_drone.append([0])  # 只考虑不动
            else:
                current_pos = positions[i]
                valid_actions = [0]  # 包括不动

                for action_idx in range(1, num_single_actions):
                    new_pos = current_pos + action_vectors[action_idx]

                    # 检查边界
                    if (0 <= new_pos[0] < self.cfg.grid_size[0] and
                            0 <= new_pos[1] < self.cfg.grid_size[1] and
                            0 <= new_pos[2] < self.cfg.grid_size[2]):

                        # 检查障碍物
                        if not self.env.obstacles[new_pos[0], new_pos[1], new_pos[2]]:
                            valid_actions.append(action_idx)

                candidate_actions_per_drone.append(valid_actions)

        # 限制评估的动作组合数量
        max_evaluations = min(500, int(planning_time_per_step * 100))  # 根据时间调整

        evaluations = 0
        import itertools
        import random

        # 生成动作组合
        all_combinations = list(itertools.product(*candidate_actions_per_drone))
        if len(all_combinations) > max_evaluations:
            # 随机采样
            all_combinations = random.sample(all_combinations, max_evaluations)

        for action_combo in all_combinations:
            if time.time() - start_time > planning_time_per_step * 0.8:
                break

            joint_action = 0
            base = 1
            for action in action_combo:
                joint_action += action * base
                base *= num_single_actions

            # 模拟一步
            next_state, reward, done, info = self.env.simulate(current_state, joint_action)

            # 计算启发式值
            heuristic_val = 0
            next_positions = next_state[:, :3]
            next_reached = next_state[:, 3].astype(bool)

            for i in range(num_drones):
                if not next_reached[i]:
                    heuristic_val -= self._manhattan_distance(next_positions[i], goals[i])

            # 总价值
            total_value = reward + self.heuristic_weight * heuristic_val

            if total_value > best_value:
                best_value = total_value
                best_action = joint_action

            evaluations += 1

        return best_action


def run(env, planner, planning_time_per_step=1.0):
    current_state = env.reset()
    num_steps = 0
    total_discounted_reward = 0.0
    history = []

    while True:
        action = planner.plan(current_state, planning_time_per_step)
        next_state, reward, done, info = env.step(action)

        total_discounted_reward += (env.get_config().discount_factor ** num_steps) * reward
        history.append((current_state, action, reward, next_state, done, info))
        current_state = next_state
        num_steps += 1

        if done or num_steps >= env.get_config().max_num_steps:
            break

    return total_discounted_reward, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to the yaml configuration file")
    parser.add_argument('--planning_time', type=float, default=1.0, help="Planning time per step in seconds")
    args = parser.parse_args()

    env = MultiDroneUnc(args.config)
    planner = ImprovedAStarPlanner(env, heuristic_weight=1.0, exploration_factor=0.1)

    total_discounted_reward, history = run(env, planner, planning_time_per_step=args.planning_time)
    print(f"Success: {history[-1][5]['success']}, Total discounted reward: {total_discounted_reward}")
    env.show()