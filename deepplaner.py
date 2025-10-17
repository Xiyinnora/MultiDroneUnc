import numpy as np
import time
from multi_drone import MultiDroneUnc
from typing import List, Tuple, Dict
import heapq


class MultiDroneAStarPlanner:
    def __init__(self, env: MultiDroneUnc, heuristic_weight: float = 1.0):
        self.env = env
        self.cfg = env.get_config()
        self.heuristic_weight = heuristic_weight

        # 获取动作向量
        self.action_vectors = self._get_action_vectors()
        self.num_single_actions = len(self.action_vectors)

    def _get_action_vectors(self) -> np.ndarray:
        """获取可用的动作向量"""
        if self.cfg.change_altitude:
            # 26个3D方向
            vectors = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if not (dx == 0 and dy == 0 and dz == 0):
                            vectors.append([dx, dy, dz])
            return np.array(vectors, dtype=np.int32)
        else:
            # 8个2D方向
            vectors = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if not (dx == 0 and dy == 0):
                        vectors.append([dx, dy, 0])
            return np.array(vectors, dtype=np.int32)

    def _manhattan_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> int:
        """计算曼哈顿距离"""
        return np.sum(np.abs(pos1 - pos2))

    def _euclidean_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """计算欧几里得距离"""
        return np.sqrt(np.sum((pos1 - pos2) ** 2))

    def _heuristic(self, state: np.ndarray, goals: np.ndarray) -> float:
        """计算启发式函数值"""
        positions = state[:, :3]
        reached = state[:, 3].astype(bool)

        total_distance = 0
        for i in range(len(positions)):
            if not reached[i]:
                total_distance += self._manhattan_distance(positions[i], goals[i])

        return total_distance * self.heuristic_weight

    def _get_single_drone_actions(self, drone_idx: int, state: np.ndarray) -> List[int]:
        """为单个无人机获取可行的动作"""
        positions = state[:, :3]
        reached = state[:, 3].astype(bool)

        if reached[drone_idx]:
            return [0]  # 已到达目标的无人机不移动

        current_pos = positions[drone_idx]
        valid_actions = []

        for action_idx in range(self.num_single_actions):
            new_pos = current_pos + self.action_vectors[action_idx]

            # 检查边界
            if (new_pos[0] < 0 or new_pos[0] >= self.cfg.grid_size[0] or
                    new_pos[1] < 0 or new_pos[1] >= self.cfg.grid_size[1] or
                    new_pos[2] < 0 or new_pos[2] >= self.cfg.grid_size[2]):
                continue

            # 检查障碍物
            if self.env.obstacles[new_pos[0], new_pos[1], new_pos[2]]:
                continue

            valid_actions.append(action_idx)

        return valid_actions if valid_actions else [0]  # 如果没有有效动作，保持不动

    def _decode_joint_action(self, action_int: int) -> np.ndarray:
        """解码联合动作为单个无人机动作列表"""
        num_drones = len(self.cfg.start_positions)
        actions = np.zeros(num_drones, dtype=np.int32)
        x = action_int

        for i in range(num_drones):
            actions[i] = x % self.num_single_actions
            x //= self.num_single_actions

        return actions

    def _encode_joint_action(self, actions: List[int]) -> int:
        """编码单个无人机动作列表为联合动作"""
        action_int = 0
        base = 1

        for action in actions:
            action_int += action * base
            base *= self.num_single_actions

        return action_int

    def plan(self, current_state: np.ndarray, planning_time_per_step: float) -> int:
        """主要的规划函数"""
        start_time = time.time()
        goals = np.array(self.cfg.goal_positions)

        # 如果所有无人机都已到达目标，返回任意动作
        if np.all(current_state[:, 3] == 1):
            return 0

        # 使用改进的A*搜索
        best_action = self._improved_astar_search(current_state, goals, start_time, planning_time_per_step)

        return best_action

    def _improved_astar_search(self, start_state: np.ndarray, goals: np.ndarray,
                               start_time: float, max_time: float) -> int:
        """改进的A*搜索算法"""
        num_drones = len(self.cfg.start_positions)

        # 为每个无人机生成候选动作
        candidate_actions = []
        for i in range(num_drones):
            candidate_actions.append(self._get_single_drone_actions(i, start_state))

        # 评估所有可能的联合动作
        best_value = -float('inf')
        best_action = 0

        # 限制评估的动作数量以避免组合爆炸
        max_actions_to_evaluate = min(1000, self.env.num_actions)
        actions_evaluated = 0

        # 使用优先级队列来先评估最有希望的联合动作
        action_queue = []

        # 生成初始候选动作
        import itertools
        action_combinations = list(itertools.product(*candidate_actions))

        # 根据启发式值对动作组合排序
        for action_combo in action_combinations:
            if time.time() - start_time > max_time * 0.9:  # 保留10%的时间
                break

            if actions_evaluated >= max_actions_to_evaluate:
                break

            joint_action = self._encode_joint_action(action_combo)

            # 模拟一步
            next_state, reward, done, info = self.env.simulate(start_state, joint_action)

            # 计算价值（奖励 + 启发式值）
            heuristic_val = self._heuristic(next_state, goals)
            value = reward + self.cfg.discount_factor * heuristic_val

            heapq.heappush(action_queue, (-value, joint_action))  # 使用负值因为heapq是最小堆
            actions_evaluated += 1

        # 返回最佳动作
        if action_queue:
            best_value, best_action = heapq.heappop(action_queue)
            best_value = -best_value  # 转换回正值

        return best_action


class GreedyPlanner:
    """备用的贪婪规划器"""

    def __init__(self, env: MultiDroneUnc):
        self.env = env
        self.cfg = env.get_config()

    def plan(self, current_state: np.ndarray, planning_time_per_step: float) -> int:
        positions = current_state[:, :3]
        reached = current_state[:, 3].astype(bool)
        goals = np.array(self.cfg.goal_positions)

        single_actions = []

        for i in range(len(positions)):
            if reached[i]:
                single_actions.append(0)  # 已到达的无人机不移动
                continue

            # 找到朝向目标的最佳方向
            best_action = 0
            best_score = float('inf')
            current_pos = positions[i]
            goal_pos = goals[i]

            for action_idx in range(8 if not self.cfg.change_altitude else 26):
                # 这里简化处理，实际应该使用env的动作向量
                # 这只是示意性的实现
                if action_idx == 0:
                    continue

                # 计算新位置（简化）
                # 实际应该使用env的动作向量系统
                new_pos = current_pos.copy()
                # 这里需要根据action_idx计算新位置

                # 计算到目标的距离
                distance = np.sum(np.abs(new_pos - goal_pos))

                if distance < best_score:
                    best_score = distance
                    best_action = action_idx

            single_actions.append(best_action)

        # 编码为联合动作
        return self._encode_actions(single_actions)

    def _encode_actions(self, actions: List[int]) -> int:
        """编码单个动作为联合动作"""
        num_per_drone = 8 if not self.cfg.change_altitude else 26
        joint_action = 0
        base = 1

        for action in actions:
            joint_action += action * base
            base *= num_per_drone

        return joint_action