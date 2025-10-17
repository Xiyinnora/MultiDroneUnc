import time
import numpy as np
import math
from collections import deque
import heapq


class MCTSNode:
    """MCTS树中的一个节点。"""

    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.untried_actions = None
        self.visits = 0
        self.total_reward = 0.0

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    @property
    def q_value(self):
        if self.visits == 0:
            return 0
        return self.total_reward / self.visits


class MCTSPlanner:
    """
    V7.0版本：使用基于“A*路径规划”的全新前瞻性启发式。
    """

    def __init__(self, env, exploration_constant: float = 2.0):
        self._env = env
        self.exploration_constant = exploration_constant
        self.num_drones = self._env.N
        self.single_drone_actions = self._env._action_vectors
        self.grid_size = self._env.get_config().grid_size
        self.goal_positions = self._env.get_config().goal_positions

        print("V7.0 Planner: 正在为每个目标点预计算导航地图 (BFS for A* heuristic)...")
        self.goal_distance_maps = [self._calculate_goal_distance_map(goal) for goal in self.goal_positions]
        print("导航地图计算完成。")

    def _calculate_goal_distance_map(self, goal_pos):
        # ... (This BFS function is now used as the perfect heuristic for A*)
        obstacles = self._env.obstacles
        dist_map = np.full(self.grid_size, float('inf'))
        q = deque([tuple(goal_pos)])
        dist_map[tuple(goal_pos)] = 0
        while q:
            pos = q.popleft()
            for move in self.single_drone_actions:
                next_pos = tuple(np.array(pos) + move)
                if not (0 <= next_pos[0] < self.grid_size[0] and 0 <= next_pos[1] < self.grid_size[1] and 0 <= next_pos[
                    2] < self.grid_size[2]): continue
                if obstacles[next_pos]: continue
                if dist_map[next_pos] == float('inf'):
                    dist_map[next_pos] = dist_map[tuple(pos)] + 1
                    q.append(next_pos)
        return dist_map

    def plan(self, current_state, planning_time_per_step):
        # ... (The main MCTS loop remains the same)
        start_time = time.time()
        root_node = MCTSNode(state=current_state)
        root_node.untried_actions = list(range(self._env.num_actions))
        num_simulations = 0
        while time.time() - start_time < planning_time_per_step:
            node = root_node
            while node.is_fully_expanded() and node.children: node = self._select_child(node)
            if not node.is_fully_expanded():
                action = np.random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                next_state, _, _, _ = self._env.simulate(node.state, action)
                child_node = MCTSNode(state=next_state, parent=node, action=action)
                child_node.untried_actions = list(range(self._env.num_actions))
                node.children.append(child_node)
                node = child_node
            reward = self._a_star_rollout(node.state)
            while node is not None:
                node.visits += 1
                node.total_reward += reward
                node = node.parent
            num_simulations += 1
        print(f"在 {time.time() - start_time:.2f}s 内运行了 {num_simulations} 次模拟。")
        if not root_node.children: return np.random.randint(self._env.num_actions)
        best_child = max(root_node.children, key=lambda c: c.visits)
        return best_child.action

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        # ... (Unchanged)
        log_total_visits = math.log(node.visits)

        def ucb1(child: MCTSNode):
            if child.visits == 0: return float('inf')
            return child.q_value + self.exploration_constant * math.sqrt(log_total_visits / child.visits)

        return max(node.children, key=ucb1)

    def _a_star_rollout(self, state: np.ndarray) -> float:
        """
        ★★ 核心 V7.0 启发式函数 ★★
        使用带冲突解决的优先A*算法来评估状态的好坏。
        """
        paths = []
        max_path_len = 0

        for i in range(self.num_drones):
            # 如果无人机已到达，它的路径就是停在原地
            if state[i, 3] == 1:
                paths.append([tuple(state[i, :3])])
                continue

            # 为当前无人机规划路径，将已规划好的其他无人机路径视为动态障碍物
            path = self._a_star_search(tuple(state[i, :3]), tuple(self.goal_positions[i]), self.goal_distance_maps[i],
                                       paths)

            if path is None:
                # 如果任何一架无人机找不到路径，则这是一个死局，返回极低的奖励
                return -1000

            paths.append(path)
            max_path_len = max(max_path_len, len(path))

        # 根据找到的路径长度给出一个高质量的奖励评估
        # 路径越长，奖励越低。成功找到路径本身就是一个巨大的奖励。
        # 这里的100和-1是估算值，可以调整
        estimated_reward = 100 - 1 * max_path_len
        return estimated_reward

    def _a_star_search(self, start, goal, heuristic_map, other_paths):
        """
        带动态障碍物躲避的A*搜索算法。
        """
        open_set = [(0, start)]  # (f_score, position)
        came_from = {}
        g_score = {start: 0}

        max_time = 100  # A*搜索的最大时间步

        while open_set:
            _, current = heapq.heappop(open_set)
            time_step = g_score[current]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            if time_step > max_time: continue

            for move in self.single_drone_actions:
                neighbor = tuple(np.array(current) + move)

                # 边界和静态障碍物检查
                if not (0 <= neighbor[0] < self.grid_size[0] and 0 <= neighbor[1] < self.grid_size[1] and 0 <= neighbor[
                    2] < self.grid_size[2]): continue
                if self._env.obstacles[neighbor]: continue

                # 动态障碍物检查（与其他无人机的路径碰撞）
                is_collision = False
                next_time_step = time_step + 1
                for path in other_paths:
                    # 检查未来位置是否与另一条路径在同一时间点重合
                    if next_time_step < len(path) and neighbor == path[next_time_step]:
                        is_collision = True
                        break
                    # 检查是否会发生“擦肩而过”的碰撞
                    if next_time_step < len(path) and current == path[next_time_step] and neighbor == path[time_step]:
                        is_collision = True
                        break
                if is_collision: continue

                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic_map[neighbor]
                    heapq.heappush(open_set, (f_score, neighbor))

        return None  # 未找到路径