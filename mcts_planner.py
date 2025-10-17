import time
import numpy as np
import math
from collections import deque


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state, self.parent, self.action, self.children, self.untried_actions, self.visits, self.total_reward = state, parent, action, [], None, 0, 0.0

    def is_fully_expanded(self): return len(self.untried_actions) == 0 if self.untried_actions is not None else True

    @property
    def q_value(self): return self.total_reward / self.visits if self.visits > 0 else 0


class MCTSPlanner:
    """
    V14.0 最终版：使用“风险感知”启发式，将风险评估内置于Rollout中。
    """

    def __init__(self, env, exploration_constant: float = 2.5, rollout_depth: int = 20,
                 risk_aversion_weight: float = 0.5):
        self._env = env
        self.exploration_constant = exploration_constant
        self.rollout_depth = rollout_depth
        self.risk_aversion = risk_aversion_weight

        self.num_drones = self._env.N
        self.single_drone_actions = self._env._action_vectors
        self.grid_size = self._env.get_config().grid_size
        self.goal_positions = self._env.get_config().goal_positions

        print("V14.0 MCTS Planner: 正在预计算导航和风险地图...")
        self.distance_field = self._env.dist_field  # 直接从环境中获取静态风险地图
        self.goal_distance_maps = [self._calculate_goal_distance_map(g) for g in self.goal_positions]
        print("地图计算完成。")

    def _calculate_goal_distance_map(self, goal_pos):
        obstacles, dist_map = self._env.obstacles, np.full(self.grid_size, float('inf'))
        q = deque([tuple(goal_pos)]);
        dist_map[tuple(goal_pos)] = 0
        while q:
            pos = q.popleft()
            for move in self.single_drone_actions:
                next_pos = tuple(np.array(pos) + move)
                if not (0 <= next_pos[0] < self.grid_size[0] and 0 <= next_pos[1] < self.grid_size[1] and 0 <= next_pos[
                    2] < self.grid_size[2]): continue
                if obstacles[next_pos]: continue
                if dist_map[next_pos] == float('inf'):
                    dist_map[next_pos] = dist_map[tuple(pos)] + 1;
                    q.append(next_pos)
        return dist_map

    def plan(self, current_state, planning_time_per_step):
        start_time, root_node = time.time(), MCTSNode(state=current_state)
        root_node.untried_actions = list(range(self._env.num_actions))
        while time.time() - start_time < planning_time_per_step:
            node = root_node
            while node.is_fully_expanded() and node.children: node = self._select_child(node)
            if not node.is_fully_expanded():
                action = np.random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                next_state, _, _, _ = self._env.simulate(node.state, action)
                child_node = MCTSNode(state=next_state, parent=node, action=action)
                child_node.untried_actions = list(range(self._env.num_actions));
                node.children.append(child_node);
                node = child_node
            reward = self._fast_heuristic_rollout(node.state)
            while node is not None:
                node.visits += 1;
                node.total_reward += reward;
                node = node.parent
        if not root_node.children: return np.random.randint(self._env.num_actions)
        return max(root_node.children, key=lambda c: c.visits).action

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        log_total_visits = math.log(node.visits)

        def ucb1(child: MCTSNode):
            if child.visits == 0: return float('inf')
            return child.q_value + self.exploration_constant * math.sqrt(log_total_visits / child.visits)

        return max(node.children, key=ucb1)

    def _get_risk_aware_greedy_action(self, state: np.ndarray) -> int:
        current_positions, active_drones_mask = state[:, :3], state[:, 3] == 0
        best_individual_actions = np.zeros(self.num_drones, dtype=np.int32)
        intended_next_positions = [None] * self.num_drones

        for i in range(self.num_drones):
            if not active_drones_mask[i]:
                best_individual_actions[i], intended_next_positions[i] = 0, current_positions[i]
                continue

            min_score, best_move_idx = float('inf'), 0

            for move_idx, move_vec in enumerate(self.single_drone_actions):
                intended_pos_np = current_positions[i] + move_vec

                if not (0 <= intended_pos_np[0] < self.grid_size[0] and 0 <= intended_pos_np[1] < self.grid_size[
                    1] and 0 <= intended_pos_np[2] < self.grid_size[2]):
                    score = float('inf')
                else:
                    intended_pos_tuple = tuple(intended_pos_np)
                    bfs_dist = self.goal_distance_maps[i][intended_pos_tuple]

                    # 1. 静态风险：离墙越近，风险越高
                    static_risk = 0
                    dist_to_obstacle = self.distance_field[intended_pos_tuple]
                    if dist_to_obstacle < 1e-6:
                        static_risk = float('inf')
                    else:
                        static_risk = self.risk_aversion / dist_to_obstacle

                    # 2. 动态风险：离队友越近，风险越高
                    dynamic_risk = 0
                    for j in range(i):
                        dist_to_teammate = np.linalg.norm(intended_pos_np - intended_next_positions[j])
                        if dist_to_teammate < 1.0: dynamic_risk = float('inf'); break
                        dynamic_risk += self.risk_aversion / (dist_to_teammate ** 2)

                    if static_risk == float('inf') or dynamic_risk == float('inf'):
                        score = float('inf')
                    else:
                        score = bfs_dist + static_risk + dynamic_risk

                if score < min_score:
                    min_score, best_move_idx = score, move_idx

            best_individual_actions[i] = best_move_idx
            intended_next_positions[i] = current_positions[i] + self.single_drone_actions[best_move_idx]

        return self._env._encode_action(best_individual_actions)

    def _fast_heuristic_rollout(self, state: np.ndarray) -> float:
        total_reward, current_state, discount = 0.0, state, 1.0
        for _ in range(self.rollout_depth):
            action = self._get_risk_aware_greedy_action(current_state)
            next_state, reward, done, _ = self._env.simulate(current_state, action)
            total_reward += discount * reward
            discount *= self._env.get_config().discount_factor
            current_state = next_state
            if done: break
        return total_reward