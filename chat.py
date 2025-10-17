# mcts_planner.py
"""
MCTSPlanner for MultiDroneUnc environment with Manhattan-distance greedy rollout.

Usage:
    from mcts_planner import MCTSPlanner
    planner = MCTSPlanner(env, c=1.0, rollout_depth=30, rng_seed=0, rollout_mode='greedy')
    action = planner.plan(current_state, planning_time_per_step=1.0)
"""

import time
import math
import numpy as np

class MCTSNode:
    def __init__(self, state_hash=None, parent=None, parent_action=None):
        self.parent = parent
        self.parent_action = parent_action
        self.children = {}      # action_int -> MCTSNode
        self.N = 0              # visit count
        self.W = 0.0            # total value
        self.Q = 0.0            # value (we store total W; Q can be W or mean)
        self.state_hash = state_hash
        self.untried_actions = None

class MCTSPlanner:
    def __init__(self, env, c: float = 1.0, rollout_depth: int = 30, rng_seed: int = None, rollout_mode: str = 'greedy'):
        """
        env: MultiDroneUnc instance
        c: exploration constant for UCB
        rollout_depth: max depth for rollouts
        rollout_mode: 'random' or 'greedy' (Manhattan greedy)
        """
        self.env = env
        self.c = c
        self.rollout_depth = rollout_depth
        self.rng = np.random.default_rng(rng_seed)
        assert rollout_mode in ('random', 'greedy')
        self.rollout_mode = rollout_mode

    def _hash_state(self, state: np.ndarray) -> bytes:
        return state.tobytes()

    def _num_actions(self) -> int:
        # env.num_actions may be property or callable
        try:
            return int(self.env.num_actions)
        except Exception:
            return int(self.env.num_actions())

    def plan(self, current_state: np.ndarray, planning_time_per_step: float) -> int:
        """
        Perform MCTS for up to planning_time_per_step seconds, return chosen joint action (int).
        """
        root = MCTSNode(state_hash=self._hash_state(current_state))
        root.untried_actions = list(range(self._num_actions()))
        start_time = time.time()

        # main loop
        while time.time() - start_time < planning_time_per_step:
            node = root
            state = current_state.copy()
            path = [node]

            # 1) Selection
            while node.untried_actions is not None and len(node.untried_actions) == 0 and len(node.children) > 0:
                total_N = max(1, node.N)
                best_score = -float('inf')
                best_a = None
                best_child = None
                for a, child in node.children.items():
                    # UCB1
                    exploit = child.Q / (child.N + 1e-9)
                    explore = self.c * math.sqrt(math.log(total_N + 1) / (child.N + 1e-9))
                    score = exploit + explore
                    if score > best_score:
                        best_score = score
                        best_a = a
                        best_child = child
                if best_child is None:
                    break
                node = best_child
                # advance state via single generative sample
                next_state, _, done, _ = self.env.simulate(state, best_a)
                state = next_state
                path.append(node)
                if done:
                    break

            # 2) Expansion
            if node.untried_actions is None:
                node.untried_actions = list(range(self._num_actions()))
            if node.untried_actions:
                a = int(self.rng.choice(node.untried_actions))
                # remove chosen action from untried (if present)
                try:
                    node.untried_actions.remove(a)
                except ValueError:
                    pass
                child = MCTSNode(state_hash=self._hash_state(state), parent=node, parent_action=a)
                child.untried_actions = list(range(self._num_actions()))
                node.children[a] = child
                # simulate the chosen action once to get next_state
                next_state, reward, done, _ = self.env.simulate(state, a)
                state = next_state
                node = child
                path.append(node)
            # else: fully expanded or terminal

            # 3) Simulation / Rollout
            total_reward = 0.0
            discount = 1.0
            gamma = self.env.get_config().discount_factor
            rollout_state = state
            done_flag = False

            if self.rollout_mode == 'random':
                for _ in range(self.rollout_depth):
                    if done_flag:
                        break
                    a_rand = int(self.rng.integers(0, self._num_actions()))
                    rollout_state, r, done_flag, _ = self.env.simulate(rollout_state, a_rand)
                    total_reward += discount * r
                    discount *= gamma
            else:
                # greedy (Manhattan-distance per-drone) rollout
                # NOTE: uses env internals: _action_vectors, _encode_action, goals, obstacles
                def greedy_joint_action_from_state(s):
                    # s: (N,4) state array
                    try:
                        action_vectors = self.env._action_vectors
                        encode_fn = self.env._encode_action
                        goals = self.env.goals
                    except Exception:
                        # fallback to uniform random if internals inaccessible
                        return int(self.rng.integers(0, self._num_actions()))

                    N = s.shape[0]
                    per_actions = np.zeros(N, dtype=np.int32)
                    num_per = action_vectors.shape[0]
                    positions = s[:, :3].astype(np.int32)
                    reached = s[:, 3].astype(bool)
                    X, Y, Z = self.env.get_config().grid_size

                    for i in range(N):
                        if reached[i]:
                            per_actions[i] = 0
                            continue
                        best_a = None
                        best_dist = None
                        for a_idx in range(num_per):
                            vec = action_vectors[a_idx]
                            cand = positions[i] + vec
                            # bounds check
                            if not (0 <= cand[0] < X and 0 <= cand[1] < Y and 0 <= cand[2] < Z):
                                continue
                            # avoid stepping into static obstacle if available
                            if getattr(self.env, 'obstacles', None) is not None:
                                if self.env.obstacles[cand[0], cand[1], cand[2]]:
                                    continue
                            goal = goals[i]
                            dist = int(abs(cand[0] - goal[0]) + abs(cand[1] - goal[1]) + abs(cand[2] - goal[2]))
                            if (best_dist is None) or (dist < best_dist):
                                best_dist = dist
                                best_a = a_idx
                        if best_a is None:
                            # no valid moves found (surrounded) -> pick 0 as fallback
                            per_actions[i] = 0
                        else:
                            per_actions[i] = best_a
                    try:
                        joint = encode_fn(per_actions)
                    except Exception:
                        base = 1
                        joint = 0
                        for a in per_actions:
                            joint += int(a) * base
                            base *= num_per
                    return int(joint)

                for _ in range(self.rollout_depth):
                    if done_flag:
                        break
                    a_greedy = greedy_joint_action_from_state(rollout_state)
                    rollout_state, r, done_flag, _ = self.env.simulate(rollout_state, a_greedy)
                    total_reward += discount * r
                    discount *= gamma

            # 4) Backpropagation: add total_reward to nodes on path
            for n in reversed(path):
                n.N += 1
                n.W += total_reward
                n.Q = n.W

        # choose best action from root: highest visit count
        best_action = None
        best_visits = -1
        for a, child in root.children.items():
            if child.N > best_visits:
                best_visits = child.N
                best_action = a
        if best_action is None:
            # fallback: uniform random
            best_action = int(self.rng.integers(0, self._num_actions()))

        return int(best_action)
