# 版权所有 (c) 2019 Eric Steinberger (修改以适应无限注和 DiscretizedPokerEnv)

import copy
import numpy as np

from PokerRL.game.Poker import Poker
from PokerRL.game._.wrappers._Wrapper import Wrapper

# NOLimitPokerHistoryWrapper 类:
# 适用于与 DiscretizedPokerEnv (或类似提供整数动作接口的) 无限注环境配合使用。
# 将游戏历史编码为扁平向量并附加到观察中。
# 适用于前馈神经网络。
class NOLimitPokerHistoryWrapper(Wrapper):
    """
    这个包装器支持使用整数动作接口的无限注扑克游戏 (如 DiscretizedPokerEnv)。
    适用于前馈神经网络架构。
    存储动作序列，并将其附加到当前观察值之后。
    动作历史位于向量中原始环境观察状态 *之后*。
    依赖于 EnvBuilder 正确处理离散化的无限注动作索引。
    """

    def __init__(self, env, env_bldr_that_built_me):
        """
        初始化包装器。
        Args:
            env: 被包装的原始扑克环境实例 (应为 DiscretizedPokerEnv 或类似接口)。
            env_bldr_that_built_me: 创建环境和包装器的 EnvBuilder 实例。
        """
        assert env.N_SEATS >= 2, "NOLimitPokerHistoryWrapper 支持2人及以上游戏"
        # 检查 env 是否有类似 DiscretizedPokerEnv 的整数动作接口（一个简单的检查）
        # 可以根据需要添加更严格的检查，例如检查是否存在 _get_env_adjusted_action_formulation
        assert hasattr(env, 'N_ACTIONS'), "被包装的环境应提供 N_ACTIONS 属性 (类似 DiscretizedPokerEnv)"
        super().__init__(env=env, env_bldr_that_built_me=env_bldr_that_built_me)

        self._action_vector_size = env_bldr_that_built_me.action_vector_size
        self._action_count_this_round = None
        self._game_round_last_tick = None
        self._action_history_vector = None
        self._last_action_int = None # << 新增：用于存储上一步的整数动作

    def _reset_state(self, **kwargs):
        """重置包装器的内部状态。"""
        self._action_count_this_round = [0] * self.env.N_SEATS
        self._game_round_last_tick = Poker.PREFLOP
        self._action_history_vector = np.zeros(shape=self._action_vector_size, dtype=np.float32)
        self._last_action_int = None # << 新增：重置时清除

    # << 新增：重写 reset 方法 >>
    def reset(self, deck_state_dict=None, **kwargs):
        """
        重置环境和包装器状态。
        Args:
            deck_state_dict: 可选，用于同步牌面。
        Returns:
            np.ndarray: 包装后的初始观察。
        """
        # 1. 重置基础环境，获取初始观察
        #    kwargs 可能包含来自上层调用者的参数，传递给基础环境
        initial_env_obs, initial_info = self.env.reset(deck_state_dict=deck_state_dict, **kwargs)
        # 2. 重置包装器的内部状态
        self._reset_state()
        # 3. 获取包装后的初始观察 (此时历史向量为空)
        wrapped_obs = self.get_current_obs(env_obs=initial_env_obs)
        # 4. 返回包装后的观察 (通常 reset 不返回 info，但如果需要可以调整)
        return wrapped_obs # , initial_info

    # << 新增：重写 step 方法 >>
    def step(self, action):
        """
        执行一步动作，更新历史，并返回包装后的结果。
        Args:
            action (int): 由代理选择的离散整数动作 (action_int)。
        Returns:
            tuple: (wrapped_obs, reward, done, info)
        """
        # 1. 存储传入的整数动作
        self._last_action_int = action

        # 2. 让基础环境执行动作
        #    基础环境 (如 DiscretizedPokerEnv) 会处理整数到 (type, size) 的转换
        obs, rew, done, info = self.env.step(action)

        # 3. 更新动作历史向量
        #    _pushback 现在会使用 self._last_action_int
        #    只有在游戏未结束时才 pushback (通常是这样，但可以加检查)
        if not done:
             self._pushback()
        # done 之后 _pushback 通常无意义，因为历史不再重要

        # 4. 获取包装后的观察 (基础观察 + 更新后的历史向量)
        wrapped_obs = self.get_current_obs(env_obs=obs)

        # 5. 返回包装后的结果
        return wrapped_obs, rew, done, info

    def _pushback(self, env_obs=None):
        """
        在每个环境步骤之后更新动作历史向量。
        使用 self._last_action_int 作为离散动作索引。
        (env_obs 参数在此处未使用)
        """
        # 检查上一步是否有有效的整数动作被记录
        if self._last_action_int is not None:
            # 仍然需要从 env 获取执行动作的玩家 ID 和发生的轮次
            # 因为 _pushback 可能在 step 之外被调用（例如 set_to_public_tree_node_state）
            # 但在标准的 step 调用流程中，这些信息应该与 _last_action_int 对应
            if self.env.last_action is None or self.env.last_action[2] is None:
                 # 如果 env.last_action 无效（可能发生在 reset 后或特殊情况）
                 # 则无法确定玩家 ID，跳过更新
                 print("警告: _pushback 时无法获取有效的 last_action 信息，跳过历史更新。")
                 return

            _last_actor_id = self.env.last_action[2] # 玩家 ID
            # 轮次使用 _game_round_last_tick，它会在轮次转换时更新
            _round = self._game_round_last_tick

            # 获取该玩家在本轮的动作序号
            # 确保 _action_count_this_round 已初始化且索引有效
            if self._action_count_this_round is None or not (0 <= _last_actor_id < len(self._action_count_this_round)):
                 print(f"警告: _pushback 时动作计数列表无效或玩家 ID {_last_actor_id} 越界，跳过历史更新。")
                 return
            nth_action = self._action_count_this_round[_last_actor_id]

            # 使用环境构建器的方法计算索引。
            # **关键**: get_vector_idx 使用的是 self._last_action_int
            try:
                 idx = self.env_bldr.get_vector_idx(round_=_round,
                                                    p_id=_last_actor_id,
                                                    nth_action_this_round=nth_action,
                                                    action_idx=self._last_action_int) # << 修改点：使用存储的整数动作
            except IndexError:
                 # Builder 的 get_vector_idx 可能会因为 nth_action 超出限制而抛出 IndexError
                 print(f"警告: get_vector_idx 抛出 IndexError。可能动作次数超出预设限制。"
                       f" 轮次={_round}, 玩家={_last_actor_id},"
                       f" 第{nth_action}次动作, 动作={self._last_action_int}")
                 # 可以选择在这里停止，或者让后续的索引检查处理
                 return
            except KeyError:
                 # 如果 round_ 无效
                 print(f"警告: get_vector_idx 抛出 KeyError。无效的轮次: {_round}")
                 return
            except ValueError:
                 # 如果 action_idx (即 _last_action_int) 无效
                  print(f"警告: get_vector_idx 抛出 ValueError。无效的动作索引: {self._last_action_int}")
                  return


            # 在历史向量中标记该动作
            if idx < self._action_vector_size:
                 self._action_history_vector[idx] = 1.0
            else:
                 # 索引超出范围的警告
                 print(f"警告: 动作历史索引 {idx} 超出向量大小 {self._action_vector_size}。"
                       f" 轮次={_round}, 玩家={_last_actor_id},"
                       f" 第{nth_action}次动作, 动作={self._last_action_int}")

            # 增加该玩家在本轮的动作计数
            self._action_count_this_round[_last_actor_id] += 1

            # 检查轮次是否变更 (与之前逻辑相同)
            if self.env.current_round != self._game_round_last_tick:
                # 轮次变更时，确保 current_round 不是 None
                if self.env.current_round is not None:
                    self._game_round_last_tick = self.env.current_round
                    # 重置所有玩家在新轮次的动作计数
                    self._action_count_this_round = [0] * self.env.N_SEATS
                else:
                    print("警告: 环境轮次变为 None，历史轮次状态未更新。")

        # 在下一次 step 调用前清除 _last_action_int，以防被错误重用？
        # 或者保持不变，直到下一次 step 覆盖它？保持不变似乎更合理。
        # self._last_action_int = None


    def print_obs(self, wrapped_obs=None):
        """打印包装后的观察向量，包括基础观察和附加的动作历史。"""
        if wrapped_obs is None:
            wrapped_obs = self.get_current_obs()

        assert isinstance(wrapped_obs, np.ndarray), "观察值必须是 NumPy 数组"
        print()
        print()
        print("*****************************************************************************************************")
        print()
        print("________________________________________ 无限注包装后的观察 + 历史 ________________________________________")
        print()
        # 基础观察的大小
        try:
            base_obs_size = self.env_bldr.pub_obs_size - self._action_vector_size
            if base_obs_size < 0: base_obs_size = 0 # 防止负数切片
            print("--- 基础环境观察 ---")
            # 确保切片不会超出 wrapped_obs 的边界
            self.env.print_obs(wrapped_obs[:min(base_obs_size, len(wrapped_obs))])
            print()
            print("--- 扁平化动作序列 (历史向量) ---")
            # 确保切片不会超出 wrapped_obs 的边界
            print(wrapped_obs[min(base_obs_size, len(wrapped_obs)):])
        except AttributeError:
            print("错误：无法访问 env_bldr.pub_obs_size，基础观察部分无法打印。")
            print(wrapped_obs) # 打印整个包装后的向量
        print("*****************************************************************************************************")


    def get_current_obs(self, env_obs=None):
        """
        获取包装后的当前观察向量。
        Args:
            env_obs (np.ndarray, optional): 基础环境的观察向量。
        Returns:
            np.ndarray: 拼接了动作历史向量的基础观察向量。
        """
        if env_obs is None:
            try:
                base_obs = self.env.get_current_obs(is_terminal=False)
            except TypeError: # 如果 get_current_obs 不接受 is_terminal
                 base_obs = self.env.get_current_obs()
        else:
            base_obs = env_obs

        # 确保历史向量已初始化
        if self._action_history_vector is None:
             self._action_history_vector = np.zeros(shape=self._action_vector_size, dtype=np.float32)

        # 拼接基础观察和历史向量
        return np.concatenate((base_obs.astype(np.float32), self._action_history_vector.astype(np.float32)), axis=0)

    def state_dict(self):
        """返回包装器状态的字典，用于保存。"""
        # 确保历史向量存在
        hist_vector_copy = np.copy(self._action_history_vector) if self._action_history_vector is not None else None
        return {
            "base": super().state_dict(),
            "a_seq": hist_vector_copy, # 保存历史向量副本
            "game_round_last_tick": self._game_round_last_tick,
            # 保存多人动作计数 (深拷贝)
            "action_count_this_round": copy.deepcopy(self._action_count_this_round),
            "_last_action_int": self._last_action_int, # << 新增：保存上一步整数动作
        }

    def load_state_dict(self, state_dict):
        """从字典加载包装器的状态。"""
        super().load_state_dict(state_dict=state_dict["base"])
        # 加载历史向量副本
        a_seq_data = state_dict.get("a_seq")
        self._action_history_vector = np.copy(a_seq_data) if a_seq_data is not None else None
        # 加载轮次和动作计数
        self._game_round_last_tick = state_dict.get("game_round_last_tick")
        self._action_count_this_round = copy.copy(state_dict.get("action_count_this_round"))
        self._last_action_int = state_dict.get("_last_action_int") # << 新增：加载上一步整数动作

    def set_to_public_tree_node_state(self, node):
        """
        将内部环境包装器设置为公共树中某个节点所代表的状态。
        通过回放从根到该节点的动作序列来重建历史向量。
        Args:
            node: 公共树 (PublicTree) 实例中的任何节点。
        """
        state_seq = []  # 存储状态字典序列 (根 -> node)

        # 递归函数：从 node 向上回溯到根，收集玩家动作节点的状态
        def add(_node):
            if _node is not None:
                is_player_node = True
                if hasattr(_node, 'tree') and hasattr(_node.tree, 'CHANCE_ID'):
                    if _node.p_id_acting_next == _node.tree.CHANCE_ID:
                        is_player_node = False
                if is_player_node and hasattr(_node, 'env_state') and _node.env_state is not None:
                     # *** 关键修改：需要存储与该状态关联的 *离散整数动作* ***
                     # 这个整数动作通常存储在公共树节点的 transition_action 或类似属性中
                     # 我们假设它存储在 _node.action_int
                     if hasattr(_node, 'action'): # 假设离散动作存在于 node.action
                         state_seq.insert(0, {'state_dict': _node.env_state, 'action_int': _node.action})
                     else:
                         # 如果节点没有存储 action_int，我们无法完美重建历史
                         # 可以尝试从 state_dict 推断，但这很困难且不可靠
                         # 这里我们选择跳过没有 action_int 的节点，或记录警告
                         print(f"警告: 树节点缺少 'action' 属性，无法在历史中记录此步骤。")
                         # 或者可以只插入 state_dict，但 _pushback_from_state 会失败
                         # state_seq.insert(0, {'state_dict': _node.env_state, 'action_int': None})

                if hasattr(_node, 'parent'):
                    add(_node.parent)

        add(node)

        # 重置环境和包装器状态
        self.reset()

        # 回放状态序列以构建历史向量
        for state_info in state_seq:
            sd = state_info['state_dict']
            action_int = state_info['action_int']
            try:
                # 加载公共状态
                self.env.load_state_dict(sd, blank_private_info=True)
                # ** 修改 _pushback 调用方式，传递 action_int **
                # 我们需要一个修改版的 _pushback 或一个新方法来处理这种情况
                # 因为标准的 _pushback 依赖 self._last_action_int，而 self._last_action_int 是在 step 中设置的
                # 我们创建一个新方法 _pushback_from_state 来处理
                self._pushback_from_state(action_int)

            except Exception as e:
                print(f"在 set_to_public_tree_node_state 中加载或处理状态时出错: {e}")
                continue

    # << 新增：用于 set_to_public_tree_node_state 的 pushback 变体 >>
    def _pushback_from_state(self, action_int):
         """
         根据给定的离散动作整数和当前环境状态更新历史向量。
         主要用于 set_to_public_tree_node_state 回放历史。
         """
         if action_int is not None:
            if self.env.last_action is None or self.env.last_action[2] is None:
                 print("警告: _pushback_from_state 时无法获取有效的 last_action 信息，跳过历史更新。")
                 return
            _last_actor_id = self.env.last_action[2]
            _round = self._game_round_last_tick

            if self._action_count_this_round is None or not (0 <= _last_actor_id < len(self._action_count_this_round)):
                 print(f"警告: _pushback_from_state 时动作计数列表无效或玩家 ID {_last_actor_id} 越界，跳过历史更新。")
                 return
            nth_action = self._action_count_this_round[_last_actor_id]

            try:
                 idx = self.env_bldr.get_vector_idx(round_=_round,
                                                    p_id=_last_actor_id,
                                                    nth_action_this_round=nth_action,
                                                    action_idx=action_int) # 使用传入的 action_int
            except (IndexError, KeyError, ValueError) as e:
                 print(f"警告: _pushback_from_state 调用 get_vector_idx 时出错: {e}."
                       f" 轮次={_round}, 玩家={_last_actor_id},"
                       f" 第{nth_action}次动作, 动作={action_int}")
                 return

            if idx < self._action_vector_size:
                 self._action_history_vector[idx] = 1.0
            else:
                 print(f"警告: 动作历史索引 {idx} 超出向量大小 {self._action_vector_size}。"
                       f" 轮次={_round}, 玩家={_last_actor_id},"
                       f" 第{nth_action}次动作, 动作={action_int}")

            self._action_count_this_round[_last_actor_id] += 1

            if self.env.current_round != self._game_round_last_tick:
                 if self.env.current_round is not None:
                     self._game_round_last_tick = self.env.current_round
                     self._action_count_this_round = [0] * self.env.N_SEATS
                 else:
                     print("警告: 环境轮次变为 None，历史轮次状态未更新。")

         # 这个方法不应该修改 self._last_action_int