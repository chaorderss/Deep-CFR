import numpy as np # 用于数值计算，特别是随机抽样和数组操作
import torch # PyTorch 框架

# 从项目中导入基类和工具
from DeepCFR.workers.la.sampling_algorithms._SamplerBase import SamplerBase as _SamplerBase # 采样器基类
from PokerRL.rl import rl_util # RL 相关工具

# 定义 MultiOutcomeSampler 类，继承自 _SamplerBase
class MultiOutcomeSampler(_SamplerBase):
    """
    Multi-Outcome Sampler (MOS) 的实现。

    状态转移逻辑:
    -   每次轮到 "traverser" (当前迭代的探索玩家) 行动时，会探索其多个可能的动作（子树）。
        对于每个探索的动作分支，剩余的牌堆会被重新洗牌，以确保未来的随机性。
    -   当其他任何玩家行动时，根据他们的当前策略采样选择 1 个动作。
    -   当环境行动时 (例如公共牌发放)，根据其自然动态选择 1 个动作。(注意: PokerRL 环境内部处理了这个逻辑)

    数据存储逻辑:
    -   每次非 "traverser" 的玩家行动时，将其在该状态下的动作概率向量存储到该玩家的平均策略缓冲区 (AvrgReservoirBuffer)。(如果提供了 avrg_buffers)
    -   每次 "traverser" 行动时，计算并存储近似的即时遗憾 (approximate immediate regrets) 到其优势缓冲区 (AdvReservoirBuffer)。
    """

    def __init__(self,
                 env_bldr, # 环境构建器
                 adv_buffers, # 优势缓冲区列表 (每个玩家一个)
                 avrg_buffers=None, # 平均策略缓冲区列表 (可选, 每个玩家一个)
                 n_actions_traverser_samples=3, # traverser 行动时采样的动作数量
                 after_x_only_one=None, # 在达到一定深度后，强制只采样一个动作 (用于优化长对局)
                 ):
        """
        初始化 MultiOutcomeSampler。

        Args:
            env_bldr: 环境构建器实例。
            adv_buffers: 优势缓冲区列表。
            avrg_buffers: (可选) 平均策略缓冲区列表。
            n_actions_traverser_samples (int): traverser 行动时采样的动作数。
                                               None: 等价于 External Sampling (ES)，采样所有合法动作。
                                               1:    等价于 Outcome Sampling (OS)，采样一个动作。
                                               >1:   介于 ES 和 OS 之间。
            after_x_only_one (int):            (可选) 指定一个深度阈值。当遍历深度超过此阈值时，
                                               即使 n_actions_traverser_samples > 1，也只采样 1 个动作。
                                               用于在深层节点减少计算量。
        """
        # 调用父类初始化方法，传递必要的参数
        super().__init__(env_bldr=env_bldr, adv_buffers=adv_buffers, avrg_buffers=avrg_buffers)

        # 存储采样参数
        self._n_actions_traverser_samples = n_actions_traverser_samples
        self._depth_after_which_one = after_x_only_one

    def _get_n_a_to_sample(self, trav_depth, n_legal_actions):
        """
        辅助函数：根据当前遍历深度和配置，决定实际要采样的动作数量。
        """
        # 如果设置了深度阈值，并且当前深度超过了阈值，则只采样 1 个动作
        if (self._depth_after_which_one is not None) and (trav_depth > self._depth_after_which_one):
            return 1
        # 如果 n_actions_traverser_samples 设置为 None (ES 模式)，则采样所有合法动作
        if self._n_actions_traverser_samples is None:
            return n_legal_actions
        # 否则，返回配置的采样数和合法动作数中的较小者
        return min(self._n_actions_traverser_samples, n_legal_actions)

    def _traverser_act(self, start_state_dict, traverser, trav_depth, plyrs_range_idxs, iteration_strats, cfr_iter):
        """
        处理当轮到 "traverser" 行动时的逻辑。
        计算并返回该状态对于 traverser 的期望值（Counterfactual Value），并将遗憾数据存入缓冲区。

        返回值计算说明 (来自原注释):
        v~(I) = Sum_a [ strat(I,a) * v~(I|a) ]  (状态 I 的期望值等于所有动作期望值的加权和)
        由于我们在 traverser 节点采样多个动作 (N = n_actions_to_smpl)，我们需要对这些样本的返回值进行平均。
        计算出的返回值 C = Sum_{a sampled} [ strat(I,a) * v~(I|a) ]
        为了得到状态的期望值，需要调整： v~(I) = C * |A(I)| / N
        其中 |A(I)| 是合法动作总数 (n_legal_actions)。
        """
        # 使用 self._env_wrapper (来自基类) 加载起始状态
        self._env_wrapper.load_state_dict(start_state_dict)
        # 获取当前状态下的合法动作列表
        legal_actions_list = self._env_wrapper.env.get_legal_actions()
        n_legal_actions = len(legal_actions_list) # 合法动作的数量
        # 创建合法动作的掩码 (Mask)，用于后续计算
        legal_action_mask = rl_util.get_legal_action_mask_torch(n_actions=self._env_bldr.N_ACTIONS,
                                                                legal_actions_list=legal_actions_list,
                                                                device=self._adv_buffers[traverser].device, # 使用 traverser 缓冲区的设备
                                                                dtype=torch.float32)
        # 获取当前的公共观察信息 (Public Observation)
        current_pub_obs = self._env_wrapper.get_current_obs()

        # 获取 traverser 的范围索引 (Range Index)，用于从缓冲区或策略网络中获取/存储数据
        traverser_range_idx = plyrs_range_idxs[traverser]

        # """""""""""""""""""""""""
        # 采样动作 (Sample actions)
        # """""""""""""""""""""""""
        # 决定在此节点要采样多少个动作
        n_actions_to_smpl = self._get_n_a_to_sample(trav_depth=trav_depth, n_legal_actions=n_legal_actions)
        # 从合法动作中随机选择 n_actions_to_smpl 个动作的索引
        _idxs = np.arange(n_legal_actions)
        np.random.shuffle(_idxs)
        _idxs = _idxs[:n_actions_to_smpl]
        # 获取采样到的动作本身
        actions = [legal_actions_list[i] for i in _idxs]

        # --- 获取 traverser 在当前状态下的策略 ---
        # 使用 IterationStrategy 对象获取 traverser 在当前观察下的动作概率分布
        strat_i = iteration_strats[traverser].get_a_probs(
            pub_obses=[current_pub_obs],            # 当前公共观察
            range_idxs=[traverser_range_idx],       # 范围索引
            legal_actions_lists=[legal_actions_list], # 合法动作列表
            to_np=True                              # 返回 NumPy 数组
        )[0] # 获取第一个 (也是唯一一个) 结果

        # --- 初始化用于计算的值 ---
        cumm_rew = 0.0 # 累积的期望回报 (用于计算状态值)
        # 初始化近似即时遗憾向量 (Approximate Immediate Regret)
        aprx_imm_reg = torch.zeros(size=(self._env_bldr.N_ACTIONS,), # 大小为总动作数
                                   dtype=torch.float32,
                                   device=self._adv_buffers[traverser].device) # 设备与缓冲区一致

        # """""""""""""""""""""""""
        # 遍历采样到的动作，创建后续状态并递归调用
        # """""""""""""""""""""""""
        for _c, a in enumerate(actions): # _c 是计数器， a 是当前处理的采样动作
            strat_i_a = strat_i[a] # 获取采取动作 a 的概率

            # !!! 关键步骤：对于除了第一个采样动作之外的后续动作 !!!
            # 必须重置环境到开始状态，并且 *重新洗牌* 剩余的牌堆！
            # 这是为了确保每个动作分支探索的是不同的、随机的未来可能性（chance outcomes）。
            if _c > 0:
                self._env_wrapper.load_state_dict(start_state_dict)
                self._env_wrapper.env.reshuffle_remaining_deck()

            # 在环境中执行动作 a
            _obs, _rew_for_all, _done, _info = self._env_wrapper.step(a)
            # 获取采取动作 a 后 traverser 获得的即时回报 (Counterfactual Value for traverser after action a)
            _cfv_traverser_a = _rew_for_all[traverser]

            # --- 递归探索子树 ---
            # 如果游戏没有结束
            if not _done:
                # 递归调用 _recursive_traversal 方法探索动作 a 之后的子树
                # 这个方法会处理后续的对手移动、机会节点移动，直到游戏结束或返回到 traverser 移动
                # 返回值是动作 a 之后子树的期望值 (对于 traverser)
                _cfv_traverser_a += self._recursive_traversal(start_state_dict=self._env_wrapper.state_dict(), # 传入动作 a 之后的状态
                                                              traverser=traverser, # 保持 traverser 不变
                                                              trav_depth=trav_depth + 1, # 深度加 1
                                                              plyrs_range_idxs=plyrs_range_idxs, # 范围索引
                                                              iteration_strats=iteration_strats, # 当前策略
                                                              cfr_iter=cfr_iter) # 迭代次数

            # --- 累积回报 ---
            # 将动作 a 的期望值 (_cfv_traverser_a) 按其概率 (strat_i_a) 加权累加到 cumm_rew
            # 注意：这里累加的是采样到的动作的期望值总和，后面需要调整得到状态期望值
            cumm_rew += strat_i_a * _cfv_traverser_a

            # """"""""""""""""""""""""
            # 计算近似即时遗憾 (Compute the approximate immediate regret)
            # """"""""""""""""""""""""
            # 核心思想: R(I, a) = v(I|a) - v(I) = v(I|a) - Sum_k [ strat(I,k) * v(I|k) ]
            # 这里 v(I|a) 就是 _cfv_traverser_a
            # v(I) 通过 cumm_rew 近似（但 cumm_rew 只包含采样到的动作）

            # 步骤 1: 对于所有动作 k != a，它们的遗憾是 v(I|k) - v(I)。
            # 先从所有动作的遗憾中减去 strat_i_a * _cfv_traverser_a。
            # 这相当于在计算 v(I) 的近似值时，暂时包含了动作 a 的贡献。
            aprx_imm_reg -= strat_i_a * _cfv_traverser_a

            # 步骤 2: 对于动作 a 本身，遗憾是 v(I|a) - v(I)。
            # 将动作 a 的实际期望值 _cfv_traverser_a 加到 aprx_imm_reg[a] 上。
            # 这同时抵消了上一步对动作 a 条目的错误扣除，并加上了 v(I|a) 项。
            aprx_imm_reg[a] += _cfv_traverser_a

            # 经过这两步，aprx_imm_reg 中存储了每个动作的近似即时遗憾 R(I, a) = v(I|a) - Sum_{b sampled} [strat(I,b)*v(I|b)]

        # --- 归一化和存储遗憾 ---
        # 将计算出的遗憾乘以合法动作掩码 (非法动作遗憾为0)
        # 然后除以采样的动作数量 (n_actions_to_smpl)，得到平均遗憾
        # (注意：这里是否应该乘以 n_legal_actions / n_actions_to_smpl 来调整?)
        # 原代码注释似乎暗示遗憾计算不需要这个调整，但返回值需要。需要仔细核对 CFR 理论。
        # 假设这里的计算是为了直接输入神经网络，可能不需要完全匹配理论遗憾定义。
        aprx_imm_reg *= legal_action_mask / n_actions_to_smpl

        # 将计算得到的 (观察, 范围索引, 合法动作掩码, 近似遗憾, 迭代次数) 数据点添加到 traverser 的优势缓冲区
        self._adv_buffers[traverser].add(pub_obs=current_pub_obs,
                                         range_idx=traverser_range_idx,
                                         legal_action_mask=legal_action_mask,
                                         adv=aprx_imm_reg, # 存储的是计算出的近似遗憾向量
                                         iteration=cfr_iter + 1, # 存储对应的迭代次数 (通常用 T+1)
                                         )

        # --- 计算并返回状态的期望值 ---
        # 乘以 n_legal_actions / n_actions_to_smpl 来校正因只采样部分动作导致的偏差
        # 得到状态 I 对于 traverser 的期望值（Counterfactual Value）的估计
        return cumm_rew * n_legal_actions / n_actions_to_smpl

    # --- 以下方法未在代码片段中提供，但根据类结构和功能推断其作用 ---
    # def _opponent_act(self, start_state_dict, opponent_id, plyrs_range_idxs, iteration_strats, cfr_iter):
    #     """处理当轮到非 traverser 玩家行动时的逻辑"""
    #     # 1. 加载状态
    #     # 2. 获取当前玩家 (opponent_id) 的策略概率 strat_opp
    #     # 3. (如果提供了 avrg_buffers) 将 (状态, strat_opp) 存入 opponent_id 的 AvrgBuffer
    #     # 4. 根据 strat_opp 采样 *一个* 动作 a_opp
    #     # 5. 执行动作 a_opp
    #     # 6. 如果游戏未结束，递归调用 _recursive_traversal
    #     # 7. 返回子树的期望值
    #     pass

    # def _recursive_traversal(self, start_state_dict, traverser, trav_depth, plyrs_range_idxs, iteration_strats, cfr_iter):
    #     """递归遍历游戏树的核心函数"""
    #     # 1. 加载状态 start_state_dict
    #     # 2. 判断当前轮到谁行动 (current_player)
    #     # 3. 如果 current_player == traverser:
    #     #       调用 self._traverser_act(...) 并返回结果
    #     # 4. 如果 current_player 是其他玩家 (opponent):
    #     #       调用 self._opponent_act(...) 并返回结果
    #     # 5. 如果是环境行动 (chance node):
    #     #       环境内部处理，然后递归调用自身处理下一个状态
    #     # 6. 如果游戏结束:
    #     #       返回最终回报
    #     pass