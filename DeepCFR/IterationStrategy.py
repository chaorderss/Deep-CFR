# 版权信息
# Copyright (c) 2019 Eric Steinberger

import numpy as np # 用于数值计算
import torch # PyTorch 框架
from torch.nn import functional as F # PyTorch 神经网络函数库 (如 relu)

from PokerRL.rl import rl_util # RL 相关工具
from PokerRL.rl.neural.DuelingQNet import DuelingQNet # 导入我们之前分析过的 DuelingQNet 网络结构

# 定义 IterationStrategy 类
class IterationStrategy:
    """
    封装了一个特定 CFR 迭代中的玩家策略。
    它使用一个优势网络 (_adv_net) 来计算给定状态下的动作概率。
    这个类主要用于在数据生成阶段 (self-play) 指导玩家行动。
    """

    def __init__(self, t_prof, owner, env_bldr, device, cfr_iter):
        """
        初始化 IterationStrategy。

        Args:
            t_prof: 训练配置 (Training Profile) 对象。
            owner (int): 拥有此策略的玩家的座位号。
            env_bldr: 环境构建器。
            device: 策略计算所使用的设备 (CPU/GPU)。
            cfr_iter (int): 当前 CFR 迭代次数。
        """
        self._t_prof = t_prof
        self._owner = owner
        self._env_bldr = env_bldr
        self._device = device
        self._cfr_iter = cfr_iter # 记录当前的迭代次数

        # 优势网络 (_adv_net) 初始化为 None。
        # 在第一次使用前 (迭代 > 0)，需要通过 load_net_state_dict 加载网络权重。
        self._adv_net = None
        # 预先创建一个包含所有可能手牌范围索引 (range_idxs) 的张量，用于后续批处理优化。
        # RANGE_SIZE 是游戏中总的私有手牌组合数量。
        self._all_range_idxs = torch.arange(self._env_bldr.rules.RANGE_SIZE, device=self._device, dtype=torch.long)

    # --- 属性 getter ---
    @property
    def owner(self):
        """返回拥有此策略的玩家 ID。"""
        return self._owner

    @property
    def cfr_iteration(self):
        """返回此策略对应的 CFR 迭代次数。"""
        return self._cfr_iter

    @cfr_iteration.setter
    def cfr_iteration(self, value):
        """设置此策略对应的 CFR 迭代次数。"""
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"CFR 迭代次数必须是非负整数，但收到：{value}")
        self._cfr_iter = value

    @property
    def device(self):
        """返回此策略运行的设备。"""
        return self._device

    def reset(self):
        """重置策略，将优势网络设为 None。"""
        self._adv_net = None

    def get_action(self, pub_obses, range_idxs, legal_actions_lists):
        """
        根据当前策略，为一批状态采样动作。

        Args:
            pub_obses (list): 公共观察信息列表。
            range_idxs (list): 对应的私有手牌范围索引列表。
            legal_actions_lists (list): 对应的合法动作列表。

        Returns:
            np.ndarray: 采样得到的动作列表 (batch_size, 1)。
        """
        # 1. 计算动作概率分布
        a_probs = self.get_a_probs(pub_obses=pub_obses, range_idxs=range_idxs,
                                   legal_actions_lists=legal_actions_lists) # 返回 numpy 数组

        # 2. 根据概率分布进行多项式采样，获取每个状态下的一个动作
        # torch.from_numpy 将 numpy 数组转为 tensor
        # torch.multinomial 进行采样
        # .cpu().numpy() 将结果转回 CPU 上的 numpy 数组
        return torch.multinomial(torch.from_numpy(a_probs), num_samples=1).cpu().numpy()

    def get_a_probs2(self, pub_obses, range_idxs, legal_action_masks, to_np=True):
        """
        核心函数：根据优势网络计算动作概率 (策略)。
        使用了 Regret Matching+ 的思想。

        Args:
            pub_obses (list): 公共观察信息列表。
            range_idxs (list): 私有手牌范围索引列表。
            legal_action_masks (Torch.tensor): 合法动作掩码张量 (batch_size, n_actions)。
            to_np (bool): 是否将结果转换为 NumPy 数组。

        Returns:
            torch.Tensor or np.ndarray: 动作概率分布 (策略) (batch_size, n_actions)。
        """

        # 在 torch.no_grad() 上下文中执行，因为这里只是进行前向推理，不需要计算梯度
        with torch.no_grad():
            bs = len(range_idxs) # 获取批次大小

            # --- 处理迭代 0 的情况 ---
            # 如果优势网络尚未加载 (即 cfr_iter == 0)
            if self._adv_net is None:
                # 返回均匀随机策略：在所有合法动作上具有均等概率
                # 合法动作掩码除以合法动作数量 (sum(-1)) 得到均匀概率
                uniform_even_legal = legal_action_masks / (legal_action_masks.sum(-1, keepdim=True) # keepdim 保持维度方便广播
                                                           .expand_as(legal_action_masks)) # 扩展回原始形状
                # 根据需要转换为 NumPy 数组
                if to_np:
                    return uniform_even_legal.cpu().numpy()
                return uniform_even_legal
            # --- 处理迭代 > 0 的情况 ---
            else:
                # 将 range_idxs 转换为 PyTorch 张量
                range_idxs = torch.tensor(range_idxs, dtype=torch.long, device=self._device)

                # 1. 通过优势网络 (_adv_net, 即 DuelingQNet 实例) 获取优势值
                advantages = self._adv_net(pub_obses=pub_obses,           # 公共观察
                                           range_idxs=range_idxs,           # 范围索引
                                           legal_action_masks=legal_action_masks) # 合法动作掩码

                # 2. 实现 Regret Matching+ 策略计算
                #    策略与正的累积遗憾 (这里用优势近似) 成正比。
                #    strategy(a) = max(0, Regret(a)) / Sum_k[max(0, Regret(k))]

                # 首先，计算正的优势值 (ReLU)
                relu_advantages = F.relu(advantages, inplace=False)

                # 计算正优势值的总和
                sum_pos_adv = relu_advantages.sum(1, keepdim=True) # 按动作维度求和, keepdim=True
                sum_pos_adv_expanded = sum_pos_adv.expand_as(relu_advantages) # 扩展回 (batch_size, n_actions)

                # --- 处理所有优势值都 <= 0 的特殊情况 ---
                # 如果 sum_pos_adv <= 0，则 Regret Matching+ 规定选择遗憾最小 (即优势最大) 的动作。
                # 创建一个确定性策略，只选择优势最大的那个合法动作。
                best_legal_deterministic = torch.zeros_like(advantages, dtype=torch.float32, device=self._device)
                # 找到每个样本中优势最大的合法动作的索引
                # torch.where 将非法动作的优势替换为一个非常小的值，确保不会被选中
                # torch.argmax 找到最大值的索引
                bests = torch.argmax(
                    torch.where(legal_action_masks.bool(), advantages, torch.full_like(advantages, fill_value=-10e20)),
                    dim=1
                )
                # 使用高级索引将选中的最佳动作位置设为 1
                _batch_arranged = torch.arange(bs, device=self._device, dtype=torch.long)
                best_legal_deterministic[_batch_arranged, bests] = 1

                # --- 合并计算最终策略 ---
                # 使用 torch.where 根据条件选择策略：
                # 条件: sum_pos_adv_expanded > 0 (即是否存在正优势值)
                # 如果为 True: 策略 = 正优势值 / 正优势值总和 (标准的 Regret Matching+)
                # 如果为 False: 策略 = best_legal_deterministic (选择优势最大的动作)
                strategy = torch.where(
                    sum_pos_adv_expanded > 0,
                    relu_advantages / torch.clamp(sum_pos_adv_expanded, min=1e-8), # clamp 防止除零
                    best_legal_deterministic
                )

                # 根据需要转换为 NumPy 数组
                if to_np:
                    strategy = strategy.cpu().numpy()
                return strategy

    def get_a_probs(self, pub_obses, range_idxs, legal_actions_lists, to_np=True):
        """
        get_a_probs2 的便捷封装。
        输入合法动作列表，自动转换为掩码张量再调用 get_a_probs2。

        Args:
            pub_obses (list): 公共观察信息列表。
            range_idxs (list): 范围索引列表。
            legal_actions_lists (list): 合法动作 *列表* 的列表。
            to_np (bool): 是否转为 NumPy。

        Returns:
            torch.Tensor or np.ndarray: 动作概率分布。
        """
        # 在 no_grad 上下文中执行
        with torch.no_grad():
            # 使用工具函数将合法动作列表批量转换为掩码张量
            masks = rl_util.batch_get_legal_action_mask_torch(n_actions=self._env_bldr.N_ACTIONS,
                                                              legal_actions_lists=legal_actions_lists,
                                                              device=self._device, dtype=torch.float32)
            # 调用核心计算函数
            return self.get_a_probs2(pub_obses=pub_obses,
                                     range_idxs=range_idxs,
                                     legal_action_masks=masks,
                                     to_np=to_np)

    # --- 以下方法用于优化计算：当公共观察和合法动作相同时，批量计算所有手牌的策略 ---

    def get_a_probs_for_each_hand(self, pub_obs, legal_actions_list):
        """
        计算给定公共观察和合法动作下，*所有* 可能的私有手牌 (range_idxs) 的策略。

        Args:
            pub_obs (np.array): 单个公共观察信息。
            legal_actions_list (list): 单个合法动作列表。

        Returns:
            np.ndarray: 所有手牌的策略 (RANGE_SIZE, n_actions)。
        """
        # DEBUGGING 断言，检查输入类型和形状
        if self._t_prof.DEBUGGING:
            assert isinstance(pub_obs, np.ndarray)
            assert len(pub_obs.shape) == 2, "所有手牌共享相同的公共观察"
            assert isinstance(legal_actions_list[0], int), "所有手牌有相同的合法动作"

        # 调用内部函数，传入预先创建的包含所有 range_idxs 的张量
        return self._get_a_probs_of_hands(pub_obs=pub_obs, legal_actions_list=legal_actions_list,
                                          range_idxs_tensor=self._all_range_idxs)

    def get_a_probs_for_each_hand_in_list(self, pub_obs, range_idxs, legal_actions_list):
        """
        计算给定公共观察和合法动作下，*指定列表* 的私有手牌 (range_idxs) 的策略。

        Args:
            pub_obs (np.array): 单个公共观察信息。
            range_idxs (np.ndarray): 需要计算策略的手牌范围索引列表。
            legal_actions_list (list): 单个合法动作列表。

        Returns:
            np.ndarray: 指定手牌的策略 (len(range_idxs), n_actions)。
        """
        # DEBUGGING 断言
        if self._t_prof.DEBUGGING:
            # ... (断言与上面类似)
            pass

        # 调用内部函数，传入指定的手牌范围索引列表 (转换为 PyTorch 张量)
        return self._get_a_probs_of_hands(pub_obs=pub_obs, legal_actions_list=legal_actions_list,
                                          range_idxs_tensor=torch.from_numpy(range_idxs).to(dtype=torch.long,
                                                                                            device=self._device))

    def _get_a_probs_of_hands(self, pub_obs, range_idxs_tensor, legal_actions_list):
        """
        内部核心函数：批量计算多个手牌在相同公共状态下的策略。
        """
        with torch.no_grad():
            n_hands = range_idxs_tensor.size(0) # 获取手牌数量 (批次大小)

            # --- 处理迭代 0 的情况 ---
            if self._adv_net is None:
                # 返回均匀随机策略，形状为 (n_hands, n_actions)
                uniform_even_legal = torch.zeros((self._env_bldr.N_ACTIONS,), dtype=torch.float32, device=self._device)
                uniform_even_legal[legal_actions_list] = 1.0 / len(legal_actions_list)
                uniform_even_legal = uniform_even_legal.unsqueeze(0).expand(n_hands, self._env_bldr.N_ACTIONS)
                return uniform_even_legal.cpu().numpy()
            # --- 处理迭代 > 0 的情况 ---
            else:
                # 创建合法动作掩码，并扩展到 (n_hands, n_actions)
                legal_action_masks = rl_util.get_legal_action_mask_torch(n_actions=self._env_bldr.N_ACTIONS,
                                                                         legal_actions_list=legal_actions_list,
                                                                         device=self._device, dtype=torch.float32)
                legal_action_masks = legal_action_masks.unsqueeze(0).expand(n_hands, -1) # -1 表示该维度不变

                # --- 批量通过优势网络计算优势值 ---
                # 将单个 pub_obs 复制 n_hands 次，与 range_idxs_tensor 对应
                advantages = self._adv_net(pub_obses=[pub_obs] * n_hands, # 输入 n_hands 个相同的 pub_obs
                                           range_idxs=range_idxs_tensor, # 对应的 n_hands 个不同的 range_idx
                                           legal_action_masks=legal_action_masks) # 相同的掩码

                # --- 后续的 Regret Matching+ 计算逻辑与 get_a_probs2 完全相同 ---
                # (计算 relu_advantages, sum_pos_adv_expanded, best_legal_deterministic, strategy)
                # ... (代码省略，与 get_a_probs2 中相同) ...
                relu_advantages = F.relu(advantages, inplace=False)
                sum_pos_adv_expanded = relu_advantages.sum(1, keepdim=True).expand_as(relu_advantages)
                best_legal_deterministic = torch.zeros_like(advantages, dtype=torch.float32, device=self._device)
                bests = torch.argmax(
                    torch.where(legal_action_masks.bool(), advantages, torch.full_like(advantages, fill_value=-10e20)),
                    dim=1
                )
                _batch_arranged = torch.arange(n_hands, device=self._device, dtype=torch.long)
                best_legal_deterministic[_batch_arranged, bests] = 1
                strategy = torch.where(
                    sum_pos_adv_expanded > 0,
                    relu_advantages / torch.clamp(sum_pos_adv_expanded, min=1e-8),
                    best_legal_deterministic,
                )

                # 返回 NumPy 数组结果
                return strategy.cpu().numpy()

    # --- 状态管理方法 ---

    def state_dict(self):
        """返回包含策略状态的字典，用于保存检查点。"""
        return {
            "owner": self._owner, # 玩家 ID
            "net": self.net_state_dict(), # 优势网络的状态字典
            "iter": self._cfr_iter, # 迭代次数
        }

    @staticmethod
    def build_from_state_dict(t_prof, env_bldr, device, state):
        """静态方法：从状态字典构建一个新的 IterationStrategy 实例。"""
        # 1. 创建一个新的实例，传入基本信息
        s = IterationStrategy(t_prof=t_prof, env_bldr=env_bldr, device=device,
                              owner=state["owner"], cfr_iter=state["iter"])
        # 2. 加载状态字典 (包括网络权重)
        s.load_state_dict(state=state)
        return s

    def load_state_dict(self, state):
        """从状态字典加载策略状态。"""
        assert self._owner == state["owner"], "加载的状态字典所有者不匹配！"
        # 加载网络状态字典 (会创建网络实例)
        self.load_net_state_dict(state["net"])
        # 更新迭代次数
        self._cfr_iter = state["iter"]

    def net_state_dict(self):
        """获取优势网络的状态字典，处理网络为 None 的情况。"""
        if self._adv_net is None:
            return None
        return self._adv_net.state_dict()

    def load_net_state_dict(self, state_dict):
        """加载优势网络的状态字典。如果网络不存在则创建它。"""
        if state_dict is None:
            # 如果传入的状态字典为 None (通常只在迭代 0 时)，则不加载网络。
            # 这种情况下，get_a_probs 会返回均匀随机策略。
            self._adv_net = None
            return
        else:
            # 如果网络实例不存在 (_adv_net is None)，则先创建它。
            # 注意：这里硬编码了使用 DuelingQNet，并从 t_prof 中获取其参数。
            if self._adv_net is None:
                 # 从训练配置中获取优势网络参数 (adv_net_args)
                adv_net_args = self._t_prof.module_args["adv_training"].adv_net_args
                # 实例化 DuelingQNet
                self._adv_net = DuelingQNet(q_args=adv_net_args,
                                            env_bldr=self._env_bldr,
                                            device=self._device)

            # 加载状态字典到网络中
            self._adv_net.load_state_dict(state_dict)
            # 确保网络在正确的设备上
            self._adv_net.to(self._device)

        # --- 重要：将网络设置为评估模式 (Evaluation Mode) ---
        # 这会禁用 Dropout 和 BatchNorm 的更新等训练特有行为。
        self._adv_net.eval()
        # 禁用网络参数的梯度计算，因为 IterationStrategy 只用于推理。
        for param in self._adv_net.parameters():
            param.requires_grad = False

    def get_copy(self, device=None):
        """创建此 IterationStrategy 对象的副本，可选地放在不同设备上。"""
        _device = self._device if device is None else device
        # 使用静态方法从当前状态字典构建新实例
        return IterationStrategy.build_from_state_dict(t_prof=self._t_prof, env_bldr=self._env_bldr,
                                                       device=_device, state=self.state_dict())