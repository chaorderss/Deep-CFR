# 版权信息
# Copyright (c) 2019 Eric Steinberger

"""
文件描述: 定义环境包装器构建器 (EnvWrapperBuilder)。

作用: 包装一个扑克环境，以追踪历史等信息。包装器不直接构造，
而是通过 env_builder 接口创建。创建 env_builder 是创建环境
(可能带有包装器) 的起点。
"""

from PokerRL.game.Poker import Poker # 导入扑克游戏相关的常量 (如动作类型)
# 导入环境包装器构建器的基类 (具体实现未在本片段中给出)
from PokerRL.game._.EnvWrapperBuilderBase import EnvWrapperBuilderBase as _EnvWrapperBuilderBase
# 导入具体的环境包装器类
from PokerRL.game._.wrappers.FlatHULimitPokerHistoryWrapper import \
    FlatHULimitPokerHistoryWrapper as _FlatHULimitPokerHistoryWrapper # 用于限注单挑扑克的扁平化历史包装器
from PokerRL.game._.wrappers.RecurrentHistoryWrapper import RecurrentHistoryWrapper as _RecurrentHistoryWrapper # 添加循环历史信息的包装器
from PokerRL.game._.wrappers.Vanilla import VanillaWrapper as _VanillaWrapper # 最基础的包装器 (可能不修改环境)
from PokerRL.game._.wrappers.NOLimitPokerHistoryWrapper import NOLimitPokerHistoryWrapper

# --- 定义各种环境构建器类 ---

class VanillaEnvBuilder(_EnvWrapperBuilderBase):
    """
    最基础的环境构建器。
    它使用 _VanillaWrapper，通常表示不对原始环境做过多修改，
    提供一个"原味"的环境接口。
    文档请参考对应的包装器类文件 (_VanillaWrapper)。
    """
    WRAPPER_CLS = _VanillaWrapper # 指定此类构建器使用的包装器类

class HistoryEnvBuilder(_EnvWrapperBuilderBase):
    """
    添加循环历史记录的环境构建器。
    它使用 _RecurrentHistoryWrapper，会将动作/状态序列添加到观察中。
    这对于使用循环神经网络 (RNN, LSTM) 作为策略或值函数模型的场景非常有用。
    文档请参考对应的包装器类文件 (_RecurrentHistoryWrapper)。
    """
    WRAPPER_CLS = _RecurrentHistoryWrapper # 指定使用的包装器类

    def __init__(self, env_cls, env_args, invert_history_order=False):
        """
        初始化 HistoryEnvBuilder。

        Args:
            env_cls: 基础扑克环境的类 (例如 LeducHoldem)。
            env_args: 传递给基础环境类的参数。
            invert_history_order (bool): 是否反转历史记录的顺序 (可能对某些 RNN 实现有用)。
        """
        super().__init__(env_cls=env_cls, env_args=env_args) # 调用基类初始化
        self.invert_history_order = invert_history_order # 存储特定于此构建器的参数

class FlatLimitPokerEnvBuilder(_EnvWrapperBuilderBase):
    """
    用于限注单挑 (Heads-Up Fixed-Limit) 扑克的扁平化历史环境构建器。
    这个构建器非常特殊，它使用 _FlatHULimitPokerHistoryWrapper 将整个游戏过程中的
    动作历史编码成一个 *固定长度的扁平向量*，并附加到公共观察后面。
    这使得可以使用非循环的神经网络 (如 MLP) 来处理历史信息，但仅适用于规则的限注游戏。
    文档请参考对应的包装器类文件 (_FlatHULimitPokerHistoryWrapper)。
    """
    WRAPPER_CLS = _FlatHULimitPokerHistoryWrapper # 指定使用的包装器类

    def __init__(self, env_cls, env_args):
        """
        初始化 FlatLimitPokerEnvBuilder。

        Args:
            env_cls: 基础扑克环境的类。
            env_args: 传递给基础环境类的参数。
        """
        # 断言检查：确保环境是固定限注游戏，并且是两人游戏 (Heads-Up)
        assert env_cls.IS_FIXED_LIMIT_GAME, "此构建器仅适用于固定限注游戏"
        assert env_args.n_seats == 2, "此构建器仅适用于两人游戏"

        # --- 计算扁平化动作历史向量的大小和索引偏移 ---
        self._VEC_ROUND_OFFSETS = {} # 存储每个下注轮次在向量中的起始偏移量
        self._VEC_HALF_ROUND_SIZE = {} # 存储每个轮次中 *单个玩家* 的动作向量部分的大小
        self.action_vector_size = 0 # 初始化总的动作历史向量大小

        # 遍历游戏规则中定义的所有下注轮次 (如 Preflop, Flop, Turn, River)
        for r in env_cls.RULES.ALL_ROUNDS_LIST:
            # 记录当前轮次的起始偏移量
            self._VEC_ROUND_OFFSETS[r] = self.action_vector_size
            # 计算单个玩家在该轮次需要多少空间来编码动作历史
            # 每个动作序列位置需要编码两种主要动作类型：下注/加注 (BET_RAISE) 或 跟注/过牌 (CHECK_CALL)
            # 动作序列的最大长度由该轮次允许的最大加注次数 (MAX_N_RAISES_PER_ROUND) 决定 (+2 是为了考虑初始行动和可能的封顶)
            self._VEC_HALF_ROUND_SIZE[r] = len([Poker.BET_RAISE, Poker.CHECK_CALL]) * (
                env_cls.MAX_N_RAISES_PER_ROUND[r] + 2)
            # 将两个玩家在该轮次所需的空间加到总向量大小中
            self.action_vector_size += self._VEC_HALF_ROUND_SIZE[r] * env_args.n_seats # n_seats 固定为 2

        # 调用基类初始化
        super().__init__(env_cls=env_cls, env_args=env_args)

    def _get_num_public_observation_features(self):
        """
        重写方法 (可能来自基类)：计算包装后环境的公共观察特征数量。
        它等于原始环境的公共观察特征数 加上 扁平化动作历史向量的大小。
        """
        # 创建一个临时的环境实例以获取原始观察空间大小
        _env = self.env_cls(env_args=self.env_args, lut_holder=self.lut_holder, is_evaluating=True)
        # 返回原始大小 + 计算得到的动作向量大小
        return _env.observation_space.shape[0] + self.action_vector_size

    def get_vector_idx(self, round_, p_id, nth_action_this_round, action_idx):
        """
        计算一个具体的动作在扁平化历史向量中的索引位置。
        这个方法会被包装器或使用此向量的模块调用。

        Args:
            round_: 当前下注轮次。
            p_id (int): 执行动作的玩家 ID (0 或 1)。
            nth_action_this_round (int): 这是该玩家在本轮中执行的第几个动作 (从 0 开始计数)。
            action_idx (int): 动作的索引 (Poker.BET_RAISE=1, Poker.CHECK_CALL=2, Poker.FOLD=0)。

        Returns:
            int: 该动作在扁平向量中的索引。
        """
        # 计算索引的逻辑：
        # 1. 获取该轮次的起始偏移量: self._VEC_ROUND_OFFSETS[round_]
        # 2. 加上该玩家的偏移量: p_id * self._VEC_HALF_ROUND_SIZE[round_]
        # 3. 加上这是该玩家本轮第几次行动的偏移量: nth_action_this_round * 2 (因为每次行动记录2种主要动作类型)
        # 4. 加上具体动作的偏移量: action_idx - 1 (因为不记录 FOLD=0，只记录 BET_RAISE=1 和 CHECK_CALL=2)
        # *2 代表 len([Poker.BET_RAISE, Poker.CHECK_CALL])
        # 如果动作是 FOLD，这个观察向量理论上不会被使用 (因为游戏通常会结束或进入下一轮)
        return self._VEC_ROUND_OFFSETS[round_] \
               + p_id * self._VEC_HALF_ROUND_SIZE[round_] \
               + nth_action_this_round * 2 \
               + action_idx - 1

class NoLimitPokerEnvBuilder(_EnvWrapperBuilderBase):
    """
    用于无限注扑克 (No-Limit) 且支持多人 (N-player) 的扁平化历史环境构建器。
    这个构建器使用 NOLimitPokerHistoryWrapper 将游戏动作历史编码成一个固定长度的扁平向量，
    并附加到公共观察之后。适用于前馈神经网络。
    **依赖于配置参数和对无限注动作的离散化处理。**
    """
    WRAPPER_CLS = NOLimitPokerHistoryWrapper # 指定使用的包装器类

    def __init__(self, env_cls, env_args,
                 num_discretized_actions,
                 max_actions_per_round_heuristic):
        """
        初始化 NoLimitPokerEnvBuilder。

        Args:
            env_cls: 基础扑克环境的类 (应为无限注类型)。
            env_args: 传递给基础环境类的参数。
            num_discretized_actions (int): 离散化后的总动作数量。
                                           这包括 FOLD, CHECK/CALL 以及所有不同的下注/加注大小档位。
                                           例如：FOLD, CALL, BET_33%, BET_50%, BET_100%, ALL_IN -> num_discretized_actions = 6。
            max_actions_per_round_heuristic (int): 每个玩家在单轮中预计最多执行的动作次数。
                                                    用于确定历史向量的大小。需要足够大以覆盖绝大多数情况。
        """
        # 检查座位数是否至少为 2
        assert env_args.n_seats >= 2, "此构建器适用于2人及以上游戏"
        # (可选) 检查是否不是固定限注游戏
        assert not getattr(env_cls, 'IS_FIXED_LIMIT_GAME', False), "此构建器不适用于固定限注游戏"

        # 存储配置参数
        self.num_discretized_actions = num_discretized_actions
        self.max_actions_per_round_heuristic = max_actions_per_round_heuristic

        # --- 计算扁平化动作历史向量的大小和索引偏移 ---
        self._VEC_ROUND_OFFSETS = {} # 存储每个下注轮次在向量中的起始偏移量
        # 存储每个轮次中 *单个玩家* 的动作向量部分的大小
        self._VEC_ACTIONS_PER_PLAYER_ROUND_SIZE = {}
        self.action_vector_size = 0 # 初始化总的动作历史向量大小

        # 遍历游戏规则中定义的所有下注轮次
        for r in env_cls.RULES.ALL_ROUNDS_LIST:
            # 记录当前轮次的起始偏移量
            self._VEC_ROUND_OFFSETS[r] = self.action_vector_size
            # 计算单个玩家在该轮次需要多少空间来编码动作历史
            # 空间 = 预计每轮最大动作数 * 离散化后的总动作数
            round_size_per_player = self.max_actions_per_round_heuristic * self.num_discretized_actions
            self._VEC_ACTIONS_PER_PLAYER_ROUND_SIZE[r] = round_size_per_player
            # 将所有玩家在该轮次所需的空间加到总向量大小中
            self.action_vector_size += round_size_per_player * env_args.n_seats

        # 调用基类初始化 (需要传递 env_cls 和 env_args)
        super().__init__(env_cls=env_cls, env_args=env_args)
        # 注意：基类初始化可能会依赖 lut_holder，如果这里没有初始化，需要确保基类能处理
        # 或者像 FlatLimit 版本一样，先初始化 lut_holder
        # self.lut_holder = env_cls.get_lut_holder() # 假设基类需要这个

    def _get_num_public_observation_features(self):
        """
        计算包装后环境的公共观察特征总数。
        等于原始环境的公共观察特征数 加上 扁平化动作历史向量的大小。
        """
        # 创建一个临时的环境实例以获取原始观察空间大小
        # 确保传递 lut_holder 如果基础环境需要
        try:
            # 尝试获取 lut_holder，如果不存在则可能不需要
            lut = self.lut_holder
        except AttributeError:
            lut = self.env_cls.get_lut_holder() # 尝试从类获取

        _env = self.env_cls(env_args=self.env_args, lut_holder=lut, is_evaluating=True)
        # 返回原始大小 + 计算得到的动作向量大小
        return _env.observation_space.shape[0] + self.action_vector_size

    def get_vector_idx(self, round_, p_id, nth_action_this_round, action_idx):
        """
        计算一个具体的 *离散化* 动作在扁平化历史向量中的索引位置。

        Args:
            round_: 当前下注轮次。
            p_id (int): 执行动作的玩家 ID。
            nth_action_this_round (int): 这是该玩家在本轮中执行的第几个动作 (从 0 开始计数)。
            action_idx (int): *离散化后* 的动作索引 (0 到 num_discretized_actions - 1)。

        Returns:
            int: 该动作在扁平向量中的索引。

        Raises:
            IndexError: 如果 nth_action_this_round 超出预期的最大值。
            KeyError: 如果 round_ 无效。
        """
        # 检查动作序号是否超出预设的启发式限制
        if nth_action_this_round >= self.max_actions_per_round_heuristic:
            # 可以选择抛出错误，或者返回一个特殊索引/记录日志
            raise IndexError(f"玩家 {p_id} 在轮次 {round_} 的动作序号 "
                             f"({nth_action_this_round}) 超出预设限制 "
                             f"({self.max_actions_per_round_heuristic})")
        # 检查动作索引是否在有效范围内
        if not (0 <= action_idx < self.num_discretized_actions):
             raise ValueError(f"无效的离散化动作索引: {action_idx}。"
                              f" 应在 [0, {self.num_discretized_actions -1}] 范围内。")

        # 计算索引的逻辑：
        # 1. 获取该轮次的起始偏移量
        round_offset = self._VEC_ROUND_OFFSETS[round_]
        # 2. 加上该玩家的偏移量
        player_offset = p_id * self._VEC_ACTIONS_PER_PLAYER_ROUND_SIZE[round_]
        # 3. 加上这是该玩家本轮第几次行动的偏移量
        #    每次行动占用 num_discretized_actions 个位置
        action_num_offset = nth_action_this_round * self.num_discretized_actions
        # 4. 加上具体离散化动作的索引
        action_type_offset = action_idx

        # 返回最终计算出的索引
        return round_offset + player_offset + action_num_offset + action_type_offset

# 包含所有已定义构建器的列表
ALL_BUILDERS = [
    HistoryEnvBuilder,
    FlatLimitPokerEnvBuilder,
    VanillaEnvBuilder,
    NoLimitPokerEnvBuilder
]