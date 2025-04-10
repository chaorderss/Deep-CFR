# Copyright (c) 2019 Eric Steinberger

# 导入必要的库
import copy # 用于深拷贝对象，防止修改原始数据
import time # 可能用于调试或人类玩家接口的延迟

import numpy as np # 用于数值计算，特别是数组操作
from gym import spaces # 来自 OpenAI Gym 库，用于定义强化学习环境的观察和动作空间

# 从 PokerRL 框架内部导入相关模块
from PokerRL.game.Poker import Poker # 定义扑克相关的常量（如动作、轮次）
from PokerRL.game.PokerEnvStateDictEnums import EnvDictIdxs, PlayerDictIdxs # 定义状态字典中使用的枚举或索引
from PokerRL.game._.rl_env.base._Deck import DeckOfCards # 牌库类
from PokerRL.game._.rl_env.base._PokerPlayer import PokerPlayer # 扑克玩家类

# PokerEnv 类: PokerRL 框架中所有扑克环境的基础抽象类。
# 它实现了通用的扑克逻辑、状态管理和与强化学习代理交互的接口。
# 具体的游戏规则（如德州扑克、奥马哈、限注/无限注等）由继承这个基类的子类来实现。
class PokerEnv:
    """
    这个抽象类实现了扑克环境的通用函数和属性。子类可以定义环境的具体规则和超参数。
    这个基础引擎支持单挑（HeadsUp）和多人扑克。
    它被 Limit（限注）和 Discrete（离散动作）包装器作为子基类，因为这个基类默认会采用无限注（No-Limit）规则集。

    功能和动态:
        奖励 (reward):
            如果到达终止状态:
                返回一个 numpy 数组 (shape=座位数)，包含每个玩家 "本局结束时的筹码 - 初始筹码"。
                奖励可以根据创建环境实例时传递的参数设置进行缩放或不缩放。
            否则:
                返回一个 numpy 零数组 (shape=座位数)。

        观察 (obs):
            如果到达终止状态:
                返回一个 numpy 零数组，形状与非终止状态相同。
            否则:
                包含环境的公共状态（筹码、底池、公共牌等）。不包括任何玩家的底牌（或任何私有信息）！
                要获取玩家的私有信息，请查看 .get_hole_cards_of_player()。

        动作 (actions):
            对于 DiscretePokerEnv 子类:
                0 = FOLD (弃牌), 1 = CHECK/CALL (看牌/跟注)
                2 = RAISE_SIZE_1 (加注大小1), ... N = RAISE_SIZE_N-1 (加注大小N-1)

            对于直接子类 (例如，像无限注德州扑克这样具有“近似连续下注大小”的扑克游戏):
                元组: (action, raise size)，其中 action 为 0 表示 FOLD, 1 表示 CHECK/CALL, 2 表示 BET/RAISE (下注/加注)。
                raise_size 总是需要传递，但仅在 action 为 2 (即 BET/RAISE) 时才重要。
                它表示玩家下注后希望总共下注的筹码数量。例如，如果当前下注是 30，代理想再下注 60 筹码，
                那么元组应该是 (2, 90)。

        牌的表示 (card representations):
            单张牌可以用两种形式引用：1D 和 2D。
            1D 指的是将牌映射到一个唯一的整数。
            2D 指的是一个数组/元组/列表，第一个索引是点数（rank），第二个索引是花色（suit）。

            一手牌或公共牌可以看作是这些牌的数组。例如，board_2d 指的是一个二维数组 [[card1_rank, card1_suit], [card2_rank, card2_suit], ...]。

            或者，一手多张牌可以通过它在按第一张牌的 1D 表示、然后按第二张牌等排序的数组中的索引来引用。
            这种格式在此框架中称为 "range_idx"。
    """

    # _____________________ 需要由子类根据其规则定义的变量 _____________________
    # 这些属性定义了具体扑克游戏的参数，必须由继承 PokerEnv 的子类指定。

    SMALL_BLIND = NotImplementedError # 小盲注大小
    BIG_BLIND = NotImplementedError # 大盲注大小
    ANTE = NotImplementedError      # 底注大小
    SMALL_BET = NotImplementedError # （限注游戏）小注
    BIG_BET = NotImplementedError   # （限注游戏）大注
    DEFAULT_STACK_SIZE = NotImplementedError # 默认起始筹码量

    EV_NORMALIZER = NotImplementedError  # 用于计算例如 MBB/H 的筹码标准化除数
    WIN_METRIC = NotImplementedError  # 用于绘制与固定量（如 Poker.MeasureAnte）相关的图表

    N_HOLE_CARDS = NotImplementedError  # 每个玩家的底牌数量
    N_RANKS = NotImplementedError       # 牌的点数种类数量 (e.g., 13 for A-K)
    N_SUITS = NotImplementedError       # 牌的花色种类数量 (e.g., 4 for Spades, Hearts, Diamonds, Clubs)
    N_CARDS_IN_DECK = NotImplementedError # 牌库中的总牌数 (N_RANKS * N_SUITS)
    RANGE_SIZE = NotImplementedError    # 可能的唯一底牌组合总数

    BTN_IS_FIRST_POSTFLOP = NotImplementedError # 翻牌后按钮位 (BTN) 是否首先行动
    FIRST_ACTION_NO_CALL = False # 翻牌前第一个动作是否不允许跟注（特殊规则）

    IS_FIXED_LIMIT_GAME = NotImplementedError # 是否是限注游戏
    IS_POT_LIMIT_GAME = NotImplementedError   # 是否是底池限注游戏

    # 仅限注游戏相关!
    MAX_N_RAISES_PER_ROUND = NotImplementedError # 每轮允许的最大加注次数
    ROUND_WHERE_BIG_BET_STARTS = NotImplementedError # 大注开始的轮次

    # 观察模式
    SUITS_MATTER = NotImplementedError  # 花色是否影响牌力（例如，是否存在同花）

    N_FLOP_CARDS = NotImplementedError      # 翻牌圈公共牌数量
    N_TURN_CARDS = NotImplementedError      # 转牌圈公共牌数量
    N_RIVER_CARDS = NotImplementedError     # 河牌圈公共牌数量
    N_TOTAL_BOARD_CARDS = NotImplementedError # 总公共牌数量

    # 这个列表不能跳过轮次。设为 [PREFLOP, FLOP] 可以，但 [PREFLOP, TURN] 不行。
    ALL_ROUNDS_LIST = NotImplementedError # 包含游戏中所有下注轮次的列表

    # 将轮次映射到其他轮次的字典。
    ROUND_BEFORE = NotImplementedError # 映射: 当前轮次 -> 前一轮次
    ROUND_AFTER = NotImplementedError  # 映射: 当前轮次 -> 下一轮次

    # 将整数映射到字符串以打印牌张
    RANK_DICT = NotImplementedError # e.g., {0: '2', ..., 12: 'A'}
    SUIT_DICT = NotImplementedError # e.g., {0: 's', 1: 'h', 2: 'd', 3: 'c'}

    # 包含游戏遵循的基本规则集的类
    RULES = NotImplementedError

    # ____________________________________________________ 构造函数 ___________________________________________________
    def __init__(self,
                 env_args,
                 lut_holder,
                 is_evaluating,
                 ):
        """
        初始化 PokerEnv 实例。

        Args:
            env_args:               根据游戏类型，是 PokerEnvArgs 或 DiscretePokerEnvArgs 的实例。
                                    包含环境特定的配置，如座位数、初始筹码、是否标准化奖励等。

            lut_holder:             根据游戏类型，是 LutHolder 的子类实例。不检查是否传递了正确的类型，
                                    请确保传递正确的！lut_holder 理论上可以由该类的实例封装创建，
                                    但为了优化（例如，每台机器只有一个，而不是每个环境一个），我们将其传入。
                                    用于快速计算牌力、范围索引等。

            is_evaluating (bool):   环境是否应在评估模式下生成（即无随机化）或否。
                                    评估模式通常禁用随机化，如随机起始筹码。
        """
        # 断言至少有2个座位
        assert env_args.n_seats >= 2

        # 深拷贝参数以防外部修改
        self._args = copy.deepcopy(env_args)
        # 存储查找表管理器
        self.lut_holder = lut_holder
        # 存储评估模式标志
        self.IS_EVALUATING = is_evaluating

        # 初始化牌库
        self.deck = DeckOfCards(num_suits=self.N_SUITS, num_ranks=self.N_RANKS)

        # 初始化将在 _init_from_args 中设置的内部状态变量
        self.BTN_POS = NotImplementedError # 按钮位置索引
        self.SB_POS = NotImplementedError  # 小盲注位置索引
        self.BB_POS = NotImplementedError  # 大盲注位置索引
        self._USE_SIMPLE_HU_OBS = NotImplementedError # 是否使用简化的单挑观察
        self.RETURN_PRE_TRANSITION_STATE_IN_INFO = NotImplementedError # 是否在 info 中返回转换前的状态
        self.N_SEATS = NotImplementedError # 座位数
        self.MAX_CHIPS = NotImplementedError # 可能的最大筹码量（用于观察空间边界）
        self.STACK_RANDOMIZATION_RANGE = NotImplementedError # 起始筹码随机化范围
        self.REWARD_SCALAR = NotImplementedError # 奖励缩放因子
        self.seats = NotImplementedError # 玩家座位列表

        # 从传入的参数初始化内部状态变量
        self._init_from_args(env_args=env_args, is_evaluating=is_evaluating)

        # ______________________________  观察空间和动作空间 ______________________________
        # 构建观察空间（向量表示）及其索引字典
        self.observation_space, self.obs_idx_dict, self.obs_parts_idxs_dict = self._construct_obs_space()
        # 定义底牌空间的形状（用于可能的内部使用或检查）
        self.hole_card_space_shape = [self.N_HOLE_CARDS, 2]

        # __________________________________  本局游戏状态变量  __________________________________
        # 所有这些变量都应在训练脚本中通过调用 env.reset() 来初始化。
        self.current_round = None # 当前下注轮 (e.g., Poker.PREFLOP)
        self.side_pots = None     # 边池列表 (list with len=n_seats)
        self.main_pot = None      # 主池筹码量
        self.board = None         # 公共牌 (np.ndarray(shape=(n_cards, 2)))
        self.last_action = None   # 最后执行的动作 [action_idx, _raise_amount, player.seat_id]
        self.capped_raise = CappedRaise() # 处理特殊的全下规则的对象
        self.current_player = None # 当前需要行动的 PokerPlayer 实例
        self.last_raiser = None    # 最近一个加注的 PokerPlayer 实例
        self.n_actions_this_episode = None # 本局游戏中已执行的动作总数

        # 仅限注游戏相关
        self.n_raises_this_round = NotImplementedError # 本轮已发生的加注次数

    def _construct_obs_space(self):
        """
        构建环境状态的向量表示（Observation），只包含公共信息。
        使用 gym.spaces 定义观察向量的每个部分。
        所有筹码值的最大值可以达到 n_seats，因为我们通过除以平均起始筹码进行标准化。
        """
        obs_idx_dict = {} # 字典：名称 -> 观察向量中的索引
        obs_parts_idxs_dict = { # 字典：部分名称 -> 对应的索引列表
            "board": [],
            "players": [[] for _ in range(self.N_SEATS)],
            "table_state": [],
        }
        next_idx = [0]  # 使用列表作为可变对象来跟踪下一个可用索引

        # 辅助函数：定义离散空间部分
        def get_discrete(size, name, _curr_idx):
            obs_idx_dict[name] = _curr_idx[0]
            _curr_idx[0] += 1
            return spaces.Discrete(size) # e.g., one-hot 编码

        # 辅助函数：定义连续（盒子）空间部分
        def get_new_box(name, _curr_idx, high, low=0):
            obs_idx_dict[name] = _curr_idx[0]
            _curr_idx[0] += 1
            # Box 定义了连续值的范围和形状
            return spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32)

        # --- 情况1: 单挑 (N_SEATS == 2) 且使用简化观察 ---
        if (self.N_SEATS == 2) and self._USE_SIMPLE_HU_OBS:
            # __________________________  游戏状态的公共信息  _________________________
            _k = next_idx[0] # 记录当前部分的起始索引
            _table_space = [ # (盲注包含在观察中，以便代理在标准化后了解起始筹码的相对大小)
                get_new_box("ante", next_idx, self.N_SEATS),                 # 底注
                get_new_box("small_blind", next_idx, self.N_SEATS),           # 小盲注
                get_new_box("big_blind", next_idx, self.N_SEATS),            # 大盲注
                get_new_box("min_raise", next_idx, self.N_SEATS),            # 当前最小加注总额
                get_new_box("pot_amt", next_idx, self.N_SEATS),             # 主池金额
                get_new_box("total_to_call", next_idx, self.N_SEATS),        # 需要跟注的总额
                get_new_box("last_action_how_much", next_idx, self.N_SEATS), # 最后动作的筹码量 (如果是加注)
            ]
            # 最后动作类型 (one-hot: FOLD, CHECK_CALL, BET_RAISE)
            for i in range(3):
                _table_space.append(get_discrete(1, "last_action_what_" + str(i), next_idx))

            # 最后动作执行者 (one-hot)
            for i in range(self.N_SEATS):
                _table_space.append(get_discrete(1, "last_action_who_" + str(i), next_idx))

            # 当前行动者 (one-hot)
            for i in range(self.N_SEATS):
                _table_space.append(get_discrete(1, "p" + str(i) + "_acts_next", next_idx))

            # 当前下注轮 (one-hot)
            for i in range(max(self.ALL_ROUNDS_LIST) + 1):
                _table_space.append(get_discrete(1, "round_" + Poker.INT2STRING_ROUND[i], next_idx)),

            # 将这部分索引添加到 parts_dict 中，方便代理切片
            obs_parts_idxs_dict["table_state"] += list(range(_k, next_idx[0]))

            # __________________________  每个玩家的公共信息  ________________________
            _player_space = []
            for i in range(self.N_SEATS):
                _k = next_idx[0] # 记录当前玩家部分的起始索引
                _player_space += [
                    get_new_box("stack_p" + str(i), next_idx, self.N_SEATS),      # 玩家筹码量
                    get_new_box("curr_bet_p" + str(i), next_idx, self.N_SEATS),  # 玩家当前下注额
                    get_discrete(1, "is_allin_p" + str(i), next_idx),        # 玩家是否全下 (one-hot, 但大小为1)
                ]

                # 将当前玩家的索引添加到 parts_dict 中
                obs_parts_idxs_dict["players"][i] += list(range(_k, next_idx[0]))

            # _______________________________  公共牌 (即 board)  ______________________________
            _board_space = []
            _k = next_idx[0] # 记录公共牌部分的起始索引
            for i in range(self.N_TOTAL_BOARD_CARDS): # 对每张可能的公共牌
                x = []
                # 点数 (one-hot)
                for j in range(self.N_RANKS):
                    x.append(get_discrete(1, str(i) + "th_board_card_rank_" + str(j), next_idx))

                # 花色 (one-hot)
                for j in range(self.N_SUITS):
                    x.append(get_discrete(1, str(i) + "th_board_card_suit_" + str(j), next_idx))

                _board_space += x

            # 将公共牌部分的索引添加到 parts_dict 中
            obs_parts_idxs_dict["board"] += list(range(_k, next_idx[0]))

            # __________________________  返回完整的观察空间  __________________________
            # 使用 Tuple 组合所有部分
            _observation_space = spaces.Tuple(_table_space + _player_space + _board_space)
            # 定义观察空间的形状（扁平化后的一维向量长度）
            _observation_space.shape = [len(_observation_space.spaces)]

        # --- 情况2: 多人游戏 或 不使用简化的单挑观察 ---
        else:
            # __________________________  游戏状态的公共信息  _________________________
            # 与单挑类似，但可能包含多人游戏特有的信息（如边池）
            _k = next_idx[0]
            _table_space = [ # (盲注包含在观察中...)
                get_new_box("ante", next_idx, self.N_SEATS),                 # 底注
                get_new_box("small_blind", next_idx, self.N_SEATS),           # 小盲注
                get_new_box("big_blind", next_idx, self.N_SEATS),            # 大盲注
                get_new_box("min_raise", next_idx, self.N_SEATS),            # 当前最小加注总额
                get_new_box("pot_amt", next_idx, self.N_SEATS),             # 主池金额
                get_new_box("total_to_call", next_idx, self.N_SEATS),        # 需要跟注的总额
                get_new_box("last_action_how_much", next_idx, self.N_SEATS), # 最后动作的筹码量
            ]
             # 最后动作类型 (one-hot)
            for i in range(3):
                _table_space.append(get_discrete(1, "last_action_what_" + str(i), next_idx))

            # 最后动作执行者 (one-hot)
            for i in range(self.N_SEATS):
                _table_space.append(get_discrete(1, "last_action_who_" + str(i), next_idx))

            # 当前行动者 (one-hot)
            for i in range(self.N_SEATS):
                _table_space.append(get_discrete(1, "p" + str(i) + "_acts_next", next_idx))

            # 当前下注轮 (one-hot)
            for i in range(max(self.ALL_ROUNDS_LIST) + 1):
                _table_space.append(get_discrete(1, "round_" + Poker.INT2STRING_ROUND[i], next_idx)),

            # 边池大小 (每个座位一个边池)
            for i in range(self.N_SEATS):
                # 这里 high=1 可能是标准化后的值，或者表示是否存在该边池？需要确认具体实现
                _table_space.append(get_new_box("side_pot_" + str(i), next_idx, 1))

            # 将这部分索引添加到 parts_dict
            obs_parts_idxs_dict["table_state"] += list(range(_k, next_idx[0]))

            # __________________________  每个玩家的公共信息  ________________________
            _player_space = []
            for i in range(self.N_SEATS):
                _k = next_idx[0]
                _player_space += [
                    get_new_box("stack_p" + str(i), next_idx, self.N_SEATS),           # 玩家筹码量
                    get_new_box("curr_bet_p" + str(i), next_idx, self.N_SEATS),       # 玩家当前下注额
                    get_discrete(1, "has_folded_this_episode_p" + str(i), next_idx), # 玩家本局是否已盖牌
                    get_discrete(1, "is_allin_p" + str(i), next_idx),             # 玩家是否全下
                ]
                # 玩家的边池级别 (one-hot)
                for j in range(self.N_SEATS):
                    _player_space.append(
                        get_discrete(1, "side_pot_rank_p" + str(i) + "_is_" + str(j), next_idx))

                # 将当前玩家的索引添加到 parts_dict
                obs_parts_idxs_dict["players"][i] += list(range(_k, next_idx[0]))

            # _______________________________  公共牌 (即 board)  ______________________________
            # 与单挑情况相同
            _board_space = []
            _k = next_idx[0]
            for i in range(self.N_TOTAL_BOARD_CARDS):
                x = []
                 # 点数 (one-hot)
                for j in range(self.N_RANKS):
                    x.append(get_discrete(1, str(i) + "th_board_card_rank_" + str(j), next_idx))

                # 花色 (one-hot)
                for j in range(self.N_SUITS):
                    x.append(get_discrete(1, str(i) + "th_board_card_suit_" + str(j), next_idx))

                _board_space += x

            # 将公共牌部分的索引添加到 parts_dict
            obs_parts_idxs_dict["board"] += list(range(_k, next_idx[0]))

            # __________________________  返回完整的观察空间  __________________________
            # 使用 Tuple 组合所有部分
            _observation_space = spaces.Tuple(_table_space + _player_space + _board_space)
            # 定义观察空间的形状（扁平化后的一维向量长度）
            _observation_space.shape = [len(_observation_space.spaces)]

        # 返回构建好的观察空间、索引字典和部分索引字典
        return _observation_space, obs_idx_dict, obs_parts_idxs_dict

    def _init_from_args(self, env_args, is_evaluating):
        """根据传入的参数初始化环境的具体设置"""
        a = copy.deepcopy(env_args) # 再次拷贝以防万一

        # --- 根据座位数设置盲注和按钮位置 ---
        # 单挑规则
        if a.n_seats == 2:
            self.BTN_POS = 0 # 按钮位
            self.SB_POS = 0  # 小盲注位 (通常与按钮位相同)
            self.BB_POS = 1  # 大盲注位
        # 多人 (>2) 规则
        else:
            self.BTN_POS = 0 # 按钮位
            self.SB_POS = 1  # 小盲注位
            self.BB_POS = 2  # 大盲注位

        # --- 从参数对象 a 中读取其他配置 ---
        self._USE_SIMPLE_HU_OBS = a.use_simplified_headsup_obs # 是否使用简化单挑观察
        self.RETURN_PRE_TRANSITION_STATE_IN_INFO = a.RETURN_PRE_TRANSITION_STATE_IN_INFO # 是否在 info 中返回状态
        self.N_SEATS = int(a.n_seats) # 确保座位数是整数

        # --- 计算最大可能筹码量 (用于观察空间边界) ---
        try:
            # 如果提供了起始筹码列表
            self.MAX_CHIPS = sum(a.starting_stack_sizes_list) \
                             + a.stack_randomization_range[1] * a.n_seats \
                             + 1 # +1 确保边界包含在内
        except TypeError:  # 如果起始筹码列表为 None 或包含 None，则使用默认值
            self.MAX_CHIPS = a.n_seats * (self.DEFAULT_STACK_SIZE + a.stack_randomization_range[1]) + 1

        # --- 设置起始筹码随机化范围 ---
        self.STACK_RANDOMIZATION_RANGE = a.stack_randomization_range

        # --- 设置奖励缩放因子 ---
        if a.scale_rewards: # 如果需要缩放奖励
            try:
                # 使用平均起始筹码的 1/5 作为缩放基准
                self.REWARD_SCALAR = float(sum(a.starting_stack_sizes_list)) / float(a.n_seats) / 5
            except TypeError: # 使用默认起始筹码计算
                self.REWARD_SCALAR = self.DEFAULT_STACK_SIZE / 5.0
        else: # 不缩放奖励
            self.REWARD_SCALAR = 1.0

        # --- 创建玩家座位列表 ---
        self.seats = [
            PokerPlayer(seat_id=i,          # 玩家座位 ID
                        poker_env=self,     # 关联的扑克环境实例
                        is_evaluating=is_evaluating, # 是否处于评估模式
                        starting_stack=     # 设置起始筹码
                        (a.starting_stack_sizes_list[i] # 优先使用列表中的值
                         if a.starting_stack_sizes_list is not None and i < len(a.starting_stack_sizes_list) and a.starting_stack_sizes_list[i] is not None
                         else self.DEFAULT_STACK_SIZE), # 否则使用默认值
                        stack_randomization_range=a.stack_randomization_range) # 传递随机化范围
            for i in range(a.n_seats)] # 为每个座位创建一个 PokerPlayer 实例

    # __________________________________________________ 需要子类覆盖的方法 ___________________________________________________
    # 这些方法定义了特定游戏的核心逻辑，必须由子类实现

    def get_hand_rank(self, hand_2d, board_2d):
        """
        计算给定手牌和公共牌的牌力等级。

        Args:
            hand_2d (np.ndarray):       要评估的手牌 (二维表示)
            board_2d (np.ndarray):      要评估的公共牌 (二维表示)

        Returns:
            int: 牌力等级，值越高越好。
        """
        raise NotImplementedError

    def get_hand_rank_all_hands_on_given_boards(self, boards_1d, lut_holder):
        """
        允许一次性计算多个给定公共牌上所有可能手牌的牌力等级。
        这通常利用预计算的查找表 (LUT) 来实现高效计算。

        Args:
            boards_1d (np.ndarray):     一维公共牌表示的数组
            lut_holder:                 与此环境类型关联的 LUT (查找表管理器)
        """
        raise NotImplementedError

    def _get_env_adjusted_action_formulation(self, action):
        """
        将子类特定的动作表示转换为 PokerEnv 的标准动作表示。
        标准表示是: Tuple(Discrete(3), Discrete(n_chips))
        或者说: (要执行的动作, 如果动作是下注则下注的总筹码量)。
        如果动作不是 Poker.BET_RAISE，则 n_chips_to_bet_if_action_is_bet 会被忽略。

        Args:
            action: 子类特定的动作表示 (例如，离散环境可能只有一个整数)

        Returns:
            Tuple:  (要执行的动作 (0:FOLD, 1:CHECK/CALL, 2:BET_RAISE), 如果是下注则下注的总筹码量)
        """
        # 基类默认不进行转换，直接返回
        return action

    def _adjust_raise(self, raise_total_amount_in_chips):
        """
        根据游戏类型（如限注、底池限注、无限注）调整玩家意图加注的总筹码量。
        例如，在限注游戏中，加注额是固定的；在底池限注中，加注额不能超过当前底池大小。

        Args:
            raise_total_amount_in_chips: 玩家意图加注到的总筹码量

        Returns:
            根据游戏规则调整后的实际加注总筹码量。
        """
        raise NotImplementedError

    @staticmethod
    def get_lut_holder():
        """ 返回一个特定于游戏规则的 lutholder 实例 """
        raise NotImplementedError

    # _____________________________________________________ 扑克核心逻辑 ______________________________________________________

    # --- 发牌方法 ---
    def _deal_hole_cards(self):
        """为每个玩家发底牌"""
        for player in self.seats:
            player.hand = self.deck.draw(self.N_HOLE_CARDS) # 从牌库抽取指定数量的牌

    def _deal_flop(self):
        """发翻牌圈公共牌"""
        self.board[:self.N_FLOP_CARDS] = self.deck.draw(self.N_FLOP_CARDS)

    def _deal_turn(self):
        """发转牌圈公共牌"""
        self.board[self.N_FLOP_CARDS:self.N_FLOP_CARDS + self.N_TURN_CARDS] = self.deck.draw(self.N_TURN_CARDS)

    def _deal_river(self):
        """发河牌圈公共牌"""
        d = self.N_FLOP_CARDS + self.N_TURN_CARDS # 计算河牌的起始索引
        self.board[d:d + self.N_RIVER_CARDS] = self.deck.draw(self.N_RIVER_CARDS)

    # --- 强制下注方法 ---
    def _post_antes(self):
        """收取底注"""
        for s in self.seats:
            s.bet_raise(self.ANTE) # 玩家下注底注金额
            s.has_acted_this_round = False # 收底注不算作主动行动

    def _post_small_blind(self):
        """收取小盲注"""
        player = self.seats[self.SB_POS] # 获取小盲注位置的玩家
        player.bet_raise(self.SMALL_BLIND) # 玩家下注小盲注金额
        player.has_acted_this_round = False # 收盲注不算作主动行动

    def _post_big_blind(self):
        """收取大盲注"""
        player = self.seats[self.BB_POS] # 获取大盲注位置的玩家
        player.bet_raise(self.BIG_BLIND) # 玩家下注大盲注金额
        player.has_acted_this_round = False # 收盲注不算作主动行动

    # --- 结算方法 ---
    def _payout_pots(self):
        """
        结算主池和所有边池。
        1. 计算所有未盖牌玩家的牌力等级。
        2. 按照边池等级从高到低（最后形成到最先形成），再处理主池。
        3. 对每个池子，找出有资格争夺该池且牌力最高的玩家（可能有多个赢家）。
        4. 将池子里的筹码均分给赢家，处理不能整除的零头筹码（随机分配或按规则分配）。
        """
        self._assign_hand_ranks_to_all_players() # 计算并存储每个玩家的牌力等级

        # --- 单挑情况 (N_SEATS == 2) ---
        if self.N_SEATS == 2:
            # 比较牌力
            if self.seats[0].hand_rank > self.seats[1].hand_rank:
                self.seats[0].award(self.main_pot) # 玩家0赢
            elif self.seats[0].hand_rank < self.seats[1].hand_rank:
                self.seats[1].award(self.main_pot) # 玩家1赢
            else: # 平局
                # 单挑时筹码总是偶数，因为双方投入相同
                self.seats[0].award(self.main_pot / 2)
                self.seats[1].award(self.main_pot / 2)
            # 清空主池
            self.main_pot = 0

        # --- 多人情况 ---
        else:
            # 将主池和边池组合起来处理
            pots = np.array([self.main_pot] + self.side_pots) # 主池放在索引0
            # pot_ranks: -1 代表主池，0 代表第一个边池，以此类推
            pot_ranks = np.arange(start=-1, stop=len(self.side_pots))
            # 将彩池金额和对应的级别配对
            pot_and_pot_ranks = np.array((pots, pot_ranks)).T

            # 从最高的边池级别开始结算 (这里代码是从低到高迭代，但逻辑上是处理每个池)
            # 注意：实际代码迭代顺序是从主池 (-1) 到最后一个边池。这似乎与注释意图相反，但逻辑正确。
            # 对于每个池子 (e[0] 是金额, e[1] 是级别 rank):
            for e in pot_and_pot_ranks:
                pot = e[0] # 当前处理的池底金额
                rank = e[1] # 当前处理的池底级别 (-1 为主池)

                # 找出有资格争夺此池且未盖牌的玩家
                # 玩家的 side_pot_rank >= 当前池的 rank 意味着他们至少投入了能构成这个池的赌注
                eligible_players = [p for p in self.seats if p.side_pot_rank >= rank and not p.folded_this_episode]

                num_eligible = len(eligible_players)
                if num_eligible > 0: # 如果有人有资格赢这个池

                    # 找出这些有资格的玩家中牌力最大的赢家 (可能不止一个)
                    winner_list = self._get_winner_list(players_to_consider=eligible_players) # 返回 PokerPlayer 对象列表
                    num_winners = int(len(winner_list)) # 赢家数量

                    # 计算每个赢家分得的筹码 (向下取整)
                    chips_per_winner = int(pot / num_winners)
                    # 计算不能整除的剩余筹码数量
                    num_non_div_chips = int(pot) % num_winners

                    # 分发整数部分
                    for p in winner_list:
                        p.award(chips_per_winner)

                    # 随机分发剩余的零头筹码
                    shuffled_winner_idxs = np.arange(num_winners)
                    np.random.shuffle(shuffled_winner_idxs) # 打乱赢家索引
                    # 只给前 num_non_div_chips 个随机选中的赢家每人发1个筹码
                    for p_idx in shuffled_winner_idxs[:num_non_div_chips]:
                        # 注意：这里用 p_idx 直接索引 self.seats 可能不准确，应该用 winner_list[p_idx]
                        # self.seats[p_idx].award(1) # 原代码疑似 Bug
                        winner_list[p_idx].award(1) # 修正：应该给赢家列表中的玩家发奖

            # 所有池子结算完毕后清空
            self.side_pots = [0] * self.N_SEATS
            self.main_pot = 0

    def _pay_all_to_one_player(self, player_to_pay_to):
        """
        将所有桌上的钱（当前下注、主池、边池）都支付给指定的玩家。
        这通常在只有一个玩家未盖牌的情况下发生。
        忽略边池级别 (IGNORES SIDEPOT RANKS)。

        Args:
            player_to_pay_to (PokerPlayer): 接收所有钱的玩家。
        """
        # 将每个座位的当前下注额支付给获胜玩家
        for seat in self.seats:
            player_to_pay_to.award(seat.current_bet)
            seat.current_bet = 0 # 清空原玩家的当前下注

        # 将所有边池支付给获胜玩家
        player_to_pay_to.award(sum(self.side_pots))
        self.side_pots = [0] * self.N_SEATS # 清空边池

        # 将主池支付给获胜玩家
        player_to_pay_to.award(self.main_pot)
        self.main_pot = 0 # 清空主池

    def _assign_hand_ranks_to_all_players(self):
        """为所有（未盖牌的）玩家计算并存储他们的牌力等级"""
        for player in self.seats:
            # 如果玩家已盖牌，其 hand_rank 可能无意义或为特定值（如 -1 或 None）
            # 这里没有显式检查 folded_this_episode，但 get_hand_rank 应该能处理无效手牌
            player.hand_rank = self.get_hand_rank(hand_2d=player.hand, board_2d=self.board)

    def _put_current_bets_into_main_pot_and_side_pots(self):
        """
        核心彩池管理逻辑：将在当前下注轮中玩家投入的筹码整理进主池和边池。
        """
        # --- 单挑情况 ---
        if self.N_SEATS == 2:
            # 计算两人当前下注的差额
            dif_p0_to_p1 = self.seats[0].current_bet - self.seats[1].current_bet

            # 如果玩家0下注更多，将差额退还给他
            if dif_p0_to_p1 > 0:
                self.seats[0].refund_from_bet(dif_p0_to_p1)
            # 如果玩家1下注更多，将差额退还给他
            elif dif_p0_to_p1 < 0:
                self.seats[1].refund_from_bet(-dif_p0_to_p1)

            # 此时两人下注额相等，将他们的下注额加入主池
            self.main_pot += self.seats[0].current_bet
            self.main_pot += self.seats[1].current_bet
            # 清空玩家的当前下注额
            self.seats[0].current_bet = 0
            self.seats[1].current_bet = 0

        # --- 多人情况 ---
        else:
            # 1. 退还未被跟注的下注：
            #    找出当前下注最多的玩家和第二多的玩家
            _players_sorted_by_bet_in_front = sorted(self.seats, key=lambda x: x.current_bet, reverse=True)
            # 计算最大下注和第二大下注之间的差额
            # 注意：如果所有下注都相等，dif=0；如果只有一个玩家下注，这里可能有问题，但 PokerPlayer.refund_from_bet(0) 无害
            dif = _players_sorted_by_bet_in_front[0].current_bet - _players_sorted_by_bet_in_front[1].current_bet
            # 将差额退还给下注最多的玩家
            _players_sorted_by_bet_in_front[0].refund_from_bet(dif)

            # 此时，所有参与到最后的玩家的 current_bet 应该是一样的（或者有些玩家因 all-in 或 fold 而更少）

            # 2. 填充主池：
            #    找出所有未盖牌的玩家
            players_not_folded = [p for p in self.seats if not p.folded_this_episode]
            # 找到未盖牌玩家中的最小当前下注额。所有玩家对主池的贡献不能超过这个值。
            # 如果玩家都未 all-in，那么所有未盖牌玩家的 current_bet 都相等。
            # 如果有玩家 all-in，他的 current_bet 可能小于其他人。
            main_pot_max_amount = min([p.current_bet for p in players_not_folded]) if players_not_folded else 0

            # 将每个玩家对主池的贡献（不超过 main_pot_max_amount）加入主池
            for p in self.seats:
                amount_contributed = min(p.current_bet, main_pot_max_amount)
                self.main_pot += amount_contributed
                p.current_bet -= amount_contributed # 从玩家当前下注中减去对主池的贡献

            # 3. 填充边池（如果需要）：
            #    迭代处理剩余的 current_bet 来创建边池。

            # 辅助函数：找到当前剩余 > 0 的最小下注额的玩家索引
            def _find_next_smallest_bet():
                current_bets = [p.current_bet for p in self.seats]
                next_bet_idx = None
                min_bet = float('inf')

                for b_idx in range(self.N_SEATS):
                    # 只考虑当前下注 > 0 且未盖牌的玩家
                    # （已盖牌玩家的 current_bet 理论上也应为 0 或已处理，但检查一下更安全）
                    # if current_bets[b_idx] > 0 and not self.seats[b_idx].folded_this_episode: # 原代码的注释逻辑
                    if current_bets[b_idx] > 0: # 简化：只要 current_bet > 0 就考虑
                        if current_bets[b_idx] < min_bet:
                             min_bet = current_bets[b_idx]
                             next_bet_idx = b_idx
                        # 原代码的逻辑是找到第一个小于当前最小值的，可能不总是对
                        # if ((next_bet_idx is None or current_bets[b_idx] < current_bets[next_bet_idx])
                        #     and not self.seats[b_idx].folded_this_episode): # 这里检查 folded 可能多余
                        #     next_bet_idx = b_idx

                return next_bet_idx

            idx_smallest_bet = _find_next_smallest_bet() # 获取填充主池后的状态

            # 循环创建边池，直到所有玩家的 current_bet 都为 0
            while idx_smallest_bet is not None:
                # 确定当前要创建的边池的索引 (级别)
                # 玩家的 side_pot_rank 初始为 -1，表示主池。第一个边池 rank 为 0，以此类推。
                current_max_side_pot_rank = max([p.side_pot_rank for p in self.seats])
                side_pot_idx = current_max_side_pot_rank + 1

                # 这个边池每人贡献的金额 = 当前剩余最小下注额
                side_pot_amount_per_player_in_it = self.seats[idx_smallest_bet].current_bet

                # 找出所有能参与到这个及之后边池的玩家
                # （即，他们的剩余 current_bet >= 这个边池的贡献额，或者他们 all-in 了且 current_bet > 0）
                # 原代码逻辑有点绕，简化理解：所有 current_bet > 0 的玩家都需要处理
                # players_not_all_in_after_this_cleanup = [p for p in self.seats if not (
                #     p.current_bet < side_pot_amount_per_player_in_it and p.is_allin)] # 原代码

                # 为所有 current_bet > 0 的玩家更新他们的 side_pot_rank，表示他们参与到了这个新边池
                for p in self.seats:
                    if p.current_bet > 0:
                        p.side_pot_rank = side_pot_idx

                # 从每个 current_bet > 0 的玩家那里收取对这个边池的贡献
                for p in self.seats:
                    amount_contributed = min(p.current_bet, side_pot_amount_per_player_in_it)
                    if amount_contributed > 0: # 确保只处理有贡献的
                        self.side_pots[side_pot_idx] += amount_contributed # 加入对应的边池
                        p.current_bet -= amount_contributed # 从玩家当前下注中减去

                # 寻找下一个最小的剩余 current_bet，准备创建下一个边池
                idx_smallest_bet = _find_next_smallest_bet()

    # --- 游戏进程控制方法 ---
    def _rundown(self):
        """
        执行 "Rundown"：当牌局结果已确定（例如，只剩下一个未盖牌的玩家，或者其他玩家都全下），
        但还需要发出剩余的公共牌来决定最终胜负时调用。
        它会发出所有剩余的公共牌，然后进行结算。
        """
        while True:
            # 进入理论上的下一轮
            self.current_round += 1

            # 如果已经发完了所有牌 (超过了最后一轮)
            if self.current_round > self.ALL_ROUNDS_LIST[-1]: # 使用 > 更清晰
                # 先将最后一轮的下注（如果有的话，理论上 rundown 时没有下注）结算进彩池
                self._put_current_bets_into_main_pot_and_side_pots()
                # 将轮次设置回最后一轮 (e.g., RIVER)
                self.current_round -= 1

                # 如果需要，保存结算前的状态
                if self.RETURN_PRE_TRANSITION_STATE_IN_INFO:
                    state_before_transition = self.state_dict()

                # 进行最终结算
                self._payout_pots()

                # 返回（如果需要的话，返回结算前的状态）
                if self.RETURN_PRE_TRANSITION_STATE_IN_INFO:
                    return state_before_transition
                return # Rundown 结束

            # 根据进入的理论轮次发牌
            elif self.current_round == Poker.FLOP:
                self._deal_flop()
            elif self.current_round == Poker.TURN:
                self._deal_turn()
            elif self.current_round == Poker.RIVER:
                self._deal_river()
            else: # 不应该有其他轮次
                raise ValueError(f"无效的轮次 during rundown: {self.current_round}")

    def _deal_next_round(self):
        """根据当前的 self.current_round 发相应轮次的牌。"""
        if self.current_round == Poker.PREFLOP:
            self._deal_hole_cards()
        elif self.current_round == Poker.FLOP:
            self._deal_flop()
        elif self.current_round == Poker.TURN:
            self._deal_turn()
        elif self.current_round == Poker.RIVER:
            self._deal_river()
        else:
            # 可能是游戏结束后的状态，或者是不应该发牌的轮次
            # raise ValueError(f"尝试在无效轮次发牌: {self.current_round}")
            pass # 或者静默处理

    def _next_round(self):
        """
        结束当前下注轮，进入下一个下注轮。
        """
        # 重置限注游戏相关的计数器
        if self.IS_FIXED_LIMIT_GAME:
            self.n_raises_this_round = 0

        # 重置 capped raise 状态
        self.capped_raise.reset()

        # 将当前轮次的下注结算进主池和边池
        self._put_current_bets_into_main_pot_and_side_pots()

        # 确定下一轮第一个行动的玩家 (必须在 current_round += 1 之前调用)
        # 注意: _get_first_to_act_post_flop 可能需要区分是 preflop 到 flop 还是其他轮次
        # 这里假设 _get_first_to_act_post_flop 能处理所有 post-flop 情况
        if self.current_round == Poker.PREFLOP: # 如果当前是翻牌前结束
             self.current_player = self._get_first_to_act_post_flop()
        else: # 如果是其他轮次结束
             # 逻辑可能相同，或者需要不同的函数？这里沿用原代码
             self.current_player = self._get_first_to_act_post_flop()


        # 重置所有玩家的本轮行动状态
        for p in self.seats:
            p.has_acted_this_round = False

        # 进入下一轮
        self.current_round += 1
        # 发下一轮的牌
        self._deal_next_round()

    def _step(self, processed_action):
        """
        核心步骤函数：处理一个标准化的动作，推进游戏状态。

        Args:
            processed_action (tuple or list): 标准化的动作 (action_idx, raise_size)。
                                              假定此动作是针对 self.current_player 的。
                                              raise_size 仅在 action_idx == Poker.BET_RAISE 时有意义。

        Returns:
            tuple: (obs, rew_for_all_players, done?, info)
                   观察, 所有玩家的奖励, 是否结束?, 附加信息
        """

        # 1. 修正动作合法性 (例如，全下、最小/最大加注额限制等)
        #    此调用后，processed_action 被认为是完全合法的。
        processed_action = self._get_fixed_action(action=processed_action)

        # 2. 执行修正后的动作
        action_idx = processed_action[0]
        raise_or_call_amount = processed_action[1]

        if action_idx == Poker.CHECK_CALL:
            # 玩家执行 看牌/跟注
            self.current_player.check_call(total_to_call=raise_or_call_amount) # total_to_call 是修正后的应下注总额
        elif action_idx == Poker.FOLD:
            # 玩家执行 弃牌
            self.current_player.fold()
        elif action_idx == Poker.BET_RAISE:
            # 玩家执行 下注/加注

            # --- 处理 CappedRaise 规则 ---
            # 如果本次加注的总额小于当前的最小加注总额（通常发生在玩家 all-in 但筹码不足时）
            if raise_or_call_amount < self._get_current_total_min_raise():
                self.capped_raise.happened_this_round = True            # 标记本轮发生了 capped raise
                self.capped_raise.player_that_raised = self.current_player # 记录进行不足额加注的玩家
                self.capped_raise.player_that_cant_reopen = self.last_raiser # 记录之前加注的玩家（他可能被限制再加注）
            # 如果之前已经发生过 capped raise
            elif self.capped_raise.happened_this_round is True:
                # 如果当前加注的玩家不是那个被限制再加注的玩家
                # （意味着有第三方玩家进行了合法的再加注，打破了限制）
                if self.capped_raise.player_that_cant_reopen is not self.current_player:
                    self.capped_raise.reset() # 重置 capped raise 状态

            # --- 更新状态 ---
            # 记录最后一个进行加注的玩家
            self.last_raiser = self.current_player
            # 执行玩家的下注/加注动作
            self.current_player.bet_raise(total_bet_amount=raise_or_call_amount)
            # 增加本局游戏的总动作数
            self.n_actions_this_episode += 1

            # 如果是限注游戏，增加本轮的加注次数
            if self.IS_FIXED_LIMIT_GAME:
                self.n_raises_this_round += 1
        else:
            # 不应该出现其他动作类型
            raise RuntimeError(f"{action_idx} 不是合法的动作索引")

        # 记录最后执行的动作 [类型, 金额, 玩家ID]
        self.last_action = [action_idx, raise_or_call_amount, self.current_player.seat_id]

        # ______________________________________________________________________________________________________________
        # 3. 检查游戏是否应该进入下一轮、进行 Rundown，或者继续当前轮

        # 获取当前仍在游戏中（未盖牌且未全下）的玩家列表
        all_non_all_in_and_non_fold_p = [p for p in self.seats if not p.folded_this_episode and not p.is_allin]
        # 获取当前未盖牌的玩家列表
        all_nonfold_p = [p for p in self.seats if not p.folded_this_episode]

        info = None # 初始化 info 字典
        is_terminal = False # 初始化是否结束标志

        # --- 情况 A: 当前下注轮继续 ---
        # 调用辅助函数判断是否需要继续本轮下注
        if self._should_continue_in_this_round(all_non_all_in_and_non_fold_p=all_non_all_in_and_non_fold_p,
                                               all_nonfold_p=all_nonfold_p):
            # 确定下一个行动的玩家
            self.current_player = self._get_player_that_has_to_act_next()
            is_terminal = False
            # 如果需要，设置 info
            if self.RETURN_PRE_TRANSITION_STATE_IN_INFO:
                # chance_acts=False 因为是玩家行动，不是发牌
                # state_dict_before_money_move=None 因为钱还没移动到彩池
                info = {"chance_acts": False, "state_dict_before_money_move": None}

        # --- 情况 B: 当前下注轮结束，进入下一轮或结束游戏 ---
        # 如果还有多于一个玩家未盖牌且未全下（意味着游戏可以继续到下一轮）
        elif len(all_non_all_in_and_non_fold_p) > 1:
            # 如果当前已经是最后一轮 (e.g., RIVER)
            if self.current_round == self.ALL_ROUNDS_LIST[-1]:
                # 游戏结束，进行结算
                is_terminal = True
                # 将当前赌注移入彩池
                self._put_current_bets_into_main_pot_and_side_pots()
                # 如果需要，保存结算前的状态
                if self.RETURN_PRE_TRANSITION_STATE_IN_INFO:
                    info = {"chance_acts": False, "state_dict_before_money_move": self.state_dict()}
                # 结算彩池
                self._payout_pots()
            # 如果还不是最后一轮
            else:
                # 进入下一轮
                is_terminal = False
                # 如果需要，保存进入下一轮前的状态
                if self.RETURN_PRE_TRANSITION_STATE_IN_INFO:
                    # chance_acts=True 因为下一轮开始会发牌
                    info = {"chance_acts": True, "state_dict_before_money_move": self.state_dict()}
                # 执行进入下一轮的操作（发牌、确定首个行动玩家等）
                self._next_round()

        # --- 情况 C: 需要进行 Rundown ---
        # 如果当前轮结束，但有玩家全下导致无法继续下注，
        # 且仍有多于一个玩家未盖牌（意味着需要比牌）
        elif len(all_nonfold_p) > 1:
            # 游戏结束，进行 Rundown
            is_terminal = True
            # 执行 Rundown (发完剩余公共牌并结算)
            state_before_payouts = self._rundown() # _rundown 内部会调用 _payout_pots
            # 如果需要，保存结算前的状态 (由 _rundown 返回)
            if self.RETURN_PRE_TRANSITION_STATE_IN_INFO:
                 info = {"chance_acts": False, "state_dict_before_money_move": state_before_payouts}

        # --- 情况 D: 只剩一个玩家未盖牌 ---
        elif len(all_nonfold_p) == 1:
            # 游戏结束，该玩家赢得所有彩池
            is_terminal = True
            # 如果需要，先结算当前赌注到彩池再保存状态
            if self.RETURN_PRE_TRANSITION_STATE_IN_INFO:
                self._put_current_bets_into_main_pot_and_side_pots()
                info = {"chance_acts": False, "state_dict_before_money_move": self.state_dict()}
                # 注意：这里调用 _payout_pots 可能不是最高效的，但逻辑上可行
                self._payout_pots()
            else: # 更高效的方式：直接支付
                self._pay_all_to_one_player(all_nonfold_p[0])
        # --- 异常情况 ---
        else:
            # 理论上不应该到达这里 (例如，0 个未盖牌玩家?)
            raise RuntimeError("游戏状态出现未处理的边缘情况")

        # 4. 返回当前步骤的结果
        return self._get_current_step_returns(is_terminal=is_terminal, info=info)

    # _____________________________________________________ 工具方法 ______________________________________________________

    def _get_winner_list(self, players_to_consider):
        """
        从给定的玩家列表中找出牌力最高的玩家（赢家）。

        Args:
            players_to_consider (list): 需要比较牌力的 PokerPlayer 对象列表。

        Returns:
            list: 包含所有牌力最高（并列）的 PokerPlayer 实例的列表。
        """
        # 如果列表为空，返回空列表
        if not players_to_consider:
            return []
        # 找到最高的牌力等级
        best_rank = max([p.hand_rank for p in players_to_consider if p.hand_rank is not None]) # 过滤 None
        # 找出所有牌力等级等于最高等级的玩家
        winners = [p for p in players_to_consider if p.hand_rank == best_rank]
        return winners

    def _get_current_total_min_raise(self):
        """
        计算当前状态下，进行一次最小加注需要加注到的总金额。
        规则：最小加注额等于上一个加注额和当前跟注额之间的差额，但至少为一个大盲注。
              加注到的总金额 = 当前需要跟注的总额 + 最小加注额。
        """
        # --- 单挑情况 ---
        if self.N_SEATS == 2:
            # 获取两人的当前下注额，并排序
            _sorted_ascending = sorted([p.current_bet for p in self.seats]) # [小, 大]
            # 计算上一个加注的增量（差额），最小为一个大盲注
            delta = max(_sorted_ascending[1] - _sorted_ascending[0], self.BIG_BLIND)
            # 最小加注后的总额 = 当前最大下注额 + 最小加注增量
            return _sorted_ascending[1] + delta

        # --- 多人情况 ---
        else:
            # 获取所有玩家的当前下注额，并降序排序
            current_bets_sorted_descending = sorted([p.current_bet for p in self.seats], reverse=True)
            # 当前需要跟注的总额 = 最大的下注额
            current_to_call_total = current_bets_sorted_descending[0]
            _largest_bet = current_bets_sorted_descending[0]

            # 找到第一个与最大下注额不同的下注额（即上上一个下注额）
            for i in range(1, self.N_SEATS):
                if current_bets_sorted_descending[i] == _largest_bet:
                    continue # 跳过与最大下注额相同的

                # 找到了上上一个下注额
                # 计算上一个加注的增量
                delta_between_last_and_before_last = _largest_bet - current_bets_sorted_descending[i]
                # 最小加注增量 = max(上一个加注增量, 大盲注)
                delta = max(delta_between_last_and_before_last, self.BIG_BLIND)
                # 最小加注后的总额 = 当前需要跟注的总额 + 最小加注增量
                return current_to_call_total + delta

            # 如果所有未盖牌玩家的下注额都相同（例如，翻牌前大盲注后的情况）
            # 则最小加注增量为一个大盲注
            return current_to_call_total + self.BIG_BLIND

    def _get_new_board(self):
        """创建一个新的、空的公共牌数组"""
        # 使用 Poker.CARD_NOT_DEALT_TOKEN_1D (-1) 填充
        return np.full((self.N_TOTAL_BOARD_CARDS, 2), Poker.CARD_NOT_DEALT_TOKEN_1D, dtype=np.int8)

    def _get_first_to_act_pre_flop(self):
        """确定翻牌前第一个行动的玩家"""
        if self.N_SEATS >= 3: # 3人及以上游戏，通常是大盲注左边的玩家 (索引可能是 3，取决于座位的定义)
                             # 原代码用了 >= 4，但 3 人桌逻辑也应不同于单挑
                             # 假设座位索引: 0=BTN, 1=SB, 2=BB, 3=UTG...
            if self.N_SEATS == 3:
                 return self.seats[0] # 3人桌，BTN 先行动
            else: # 4人及以上
                 return self.seats[3 % self.N_SEATS] # 大盲注左边 (UTG)
        else: # N_SEATS == 2 (单挑)
            # 按钮位 (SB) 先行动
            return self.seats[0]

    def _get_first_to_act_post_flop(self):
        """
        确定翻牌后第一个行动的玩家。
        规则：通常是仍在牌局中、位置最靠前（座位号最小，按钮位除外）的玩家。
              按钮位 (索引 0) 通常最后行动。
              单挑时可能有特殊规则 (BTN_IS_FIRST_POSTFLOP)。
        """
        # --- 单挑情况 ---
        if self.N_SEATS == 2:
            if self.BTN_IS_FIRST_POSTFLOP: # 如果按钮位先动
                return self.seats[0]
            else: # 否则大盲注位先动
                return self.seats[1]
        # --- 多人情况 ---
        else:
            # 找出所有未盖牌且未全下的玩家
            players_to_consider = [p for p in self.seats if not p.folded_this_episode and not p.is_allin]

            # 如果没有这样的玩家（例如都 all-in 了），理论上不应该调用此函数
            if not players_to_consider:
                 # 返回一个默认值或引发错误，取决于调用上下文
                 # 这里假设至少有一个玩家符合条件
                 # return self.seats[0] # 或者返回当前玩家？
                 raise RuntimeError("无法确定翻牌后第一个行动者，没有符合条件的玩家")


            # 找到座位号最小的玩家（按钮位 0 被视为最大）
            first_p = players_to_consider[0]
            for p in players_to_consider[1:]: # 从第二个开始比较
                # 如果 p 的座位号更小 (且不是按钮位 0)，或者当前的 first_p 是按钮位 0
                if (p.seat_id != 0 and (p.seat_id < first_p.seat_id or first_p.seat_id == 0)):
                    first_p = p
                # 如果 p 不是按钮位，但 first_p 是按钮位，那么 p 更靠前
                # elif first_p.seat_id == 0 and p.seat_id != 0:
                #      first_p = p

            return first_p

    def _get_biggest_bet_out_there_aka_total_to_call(self):
        """获取当前桌面上最大的下注额，即玩家需要跟注到的总额"""
        # 如果没有玩家（不应该发生），返回 0
        if not self.seats:
            return 0
        return max([p.current_bet for p in self.seats])

    def _get_player_that_has_to_act_next(self):
        """确定当前玩家行动之后，下一个应该行动的玩家"""
        # 从当前玩家的下一个座位开始查找
        idx = (self.seats.index(self.current_player) + 1) % self.N_SEATS # 使用模运算处理循环

        # 循环查找 N_SEATS 次（最多一圈）
        for i in range(self.N_SEATS):
            current_check_idx = (idx + i) % self.N_SEATS
            p = self.seats[current_check_idx]

            # 如果找到一个未全下且未盖牌的玩家，他就是下一个行动者
            if not p.is_allin and not p.folded_this_episode:
                return p

        # 如果循环一圈都没有找到（例如，所有其他玩家都 all-in 或 folded）
        # 这通常意味着一轮结束，或者游戏状态有问题
        raise RuntimeError("找不到下一个行动的玩家。游戏逻辑可能存在问题。")

    def _get_fixed_action(self, action):
        """
        核心方法：验证并修正玩家意图执行的动作，确保其符合当前游戏状态和规则。

        Args:
            action (iterable): 玩家意图执行的动作 [action_idx, raise_amount_in_chips]。

        Returns:
            list: 修正后的合法动作 [action_idx, amount]。
                  对于 FOLD，amount 为 -1。
                  对于 CHECK/CALL，amount 是玩家需要下注到的总额。
                  对于 BET_RAISE，amount 是玩家加注到的总额。
        """
        _action_idx = action[0] # 动作类型 (FOLD, CHECK_CALL, BET_RAISE)
        # 意图加注的金额（仅在 BET_RAISE 时相关，但可能未提供或无效）
        intended_raise_total_amount = action[1] if len(action) > 1 else 0

        # 获取当前需要跟注的总额
        total_to_call = self._get_biggest_bet_out_there_aka_total_to_call()

        # --- 处理 FOLD ---
        if _action_idx == Poker.FOLD:
            # 如果当前无需跟注 (total_to_call <= 当前下注额)，则强制修正为 CHECK
            # 因为弃牌在无需跟注时是非法的（或至少是等效于看牌）
            if total_to_call <= self.current_player.current_bet:
                # 调用 _process_check_call 获取修正后的 CHECK 动作
                return self._process_check_call(total_to_call=total_to_call)
            else:
                # 合法的弃牌
                return [Poker.FOLD, -1] # 返回标准 FOLD 表示

        # --- 处理 CHECK/CALL ---
        elif _action_idx == Poker.CHECK_CALL:
            # 特殊规则：翻牌前第一手不允许跟注（可能用于某些变种）
            if (self.FIRST_ACTION_NO_CALL
                and (self.n_actions_this_episode == 0) # 是本局的第一个动作
                and self.current_round == Poker.PREFLOP): # 且在翻牌前
                # 强制修正为 FOLD
                return [Poker.FOLD, -1]

            # 否则，处理为标准的 CHECK 或 CALL
            return self._process_check_call(total_to_call=total_to_call)

        # --- 处理 BET/RAISE ---
        elif _action_idx == Poker.BET_RAISE:
            # 1. 检查限注游戏规则：是否已达到本轮最大加注次数？
            if self.IS_FIXED_LIMIT_GAME:
                if self.n_raises_this_round >= self.MAX_N_RAISES_PER_ROUND[self.current_round]:
                    # 已达上限，强制修正为 CALL
                    return self._process_check_call(total_to_call=total_to_call)

            # 2. 检查是否能加注：
            #    a) 玩家是否已经 all-in (无法再加注)?
            #       (检查 stack + current_bet <= total_to_call，即跟注后就 all-in 了)
            #    b) 玩家是否因为 CappedRaise 规则被禁止再加注？
            if ((self.current_player.stack + self.current_player.current_bet <= total_to_call)
                or (self.capped_raise.player_that_cant_reopen is self.current_player)):
                # 不能加注，强制修正为 CALL
                return self._process_check_call(total_to_call=total_to_call)
            else:
                # 可以加注，处理加注逻辑
                return self._process_raise(raise_total_amount_in_chips=intended_raise_total_amount)
        else:
            # 无效的动作索引
            raise RuntimeError(f'无效的动作索引 ({_action_idx})，必须是 FOLD (0), CHECK/CALL (1), 或 BET/RAISE (2)')

    def _process_check_call(self, total_to_call):
        """处理 CHECK 或 CALL 动作，返回修正后的动作和金额"""
        # 计算实际需要跟注的差额，不能超过玩家剩余筹码
        delta_to_call = min(total_to_call - self.current_player.current_bet, self.current_player.stack)
        # 计算跟注/看牌后，玩家在桌上的总下注额
        total_bet_to_be_placed = int(delta_to_call + self.current_player.current_bet)
        # 返回修正后的动作 [动作类型, 总下注额]
        return [Poker.CHECK_CALL, total_bet_to_be_placed]

    def _process_raise(self, raise_total_amount_in_chips):
        """处理 BET 或 RAISE 动作，返回修正后的动作和金额"""
        # 1. 根据游戏类型调整意图加注的金额 (限注、底池限注等规则)
        #    _adjust_raise 是子类需要实现的方法
        raise_to = self._adjust_raise(raise_total_amount_in_chips=raise_total_amount_in_chips)

        # 2. 检查调整后的金额是否超过玩家总筹码 (当前下注 + 剩余筹码)
        if self.current_player.current_bet + self.current_player.stack < raise_to:
            # 如果超过，则修正为 all-in
            raise_to = self.current_player.stack + self.current_player.current_bet

        # 返回修正后的动作 [动作类型, 加注到的总额]
        return [Poker.BET_RAISE, int(raise_to)]

    def _should_continue_in_this_round(self, all_non_all_in_and_non_fold_p, all_nonfold_p):
        """
        判断当前下注轮是否应该继续。
        如果满足以下任一条件，则轮次结束 (返回 False)：
        1. 只剩下一个或零个未盖牌的玩家。
        2. 所有未盖牌的玩家要么已经 all-in，要么其当前下注额等于最大下注额，
           并且所有未 all-in 且未盖牌的玩家在本轮都已经行动过至少一次。

        Args:
            all_non_all_in_and_non_fold_p (list): 未盖牌且未全下的玩家列表。
            all_nonfold_p (list): 未盖牌的玩家列表。

        Returns:
            bool: True 表示本轮应继续，False 表示本轮结束。
        """

        # 条件 1: 只剩下一个或零个未盖牌玩家
        if len(all_nonfold_p) < 2:
            return False # 轮次结束

        # 条件 2:
        largest_bet = max([p.current_bet for p in self.seats]) # 获取当前最大下注额
        # 检查是否所有未盖牌玩家都已匹配最大下注或 all-in
        all_matched_or_allin = len([p for p in all_nonfold_p if p.is_allin or p.current_bet == largest_bet]) == len(all_nonfold_p)
        # 检查是否所有有能力行动的玩家（未盖牌未 all-in）都已经行动过
        all_active_acted = len([p for p in all_non_all_in_and_non_fold_p if not p.has_acted_this_round]) == 0

        # 如果上述两个子条件都满足，则轮次结束
        if all_matched_or_allin and all_active_acted:
            return False # 轮次结束

        # 否则，轮次继续
        return True

    # _____________________________________________________ 输出处理 _____________________________________________________

    def _get_current_step_returns(self, is_terminal, info):
        """获取当前步骤的返回值 (obs, reward, done, info)"""
        obs = self.get_current_obs(is_terminal) # 获取当前观察
        reward = self._get_step_reward(is_terminal) # 获取当前奖励
        return obs, reward, is_terminal, info # done? 即 is_terminal

    def _get_player_states_all_players(self, normalization_sum):
        """ 获取所有玩家的公共状态信息，用于构建观察向量 """
        player_states = []
        # --- 单挑简化观察 ---
        if (self.N_SEATS == 2) and self._USE_SIMPLE_HU_OBS:
            for player in self.seats:
                player_states += [
                    player.stack / normalization_sum,      # 标准化后的筹码
                    player.current_bet / normalization_sum, # 标准化后的当前下注
                    float(player.is_allin)                  # 是否 all-in (转为 float)
                ]
        # --- 标准观察 ---
        else:
            for player in self.seats:
                player_states += [
                    player.stack / normalization_sum,       # 标准化筹码
                    player.current_bet / normalization_sum,  # 标准化当前下注
                    float(player.folded_this_episode),       # 是否已盖牌
                    float(player.is_allin)                   # 是否 all-in
                ]
                # 边池级别 (one-hot)
                x = [0.0] * self.N_SEATS
                # side_pot_rank > -1 表示参与了至少一个边池
                # 注意: side_pot_rank 的范围是从 -1 (主池) 到 N_SEATS-1
                # 这里直接用 rank 做索引可能不对，应该 one-hot 编码级别
                # 原代码逻辑：如果 rank > 0，则在对应索引处设为 1
                # 修正/澄清：如果 player.side_pot_rank >= 0，应该在索引 player.side_pot_rank 处设为 1？
                # 假设原始意图是 one-hot 编码边池级别 (0 到 N_SEATS-1)
                if player.side_pot_rank >= 0:
                     if player.side_pot_rank < self.N_SEATS: # 确保索引有效
                         x[int(player.side_pot_rank)] = 1.0
                player_states += x

        return player_states

    def _get_board_state(self, ):
        """ 获取公共牌的状态信息，用于构建观察向量 (通常是 one-hot 编码) """
        # 每张牌需要 N_RANKS (点数) + N_SUITS (花色) 个元素来 one-hot 编码
        K = (self.N_RANKS + self.N_SUITS)
        # 初始化一个足够大的零向量
        _board_space = [0.0] * (self.N_TOTAL_BOARD_CARDS * K)
        # 遍历当前已发出的公共牌
        for i, card in enumerate(self.board.tolist()):
            # 如果遇到未发牌标记，则停止
            if card[0] == Poker.CARD_NOT_DEALT_TOKEN_1D:
                break
            # 计算这张牌在向量中的起始索引
            D = K * i
            # 将点数对应的位置设为 1
            _board_space[card[0] + D] = 1.0

            # 如果游戏规则关心花色
            if self.SUITS_MATTER:
                # 将花色对应的位置设为 1 (索引 = 点数索引 + N_RANKS)
                _board_space[card[1] + D + self.N_RANKS] = 1.0

        return _board_space

    def _get_table_state(self, normalization_sum):
        """ 获取牌桌的公共状态信息，用于构建观察向量 """
        # --- 单挑简化观察 ---
        if (self.N_SEATS == 2) and self._USE_SIMPLE_HU_OBS:
            community_state = [
                self.ANTE / normalization_sum,                  # 标准化底注
                self.SMALL_BLIND / normalization_sum,           # 标准化小盲
                self.BIG_BLIND / normalization_sum,             # 标准化大盲
                self._get_current_total_min_raise() / normalization_sum, # 标准化最小加注总额
                self.main_pot / normalization_sum,              # 标准化主池
                self._get_biggest_bet_out_there_aka_total_to_call() / normalization_sum, # 标准化跟注总额
                # 标准化最后动作金额 (如果上个动作是下注/加注)
                (self.last_action[1] / normalization_sum) if self.last_action is not None and self.last_action[0] == Poker.BET_RAISE else 0.0,
                 # (原代码未检查动作类型，可能不准确) 修正: 仅在 BET_RAISE 时记录金额
                 # (self.last_action[1] / normalization_sum) if self.last_action[0] is not None else 0, # 原代码
            ]

            # 最后动作类型 (one-hot)
            x_what = [0.0] * 3
            # 最后动作执行者 (one-hot)
            x_who = [0.0] * self.N_SEATS
            if self.last_action is not None and self.last_action[0] is not None: # 检查 last_action 和 动作类型 是否有效
                x_what[self.last_action[0]] = 1.0
                if self.last_action[2] is not None and 0 <= self.last_action[2] < self.N_SEATS: # 检查玩家索引有效性
                     x_who[self.last_action[2]] = 1.0
            community_state += x_what + x_who

            # 当前行动者 (one-hot)
            x = [0.0] * self.N_SEATS
            if self.current_player is not None: # 检查当前玩家是否存在
                 x[self.current_player.seat_id] = 1.0
            community_state += x

            # 当前轮次 (one-hot)
            x = [0.0] * (self.ALL_ROUNDS_LIST[-1] + 1)
            if self.current_round is not None: # 检查当前轮次是否存在
                 x[self.current_round] = 1.0
            community_state += x

        # --- 标准观察 ---
        else:
            community_state = [
                self.ANTE / normalization_sum,                  # 标准化底注
                self.SMALL_BLIND / normalization_sum,           # 标准化小盲
                self.BIG_BLIND / normalization_sum,             # 标准化大盲
                self._get_current_total_min_raise() / normalization_sum, # 标准化最小加注总额
                self.main_pot / normalization_sum,              # 标准化主池
                self._get_biggest_bet_out_there_aka_total_to_call() / normalization_sum, # 标准化跟注总额
                # 标准化最后动作金额 (修正同上)
                (self.last_action[1] / normalization_sum) if self.last_action is not None and self.last_action[0] == Poker.BET_RAISE else 0.0,
            ]

             # 最后动作类型 (one-hot)
            x_what = [0.0] * 3
            # 最后动作执行者 (one-hot)
            x_who = [0.0] * self.N_SEATS
            if self.last_action is not None and self.last_action[0] is not None:
                x_what[self.last_action[0]] = 1.0
                if self.last_action[2] is not None and 0 <= self.last_action[2] < self.N_SEATS:
                     x_who[self.last_action[2]] = 1.0
            community_state += x_what + x_who

            # 当前行动者 (one-hot)
            x = [0.0] * self.N_SEATS
            if self.current_player is not None:
                 x[self.current_player.seat_id] = 1.0
            community_state += x

            # 当前轮次 (one-hot)
            x = [0.0] * (self.ALL_ROUNDS_LIST[-1] + 1)
            if self.current_round is not None:
                 x[self.current_round] = 1.0
            community_state += x

            # 边池大小 (标准化的)
            if self.N_SEATS > 2 and self.side_pots is not None:
                community_state += [(sp / normalization_sum) if normalization_sum > 0 else 0.0 for sp in self.side_pots]
            else: # 如果是单挑或 side_pots 未初始化
                community_state += [0.0] * self.N_SEATS # 填充零

        return community_state

    def _get_step_reward(self, is_terminal):
        """ 计算当前步骤的奖励 """
        # 如果游戏未结束，奖励为零
        if not is_terminal:
            return np.zeros(shape=self.N_SEATS, dtype=np.float32)
        # 如果游戏结束，奖励 = (结束时筹码 - 本局起始筹码) / 奖励缩放因子
        return np.array([(p.stack - p.starting_stack_this_episode) / self.REWARD_SCALAR
                         if self.REWARD_SCALAR != 0 else (p.stack - p.starting_stack_this_episode)
                         for p in self.seats], dtype=np.float32)


    # ______________________________________________________ 公开 API _______________________________________________________

    def reset(self, deck_state_dict=None):
        """
        重置游戏状态以开始新的一局。
        如果 env_args 中指定了，会应用起始筹码随机化。
        如果 deck_state_dict 不为 None，则从此字典同步牌库、手牌和公共牌状态，
        以确保多个环境实例在调用 step() 时产生相同的结果（用于并行计算等）。

        Args:
            deck_state_dict (dict, optional): 用于同步牌面随机性的状态字典。默认为 None。

        Returns:
            tuple: (obs, reward, done, info) 初始状态的返回值。
        """
        # --- 重置游戏内部状态 ---
        # 重置限注游戏计数器
        if self.IS_FIXED_LIMIT_GAME:
            # 如果有大盲注，大盲注算作第一次 "加注" (对于后续加注次数限制)
            # 如果是只有底注的游戏 (如 Leduc)，则不算
            self.n_raises_this_round = 1 if self.BIG_BLIND > 0 else 0

        # 重置牌桌状态
        self.side_pots = [0] * self.N_SEATS  # 清空边池
        self.main_pot = 0                    # 清空主池
        self.board = self._get_new_board()   # 重置公共牌
        self.last_action = [None, None, None] # 重置最后动作
        self.current_round = self.ALL_ROUNDS_LIST[0] # 设置为第一轮 (通常是 PREFLOP)
        self.capped_raise.reset()            # 重置 CappedRaise 状态
        self.last_raiser = None              # 清除最后一个加注者
        self.n_actions_this_episode = 0      # 重置本局动作计数

        # 重置每个玩家的状态（会处理筹码随机化）
        for p in self.seats:
            p.reset()

        # 重置牌库
        self.deck.reset()

        # --- 开始新游戏流程 ---
        self._post_antes() # 收底注
        self._put_current_bets_into_main_pot_and_side_pots() # 将底注移入主池
        self._post_small_blind() # 收小盲注
        self._post_big_blind() # 收大盲注
        # 确定翻牌前第一个行动者
        self.current_player = self._get_first_to_act_pre_flop()
        # 发牌 (根据 current_round = PREFLOP，会发底牌)
        self._deal_next_round()

        # --- 可选：同步牌面状态 ---
        if deck_state_dict is not None:
            self.load_cards_state_dict(cards_state_dict=deck_state_dict)

        # --- 返回初始状态 ---
        # info: [是否轮到机会节点行动(发牌), 转换前的状态字典(如果需要)]
        # reset 后是玩家行动，所以 chance_acts=False, state_dict=None
        # (原 info=[False, None] 可能是旧格式，统一用字典)
        initial_info = {"chance_acts": False, "state_dict_before_money_move": None}
        return self._get_current_step_returns(is_terminal=False, info=initial_info)

    def step_raise_pot_frac(self, pot_frac):
        """
        执行一个按当前底池比例加注的动作。
        这个函数假定当前玩家想要加注。

        Args:
            pot_frac (float): 加注额相对于底池的比例 (e.g., 0.5 for half pot, 1.0 for pot size raise)。

        Returns:
            tuple: (obs, rew_for_all_players, done?, info) step() 的返回值。
        """
        # 1. 计算按比例加注对应的筹码量
        raise_amount_chips = self.get_fraction_of_pot_raise(fraction=pot_frac, player_that_bets=self.current_player)
        # 2. 构建标准化的动作元组
        processed_action = (Poker.BET_RAISE, raise_amount_chips)
        # 3. 调用核心 _step 函数处理该动作
        return self._step(processed_action=processed_action)

    def step_from_processed_tuple(self, action):
        """
        直接使用标准化的动作元组执行一步。

        Args:
            action (tuple or list): 标准化的动作 (action_idx, raise_size)。

        Returns:
            tuple: (obs, rew_for_all_players, done?, info) step() 的返回值。
        """
        # 直接调用核心 _step 函数
        return self._step(action)

    def step(self, action):
        """
        标准 Gym 风格的 step 函数。接收环境特定的动作，执行一步。

        Args:
            action: 环境特定（子类定义）的动作表示。

        Returns:
            tuple: (obs, rew_for_all_players, done?, info)
                   观察, 所有玩家的奖励, 是否结束?, 附加信息
        """
        # 1. 将环境特定的动作转换为标准格式
        processed_action = self._get_env_adjusted_action_formulation(action)
        # 2. 调用核心 _step 函数处理标准化动作
        return self._step(processed_action=processed_action)

    def state_dict(self):
        """
        返回包含当前环境完整状态的字典。
        用于保存、加载、调试或传递状态。
        """
        env_state_dict = {
            EnvDictIdxs.is_evaluating: self.IS_EVALUATING,         # 是否评估模式
            EnvDictIdxs.current_round: self.current_round,         # 当前轮次
            EnvDictIdxs.side_pots: copy.deepcopy(self.side_pots), # 边池列表 (深拷贝)
            EnvDictIdxs.main_pot: self.main_pot,                 # 主池金额
            EnvDictIdxs.board_2d: np.copy(self.board),             # 公共牌 (拷贝)
            EnvDictIdxs.last_action: copy.deepcopy(self.last_action), # 最后动作 (深拷贝)
            # CappedRaise 状态 (只存储相关玩家的 ID)
            EnvDictIdxs.capped_raise: [self.capped_raise.player_that_raised.seat_id,
                                       (None if self.capped_raise.player_that_cant_reopen is None
                                        else self.capped_raise.player_that_cant_reopen.seat_id)]
                                      if self.capped_raise.happened_this_round else None,
            EnvDictIdxs.current_player: self.current_player.seat_id if self.current_player else None, # 当前玩家 ID
            EnvDictIdxs.last_raiser: None if self.last_raiser is None else self.last_raiser.seat_id, # 最后加注者 ID
            EnvDictIdxs.deck: self.deck.state_dict(),              # 牌库状态字典
            EnvDictIdxs.n_actions_this_episode: self.n_actions_this_episode, # 本局动作数
            # 所有玩家的状态字典列表
            EnvDictIdxs.seats:
                [
                    { # 每个玩家的状态
                        PlayerDictIdxs.seat_id: p.seat_id,                 # 座位 ID
                        PlayerDictIdxs.hand: np.copy(p.hand) if p.hand is not None else None, # 手牌 (拷贝)
                        PlayerDictIdxs.hand_rank: p.hand_rank,             # 牌力等级
                        PlayerDictIdxs.stack: p.stack,                     # 筹码量
                        PlayerDictIdxs.current_bet: p.current_bet,         # 当前下注
                        PlayerDictIdxs.is_allin: p.is_allin,               # 是否全下
                        PlayerDictIdxs.folded_this_episode: p.folded_this_episode, # 是否已盖牌
                        PlayerDictIdxs.has_acted_this_round: p.has_acted_this_round, # 本轮是否已行动
                        PlayerDictIdxs.side_pot_rank: p.side_pot_rank       # 边池级别
                    }
                    for p in self.seats] # 遍历所有座位
        }
        # 如果是限注游戏，额外保存本轮加注次数
        if self.IS_FIXED_LIMIT_GAME:
            env_state_dict[EnvDictIdxs.n_raises_this_round] = self.n_raises_this_round
        return env_state_dict

    def load_state_dict(self, env_state_dict, blank_private_info=False):
        """
        从状态字典加载环境状态。

        Args:
            env_state_dict (dict): 通过 state_dict() 获取的状态字典。
            blank_private_info (bool): 如果为 True，则不加载玩家手牌信息，用于加载公共状态。默认为 False。
        """
        # --- 加载全局状态 ---
        self.IS_EVALUATING = env_state_dict[EnvDictIdxs.is_evaluating]
        self.current_round = env_state_dict[EnvDictIdxs.current_round]
        self.side_pots = copy.deepcopy(env_state_dict[EnvDictIdxs.side_pots])
        self.main_pot = env_state_dict[EnvDictIdxs.main_pot]
        self.board = np.copy(env_state_dict[EnvDictIdxs.board_2d])
        self.last_action = copy.deepcopy(env_state_dict[EnvDictIdxs.last_action])

        # --- 加载 CappedRaise 状态 ---
        capped_raise_state = env_state_dict.get(EnvDictIdxs.capped_raise) # 使用 .get() 处理可能不存在的情况
        self.capped_raise = CappedRaise() # 先重置
        if capped_raise_state is not None:
            self.capped_raise.happened_this_round = True
            # 根据 ID 找到对应的玩家对象
            raiser_id = capped_raise_state[0]
            cant_reopen_id = capped_raise_state[1]
            self.capped_raise.player_that_raised = self.seats[raiser_id] if raiser_id is not None and 0 <= raiser_id < self.N_SEATS else None
            self.capped_raise.player_that_cant_reopen = self.seats[cant_reopen_id] if cant_reopen_id is not None and 0 <= cant_reopen_id < self.N_SEATS else None

        # --- 加载当前玩家和最后加注者 ---
        current_player_id = env_state_dict.get(EnvDictIdxs.current_player)
        self.current_player = self.seats[current_player_id] if current_player_id is not None and 0 <= current_player_id < self.N_SEATS else None

        last_raiser_id = env_state_dict.get(EnvDictIdxs.last_raiser)
        self.last_raiser = self.seats[last_raiser_id] if last_raiser_id is not None and 0 <= last_raiser_id < self.N_SEATS else None

        # --- 加载牌库状态和动作计数 ---
        self.deck.load_state_dict(env_state_dict[EnvDictIdxs.deck])
        self.n_actions_this_episode = env_state_dict[EnvDictIdxs.n_actions_this_episode]

        # --- 加载限注游戏状态 ---
        if self.IS_FIXED_LIMIT_GAME:
            self.n_raises_this_round = env_state_dict.get(EnvDictIdxs.n_raises_this_round, 0) # 使用 .get 提供默认值

        # --- 加载每个玩家的状态 ---
        player_state_dicts = env_state_dict.get(EnvDictIdxs.seats, [])
        for i, p_state in enumerate(player_state_dicts):
            if i < len(self.seats): # 确保玩家对象存在
                p = self.seats[i]
                # 验证 seat_id 是否匹配 (可选但推荐)
                # assert p.seat_id == p_state[PlayerDictIdxs.seat_id]
                p.stack = p_state[PlayerDictIdxs.stack]
                p.current_bet = p_state[PlayerDictIdxs.current_bet]
                p.is_allin = p_state[PlayerDictIdxs.is_allin]
                p.folded_this_episode = p_state[PlayerDictIdxs.folded_this_episode]
                p.has_acted_this_round = p_state[PlayerDictIdxs.has_acted_this_round]
                p.side_pot_rank = p_state[PlayerDictIdxs.side_pot_rank]

                # 根据 blank_private_info 决定是否加载私有信息
                if blank_private_info:
                    p.hand = None # 清空手牌
                    p.hand_rank = None # 清空牌力
                else:
                    # 加载手牌和牌力
                    hand_data = p_state.get(PlayerDictIdxs.hand) # 使用 .get 处理可能没有手牌的情况
                    p.hand = np.copy(hand_data) if hand_data is not None else None
                    p.hand_rank = p_state.get(PlayerDictIdxs.hand_rank) # 使用 .get

    def get_current_obs(self, is_terminal):
        """
        获取当前状态的观察向量。
        这对于手动设置环境状态然后获取观察很有用，而无需实际执行 step()。

        Args:
            is_terminal (bool): 游戏是否处于终止状态。

        Returns:
            np.ndarray: 当前状态的观察向量。
        """
        # 如果是终止状态，返回全零向量
        if is_terminal:
            return np.zeros(shape=self.observation_space.shape, dtype=np.float32)

        # 计算用于标准化的基准值（平均起始筹码）
        try:
             normalization_sum = float(sum([s.starting_stack_this_episode for s in self.seats])) / self.N_SEATS
             if normalization_sum == 0: normalization_sum = 1.0 # 防止除零
        except: # 如果计算出错，使用默认值 1.0
             normalization_sum = 1.0

        # 组合牌桌状态、所有玩家状态和公共牌状态，形成最终的观察向量
        table_state = self._get_table_state(normalization_sum=normalization_sum)
        player_states = self._get_player_states_all_players(normalization_sum=normalization_sum)
        board_state = self._get_board_state()

        # 将所有部分连接成一个 NumPy 数组
        # 注意：需要确保各个部分的类型一致（通常是 float32）
        # return np.array(table_state + player_states + board_state, dtype=np.float32)
        # 使用 np.concatenate 更安全
        return np.concatenate([
             np.array(table_state, dtype=np.float32),
             np.array(player_states, dtype=np.float32),
             np.array(board_state, dtype=np.float32)
        ])


    def print_obs(self, obs):
        """打印观察向量的每个组成部分及其值，方便调试。"""
        print("______________________________________ 打印观察向量 _________________________________________")
        # 准备名称列表，并找到最长名称以对齐打印
        names = [e + ":  " for e in list(self.obs_idx_dict.keys())]
        if not names: return # 如果 obs_idx_dict 为空
        str_len = max([len(e) for e in names]) if names else 0
        # 遍历索引字典，打印对应的值
        for name, key in zip(names, list(self.obs_idx_dict.keys())):
            name = name.rjust(str_len) # 右对齐名称
            try:
                 print(name, obs[self.obs_idx_dict[key]]) # 打印名称和对应索引的值
            except IndexError:
                 print(f"{name} 索引 {self.obs_idx_dict[key]} 超出观察向量边界 (长度 {len(obs)})")
            except KeyError:
                 print(f"{name} 键 '{key}' 不在 obs_idx_dict 中")


    def set_to_public_tree_node_state(self, node):
        """
        将环境状态设置为一个公共信息树节点的状态。
        由于节点的 {私有牌, 剩余牌库, 公共牌} 可能与当前环境冲突，规则如下:
        - 公共牌来自节点。
        - 私有牌（底牌）来自当前环境（保留）。
        - 剩余牌库不重要（忽略节点的牌库状态）。
        这可能导致牌张重复（如果使用不当），但最符合用例。

        Args:
            node: 一个包含公共状态信息的对象，需要有 .env_state 属性（状态字典）。
        """
        # 加载节点的状态字典，同时清除私有信息（手牌）
        # load_state_dict 会保留环境当前的牌库状态
        self.load_state_dict(node.env_state, blank_private_info=True)
        # 显式地设置公共牌 (因为 load_state_dict(blank=True) 可能不会覆盖所有细节？确保覆盖)
        # self.board = np.copy(node.env_state[EnvDictIdxs.board_2d]) # load_state_dict 应该已经做了

    def cards2str(self, cards_2d, seperator=", "):
        """
        将二维牌张表示转换为可读的字符串。

        Args:
            cards_2d (np.ndarray): 任意数量牌张的二维表示 [[rank, suit], ...]
            seperator (str): 打印时牌张之间的分隔符。

        Returns:
            str: 牌张的字符串表示。
        """
        hand_as_str = ""
        if cards_2d is None: return "" # 处理 None 输入
        try:
            for c in cards_2d:
                # 检查是否是有效的牌张（不是未发牌标记）
                if not np.array_equal(c, Poker.CARD_NOT_DEALT_TOKEN_2D) and c[0] != Poker.CARD_NOT_DEALT_TOKEN_1D:
                     # 检查点数和花色是否在字典中
                     rank_str = self.RANK_DICT.get(c[0], '?') # 使用 .get 提供默认值
                     suit_str = self.SUIT_DICT.get(c[1], '?')
                     hand_as_str += rank_str + suit_str + seperator
        except (TypeError, IndexError): # 处理 cards_2d 不是预期格式的情况
            return "[无效牌数据]"
        # 移除末尾的分隔符
        return hand_as_str.rstrip(seperator)


    def get_legal_actions(self):
        """
        获取当前玩家可以执行的合法动作列表。

        Returns:
            list: 合法动作索引的列表 (Poker.FOLD, Poker.CHECK_CALL, Poker.BET_RAISE 中的子集)。
        """
        legal_actions = []
        # --- 检查 FOLD 是否合法 ---
        # 意图动作是 FOLD
        intended_action_fold = (Poker.FOLD, -1,)
        # 获取修正后的动作
        fixed_action_fold = self._get_fixed_action(action=intended_action_fold)
        # 如果修正后的动作仍然是 FOLD，则 FOLD 合法
        if fixed_action_fold[0] == Poker.FOLD:
            legal_actions.append(Poker.FOLD)

        # --- 检查 CHECK/CALL 是否合法 ---
        # 意图动作是 CHECK/CALL
        intended_action_cc = (Poker.CHECK_CALL, -1,)
        # 获取修正后的动作
        fixed_action_cc = self._get_fixed_action(action=intended_action_cc)
        # 如果修正后的动作仍然是 CHECK/CALL，则 CHECK/CALL 合法
        if fixed_action_cc[0] == Poker.CHECK_CALL:
            legal_actions.append(Poker.CHECK_CALL)

        # --- 检查 BET/RAISE 是否合法 ---
        # 意图动作是 BET/RAISE (金额暂时设为 1，因为 _get_fixed_action 会修正)
        # (注意：_get_env_adjusted_action_formulation 可能需要先调用，
        #  但这里假设 _get_fixed_action 能处理基本元组)
        # intended_action_raise = self._get_env_adjusted_action_formulation(action=(Poker.BET_RAISE, 1,)) # 更严谨
        intended_action_raise = (Poker.BET_RAISE, 1,)
        # 获取修正后的动作
        fixed_action_raise = self._get_fixed_action(action=intended_action_raise)
        # 如果修正后的动作仍然是 BET/RAISE，则 BET/RAISE 合法
        if fixed_action_raise[0] == Poker.BET_RAISE:
            legal_actions.append(Poker.BET_RAISE)

        return legal_actions

    def get_range_idx(self, p_id):
        """
        获取指定玩家底牌对应的范围索引 (range index)。
        范围索引是手牌的一种整数表示，通常用于策略查找或存储。

        Args:
            p_id (int): 玩家的座位 ID。

        Returns:
            int: 底牌的范围索引。
        """
        # 获取玩家的手牌 (二维表示)
        hole_cards_2d = self.get_hole_cards_of_player(p_id=p_id)
        if hole_cards_2d is None: return None # 处理玩家没有手牌的情况
        # 使用 lut_holder 将二维手牌转换为范围索引
        return int(self.lut_holder.get_range_idx_from_hole_cards(hole_cards_2d=hole_cards_2d))

    def get_random_action(self):
        """生成一个随机的动作（主要用于测试或基准）"""
        # 随机选择动作类型 (0, 1, 2)
        a = np.random.randint(low=0, high=3)
        # 随机生成一个加注/下注金额（以当前总底池为参考）
        pot_sum = self.main_pot + sum(self.side_pots) if self.side_pots else self.main_pot
        # 生成一个大致在半池左右的正态分布随机数作为金额
        n = int(np.random.normal(loc=pot_sum / 2, scale=pot_sum / 5))
        n = max(0, n) # 确保金额非负
        # 返回随机动作元组
        return a, n

    def get_all_winnable_money(self):
        """计算当前牌桌上所有可赢得的筹码总量（主池+边池+当前下注）"""
        current_bets_sum = sum([p.current_bet for p in self.seats])
        side_pots_sum = sum(self.side_pots) if self.side_pots else 0
        return self.main_pot + current_bets_sum + side_pots_sum

    def cards_state_dict(self):
        """返回只包含牌面相关信息的状态字典"""
        return {
            "deck": self.deck.state_dict(), # 牌库状态
            "board": np.copy(self.board),    # 公共牌
            # 所有玩家的手牌列表
            "hand": [np.copy(p.hand) if p.hand is not None else None for p in self.seats]
        }

    def load_cards_state_dict(self, cards_state_dict):
        """从字典加载牌面相关信息"""
        self.deck.load_state_dict(cards_state_dict["deck"])
        self.board = np.copy(cards_state_dict["board"])
        hands = cards_state_dict.get("hand", [])
        for i, hand_data in enumerate(hands):
            if i < len(self.seats):
                self.seats[i].hand = np.copy(hand_data) if hand_data is not None else None

    def reshuffle_remaining_deck(self):
        """重新洗牌库中剩余的牌（不包括已发出的牌）"""
        self.deck.shuffle()


    def get_fraction_of_pot_raise(self, fraction, player_that_bets):
        """
        将相对于当前底池大小的下注/加注比例转换为具体的筹码数量。
        底池大小计算为：主池 + 所有边池 + 所有当前下注 + 当前玩家需要跟注的额外面额。
        这个计算方式遵循通用的"底池大小下注"定义。

        Args:
            fraction (float):       下注/加注额相对于底池的比例 (e.g., 0.5 for 半池, 1.0 for 满池)。
            player_that_bets (int or PokerPlayer): 执行下注/加注的玩家（可以是 ID 或对象）。

        Returns:
            int: 计算得出的、玩家下注后桌上应有的总筹码量。
        """
        # 获取玩家对象
        _player = player_that_bets if isinstance(player_that_bets, PokerPlayer) else self.seats[player_that_bets]

        # 计算玩家需要额外跟注的额度
        to_call = self._get_biggest_bet_out_there_aka_total_to_call() - _player.current_bet
        to_call = max(0, to_call) # 确保非负

        # 计算跟注后的底池大小
        pot_before_action = self.main_pot \
                            + (sum(self.side_pots) if self.side_pots else 0) \
                            + sum([p.current_bet for p in self.seats])
        pot_after_call = pot_before_action + to_call

        # 计算加注的增量 = 需要跟注的额度 + 底池比例下注额
        delta = int(to_call + (pot_after_call * fraction))
        # 计算最终玩家桌上的总下注额
        total_raise = delta + _player.current_bet

        # 返回计算出的总下注额
        return total_raise

    def get_frac_from_chip_amt(self, amt, player_that_bets):
        """
        执行与 get_fraction_of_pot_raise() 相反的操作：
        将一个具体的总下注筹码量转换为相对于底池的比例。

        Args:
            amt (int):              玩家下注后的总筹码量。
            player_that_bets (int or PokerPlayer): 执行下注的玩家。

        Returns:
            float: 计算得出的相对于底池的下注/加注比例。
                   如果计算结果无效（例如，分母为零），可能返回 0 或其他值。
        """
        # 获取玩家对象
        _player = player_that_bets if isinstance(player_that_bets, PokerPlayer) else self.seats[player_that_bets]

        # 计算需要额外跟注的额度
        to_call = self._get_biggest_bet_out_there_aka_total_to_call() - _player.current_bet
        to_call = max(0, to_call)

        # 计算跟注后的底池大小
        pot_before_action = self.main_pot \
                            + (sum(self.side_pots) if self.side_pots else 0) \
                            + sum([p.current_bet for p in self.seats])
        pot_after_call = pot_before_action + to_call

        # 计算实际下注/加注的增量
        delta = amt - _player.current_bet
        # 计算相对于底池的比例 = (总增量 - 跟注部分) / 跟注后的底池
        # 添加检查以防除以零
        if pot_after_call > 0:
            fraction = float(delta - to_call) / float(pot_after_call)
        else:
            fraction = 0.0 # 或者根据需要返回 None 或引发错误

        # 返回计算出的比例
        return fraction

    def get_hole_cards_of_player(self, p_id):
        """获取指定玩家的底牌（私有信息）"""
        # 检查 p_id 是否有效
        if 0 <= p_id < len(self.seats):
            return self.seats[p_id].hand
        else:
            return None # 或者引发错误

    def eval(self):
        """
        将环境设置为评估模式。
        评估模式不允许随机化（例如起始筹码随机化）。
        """
        self.IS_EVALUATING = True
        # 将所有玩家对象也设置为评估模式
        for p in self.seats:
            p.IS_EVALUATING = True

    def training(self):
        """
        将环境设置为训练模式。
        允许根据初始化参数定义的随机化。
        """
        self.IS_EVALUATING = False
        # 将所有玩家对象也设置为训练模式
        for p in self.seats:
            p.IS_EVALUATING = False

    def render(self, mode='TEXT'):
        """
        渲染（打印或显示）当前环境的状态。

        Args:
            mode (str): 渲染模式。目前仅支持 'TEXT'。
        """

        if mode.upper() == 'TEXT':
            # 打印分隔符和标题，显示当前轮次和行动玩家
            print('\n\n')
            print('___________________________________',
                  (Poker.INT2STRING_ROUND.get(self.current_round, '?') + " - " + # 使用 .get 处理无效轮次
                   (str(self.current_player.seat_id) if self.current_player else '?') + " acts").center(15),
                  '___________________________________')

            # 打印公共牌
            print("Board: ", self.cards2str(self.board))

            # 打印最后动作信息和主池信息
            last_action_str = "None"
            if self.last_action and self.last_action[0] is not None:
                 action_name = ["FOLD", "CHECK/CALL", "BET/RAISE"][self.last_action[0]]
                 player_id = self.last_action[2]
                 amount_str = str(self.last_action[1]) if self.last_action[0] == Poker.BET_RAISE else ""
                 last_action_str = f"player_{player_id}: {action_name} {amount_str}"

            print(f"Last Action:   {last_action_str}".ljust(113), # 使用 ljust 对齐
                  f"|   Main_pot: {str(self.main_pot).rjust(7)}")

            # 打印每个玩家的信息
            for p in self.seats:
                # 标记玩家状态：'-' 盖牌, '+' 全下, '' 正常
                status_flag = "-" if p.folded_this_episode else "+" if p.is_allin else " "
                player_info = (f"{status_flag}Player_{p.seat_id}:").rjust(14) + \
                              f" stack: {str(p.stack).rjust(8)}" + \
                              f" current_bet: {str(p.current_bet).rjust(8)}" + \
                              f" side_pot_rank: {str(p.side_pot_rank).rjust(8)}" + \
                              f" hand: {self.cards2str(p.hand).rjust(8)}" # 打印手牌（用于调试）

                # 打印边池信息（与玩家信息在同一行）
                side_pot_info = ""
                if self.side_pots and p.seat_id < len(self.side_pots):
                     side_pot_info = f'|   Side_pot{p.seat_id}: {str(self.side_pots[p.seat_id]).rjust(6)}'

                print(player_info.ljust(95) + side_pot_info) # 使用 ljust 对齐

            # 如果是限注游戏，打印本轮加注次数
            if self.IS_FIXED_LIMIT_GAME:
                print("Num raises this round: ", self.n_raises_this_round)
            print("\n") # 打印空行

        else: # 不支持的渲染模式
            raise ValueError(f"渲染模式 {mode} 不被支持")

    def print_tutorial(self):
        """打印一个简单的教程，说明人类玩家如何输入动作。"""
        print("0 \t弃牌 (Fold)")
        print("1 \t看牌/跟注 (Call/Check)")
        print("2 \t下注/加注 (Raise/Bet)")
        print("如果你输入 2 进行加注，并且游戏允许不同的下注大小，系统会询问你想要加注到的总额。")

    def human_api_ask_action(self):
        """
        为人类玩家提供交互接口，获取其动作输入。
        """
        # 循环直到获得有效的动作类型输入
        while True:
            try:
                # 提示当前玩家输入动作索引
                action_idx_str = input(f"玩家 {self.current_player.seat_id} 请输入你的动作 (0:弃牌, 1:看牌/跟注, 2:下注/加注): ")
                action_idx = int(action_idx_str)
            except ValueError: # 处理非整数输入
                print("输入无效，请输入数字 0, 1 或 2。请参考教程。")
                continue
            # 检查动作索引是否在合法范围内
            if action_idx not in [Poker.FOLD, Poker.CHECK_CALL, Poker.BET_RAISE]:
                print("无效的动作索引！请输入 0, 1 或 2。")
                continue
            # 如果输入合法，跳出循环
            break

        # 如果动作是下注/加注
        raise_size = 0 # 默认为 0
        if action_idx == Poker.BET_RAISE:
             # 循环直到获得有效的加注金额输入
             while True:
                 try:
                     # 提示输入加注到的总金额
                     raise_size_str = input("请输入你想要下注/加注到的总金额: ")
                     raise_size = int(raise_size_str)
                     if raise_size < 0: # 确保金额非负
                          print("金额不能为负数。")
                          continue
                     break # 金额有效，跳出循环
                 except ValueError: # 处理非整数输入
                     print("输入无效，请输入一个数字。")
                     continue

        # 返回动作元组 [动作类型, 金额]
        action_tuple = [action_idx, raise_size]
        return action_tuple

    def set_stack_size(self, stack_size):
        """
        设置（覆盖）所有玩家的起始筹码量。
        注意：这会重新初始化环境的部分状态。

        Args:
            stack_size (list or int): 如果是列表，则为每个玩家设置对应的起始筹码。
                                     如果是整数，则所有玩家使用相同的起始筹码。
        """
        args = copy.deepcopy(self._args) # 获取当前参数副本
        # 处理 stack_size 输入
        if isinstance(stack_size, int):
             # 如果是整数，为每个座位创建列表
             args.starting_stack_sizes_list = [stack_size] * self.N_SEATS
        elif isinstance(stack_size, list) and len(stack_size) == self.N_SEATS:
             # 如果是列表且长度匹配，直接使用
             args.starting_stack_sizes_list = copy.deepcopy(stack_size)
        else:
             # 输入格式无效
             raise ValueError("stack_size 必须是整数或长度等于座位数的列表")

        # 使用更新后的参数重新初始化环境的相关部分
        # is_evaluating 状态保持不变
        self._init_from_args(env_args=args, is_evaluating=self.IS_EVALUATING)

    def get_args(self):
        """返回环境初始化参数的深拷贝"""
        return copy.deepcopy(self._args)

    def set_args(self, env_args):
        """
        使用新的参数对象重新初始化环境。
        警告：这会完全重置环境状态，类似于重新创建实例。

        Args:
            env_args: 新的环境参数对象。
        """
        # 使用新参数完全重新初始化
        # is_evaluating 状态保持不变
        self._init_from_args(env_args=env_args, is_evaluating=self.IS_EVALUATING)


class CappedRaise:
    """
    扑克游戏中有一个特殊规则：当玩家 A 加注，玩家 B 再加注但选择全下（all-in），
    而玩家 B 的筹码量不足以构成一次最小加注额时，如果之后轮到玩家 A 再次行动，
    玩家 A 将不能进行再加注（他的行动被 "capped" 或限制了）。
    但是，牌桌上的其他玩家仍然可以选择加注。
    这个限制会在以下情况下被解除：
    a) 有其他玩家（非玩家 A 或 B）进行了合法的再加注。
    b) 当前的下注轮结束。

    此类用于跟踪和管理这种状态。
    """

    def __init__(self):
        """初始化 CappedRaise 状态"""
        self.happened_this_round = None     # 布尔值: 本轮是否已发生不足额全下再加注的情况
        self.player_that_raised = None      # PokerPlayer 对象: 进行了不足额全下再加注的玩家 (玩家 B)
        self.player_that_cant_reopen = None # PokerPlayer 对象: 被限制不能再加注的玩家 (玩家 A)
        self.reset() # 调用 reset 进行初始化

    def reset(self):
        """在每轮下注开始时重置 CappedRaise 状态"""
        self.happened_this_round = False     # 重置发生标志
        self.player_that_raised = None      # 清除相关玩家记录
        self.player_that_cant_reopen = None # 清除相关玩家记录
