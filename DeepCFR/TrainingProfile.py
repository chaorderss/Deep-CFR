# Copyright (c) 2019 Eric Steinberger # 版权信息

import copy  # 导入 copy 模块，用于深拷贝对象（如下注设置）
import torch # 导入 PyTorch 库

from PokerRL.game import bet_sets # 从 PokerRL 库导入预定义的下注尺寸集合
from PokerRL.game.games import DiscretizedNLLeduc # 从 PokerRL 库导入特定的扑克游戏变种 (离散化 Leduc Hold'em)
from PokerRL.game.wrappers import HistoryEnvBuilder, FlatLimitPokerEnvBuilder # 导入环境构建器，用于根据网络类型创建合适的游戏环境接口
from PokerRL.rl.base_cls.TrainingProfileBase import TrainingProfileBase # 导入所有训练配置文件的基类
from PokerRL.rl.neural.AvrgStrategyNet import AvrgNetArgs # 导入平均策略网络参数类
from PokerRL.rl.neural.DuelingQNet import DuelingQArgs # 导入 Dueling Q-Network 参数类 (用于优势网络)

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR # 导入 DeepCFR 的评估代理
from DeepCFR.workers.la.AdvWrapper import AdvTrainingArgs # 导入优势网络训练参数包装类
from DeepCFR.workers.la.AvrgWrapper import AvrgTrainingArgs # 导入平均策略网络训练参数包装类


class TrainingProfile(TrainingProfileBase):
    """
    此类用于配置 DeepCFR 算法的训练运行。
    它继承自 TrainingProfileBase，并设置了 DeepCFR 特有的参数以及通用的训练、环境和网络参数。
    """

    def __init__(self,

                 # ------ 通用设置 ------
                 name="",                           # 实验或训练运行的名称
                 log_verbose=True,                  # 是否在控制台打印详细日志
                 log_export_freq=1,                 # 每隔多少次迭代导出一次日志/指标 (例如到 TensorBoard)
                 checkpoint_freq=99999999,          # 每隔多少次迭代保存一次训练检查点 (模型状态等)
                 eval_agent_export_freq=999999999,  # 每隔多少次迭代导出一次可用于评估的智能体
                 n_learner_actor_workers=8,         # (分布式) Learner Actor 工作进程的数量
                 max_n_las_sync_simultaneously=10,  # (分布式) 同时同步的最大 Learner Actor 数量
                 nn_type="feedforward",             # 神经网络类型: "recurrent" (RNN) 或 "feedforward" (MLP/全连接)

                 # ------ 计算资源设置 ------
                 path_data=None,                    # 存储日志、模型等数据的路径
                 local_crayon_server_docker_address="localhost", # (可选) 用于日志记录的 Crayon/TensorBoard 服务器地址
                 device_inference="cpu",            # 推理时使用的设备 ("cpu" or "cuda")
                 device_training="cpu",             # 训练神经网络时使用的设备 ("cpu" or "cuda")
                 device_parameter_server="cpu",     # (分布式) 参数服务器使用的设备 ("cpu" or "cuda")
                 DISTRIBUTED=False,                 # 是否启用分布式训练 (使用 Ray)
                 CLUSTER=False,                     # 是否在集群环境运行 (通常与 DISTRIBUTED 一起使用)
                 DEBUGGING=False,                   # 是否启用调试模式 (可能会影响性能或行为)

                 # ------ 环境设置 ------
                 game_cls=DiscretizedNLLeduc,       # 要训练的扑克游戏类 (例如: Leduc, NoLimit Hold'em)
                 n_seats=2,                         # 游戏中的玩家座位数
                 agent_bet_set=bet_sets.B_2,        # 智能体允许使用的下注尺寸集合 (相对于底池大小)
                 start_chips=None,                  # 玩家的初始筹码量 (如果为 None, 通常使用游戏默认值)
                 chip_randomness=(0, 0),            # 初始筹码的随机范围 (最小值偏移, 最大值偏移)
                 uniform_action_interpolation=False,# 是否在策略中对动作进行均匀插值
                 use_simplified_headsup_obs=True,   # (特定游戏) 是否使用简化的单挑 (heads-up) 观测空间
                 # ------ 评估设置 ------
                 eval_modes_of_algo=(EvalAgentDeepCFR.EVAL_MODE_SINGLE,), # 评估算法的模式 (例如: 单独评估)
                 eval_stack_sizes=None,             # 评估时使用的筹码量列表 (如果为 None, 通常使用训练时的筹码)

                 # ------ 通用 Deep CFR 参数 ------
                 n_traversals_per_iter=30000,       # DeepCFR 每次迭代中进行的博弈树遍历次数 (采样游戏轨迹)
                 online=False,                      # 是否使用在线学习模式 (通常 DeepCFR 是离线的)
                 iter_weighting_exponent=1.0,       # 迭代加权指数 (用于计算平均策略，通常为 1.0，即线性加权)
                 n_actions_traverser_samples=3,     # 在遍历过程中，对于每个信息集，采样多少个动作来构建子树 (CFR+ 变种或采样策略)
                                                    # 注意：这个参数可能与特定的采样器实现有关

                 sampler="mo",                      # 采样器类型 ("external", "outcome", "mo" - Monte Carlo)

                 # --- 优势网络 (Advantage Network) 超参数 ---
                 # 这个网络用于近似每个状态动作对的即时后悔值 (Immediate Regret)
                 n_batches_adv_training=5000,       # 每个 Learner Actor 更新周期内，训练优势网络的批次数
                 init_adv_model="random",           # 优势网络模型的初始化方式 ("random" 或加载已有模型路径)

                 rnn_cls_str_adv="lstm",            # (如果 nn_type="recurrent") RNN 单元类型 ("lstm", "gru")
                 rnn_units_adv=128,                 # (如果 nn_type="recurrent") RNN 单元数量
                 rnn_stack_adv=1,                   # (如果 nn_type="recurrent") RNN 层数
                 dropout_adv=0.0,                   # (如果 nn_type="recurrent") RNN 的 Dropout 比率
                 use_pre_layers_adv=False,          # 是否在主网络模块前使用预处理层
                 n_cards_state_units_adv=96,        # (根据网络结构) 处理牌信息的层/块的单元数
                 n_merge_and_table_layer_units_adv=32, # (根据网络结构) 处理合并信息和牌桌状态的层的单元数
                 n_units_final_adv=64,              # 优势网络最后隐藏层的单元数
                 mini_batch_size_adv=4096,          # 训练优势网络时的小批量大小
                 n_mini_batches_per_la_per_update_adv=1, # 每个 Learner Actor 更新周期内，使用多少个小批量数据更新一次网络
                 optimizer_adv="adam",              # 优势网络的优化器 ("adam", "sgd", etc.)
                 loss_adv="weighted_mse",           # 优势网络的损失函数 ("mse", "weighted_mse" - 通常用后悔值加权)
                 lr_adv=0.001,                      # 优势网络的学习率
                 grad_norm_clipping_adv=10.0,       # 优势网络的梯度裁剪阈值 (防止梯度爆炸)
                 lr_patience_adv=999999999,         # (如果使用学习率调度器) 多少次没有改善后降低学习率 (这里设置得很大，基本等于禁用)
                 normalize_last_layer_FLAT_adv=True,# (如果 nn_type="feedforward") 是否归一化最后一个特征层

                 max_buffer_size_adv=3e6,           # 优势网络训练数据的重放缓冲区 (Replay Buffer) 最大容量

                 # ------ 平均策略网络 (Average Strategy Network) 超参数 ------
                 # 这个网络用于近似累积的平均策略 (即在所有迭代中动作被选择的频率)
                 n_batches_avrg_training=15000,     # 每个 Learner Actor 更新周期内，训练平均策略网络的批次数
                 init_avrg_model="random",          # 平均策略网络模型的初始化方式

                 rnn_cls_str_avrg="lstm",           # (如果 nn_type="recurrent") RNN 类型
                 rnn_units_avrg=128,                # (如果 nn_type="recurrent") RNN 单元数
                 rnn_stack_avrg=1,                  # (如果 nn_type="recurrent") RNN 层数
                 dropout_avrg=0.0,                  # (如果 nn_type="recurrent") RNN Dropout 比率
                 use_pre_layers_avrg=False,         # 是否使用预处理层
                 n_cards_state_units_avrg=96,       # (根据网络结构) 处理牌信息的单元数
                 n_merge_and_table_layer_units_avrg=32,# (根据网络结构) 处理合并信息的单元数
                 n_units_final_avrg=64,             # 平均策略网络最后隐藏层的单元数
                 mini_batch_size_avrg=4096,         # 训练平均策略网络时的小批量大小
                 n_mini_batches_per_la_per_update_avrg=1,# 每个 Learner Actor 更新周期内，使用多少个小批量数据更新一次网络
                 loss_avrg="weighted_mse",          # 平均策略网络的损失函数 ("mse", "weighted_mse" - 通常用迭代次数加权)
                 optimizer_avrg="adam",             # 平均策略网络的优化器
                 lr_avrg=0.001,                     # 平均策略网络的学习率
                 grad_norm_clipping_avrg=10.0,      # 平均策略网络的梯度裁剪阈值
                 lr_patience_avrg=999999999,        # 平均策略网络的学习率耐心值
                 normalize_last_layer_FLAT_avrg=True,# (如果 nn_type="feedforward") 是否归一化最后特征层

                 max_buffer_size_avrg=3e6,          # 平均策略网络训练数据的重放缓冲区最大容量

                 # ------ 特定于 SINGLE 模式的参数 ------ (Single 模式可能是指非分布式或特定评估方式)
                 export_each_net=False,             # 是否在导出评估智能体时，分别导出优势和平均策略网络 (而不是合并的评估智能体)
                 eval_agent_max_strat_buf_size=None,# 评估智能体内部策略缓冲区的最大大小 (可能用于平滑策略或特定评估)

                 # ------ 可选模块参数 ------ (用于集成其他算法或评估方法)
                 lbr_args=None,                     # LBR (Local Best Response) 算法的参数
                 rl_br_args=None,                   # RLBR (Reinforcement Learning Best Response) 算法的参数
                 h2h_args=None,                     # H2H (Head-to-Head) 对抗评估的参数

                 ):
        print(" ************************** Initing args for: ", name, "  **************************")

        # --- 根据 nn_type 配置网络结构参数和环境构建器 ---
        if nn_type == "recurrent":
            # 如果使用循环神经网络 (RNN)
            from PokerRL.rl.neural.MainPokerModuleRNN import MPMArgsRNN # 导入 RNN 主扑克模块的参数类

            env_bldr_cls = HistoryEnvBuilder # RNN 需要历史信息，使用 HistoryEnvBuilder

            # 为优势网络配置 RNN 参数
            mpm_args_adv = MPMArgsRNN(rnn_cls_str=rnn_cls_str_adv,
                                      rnn_units=rnn_units_adv,
                                      rnn_stack=rnn_stack_adv,
                                      rnn_dropout=dropout_adv,
                                      use_pre_layers=use_pre_layers_adv,
                                      n_cards_state_units=n_cards_state_units_adv,
                                      n_merge_and_table_layer_units=n_merge_and_table_layer_units_adv)
            # 为平均策略网络配置 RNN 参数
            mpm_args_avrg = MPMArgsRNN(rnn_cls_str=rnn_cls_str_avrg,
                                       rnn_units=rnn_units_avrg,
                                       rnn_stack=rnn_stack_avrg,
                                       rnn_dropout=dropout_avrg,
                                       use_pre_layers=use_pre_layers_avrg,
                                       n_cards_state_units=n_cards_state_units_avrg,
                                       n_merge_and_table_layer_units=n_merge_and_table_layer_units_avrg)

        elif nn_type == "feedforward":
            # 如果使用前馈神经网络 (Feedforward / MLP)
            from PokerRL.rl.neural.MainPokerModuleFLAT import MPMArgsFLAT # 导入 Flat (前馈) 主扑克模块的参数类

            env_bldr_cls = FlatLimitPokerEnvBuilder # 前馈网络通常用于状态信息扁平化的环境

            # 为优势网络配置前馈网络参数
            mpm_args_adv = MPMArgsFLAT(use_pre_layers=use_pre_layers_adv,
                                       card_block_units=n_cards_state_units_adv,
                                       other_units=n_merge_and_table_layer_units_adv,
                                       normalize=normalize_last_layer_FLAT_adv)
            # 为平均策略网络配置前馈网络参数
            mpm_args_avrg = MPMArgsFLAT(use_pre_layers=use_pre_layers_avrg,
                                        card_block_units=n_cards_state_units_avrg,
                                        other_units=n_merge_and_table_layer_units_avrg,
                                        normalize=normalize_last_layer_FLAT_avrg)

        else:
            # 如果 nn_type 无效，则抛出错误
            raise ValueError(nn_type)

        # --- 调用父类 (TrainingProfileBase) 的构造函数 ---
        # 将通用的和特定模块的参数传递给基类进行初始化
        super().__init__(
            # 通用参数
            name=name,
            log_verbose=log_verbose,
            log_export_freq=log_export_freq,
            checkpoint_freq=checkpoint_freq,
            eval_agent_export_freq=eval_agent_export_freq,
            path_data=path_data,
            game_cls=game_cls,
            env_bldr_cls=env_bldr_cls, # 传递上面根据 nn_type 选择的环境构建器
            start_chips=start_chips,
            eval_modes_of_algo=eval_modes_of_algo,
            eval_stack_sizes=eval_stack_sizes,
            # 计算和调试参数
            DEBUGGING=DEBUGGING,
            DISTRIBUTED=DISTRIBUTED,
            CLUSTER=CLUSTER,
            device_inference=device_inference,
            local_crayon_server_docker_address=local_crayon_server_docker_address,

            # --- 模块特定参数 (Module Arguments) ---
            # 这是一个字典，包含了各个子模块（如优势网络训练、平均策略网络训练、环境等）的配置对象
            module_args={
                # 优势网络训练模块的参数
                "adv_training": AdvTrainingArgs(
                    adv_net_args=DuelingQArgs( # 使用 Dueling Q-Network 结构作为优势网络
                        mpm_args=mpm_args_adv, # 传递上面配置好的主扑克模块参数 (RNN 或 Flat)
                        n_units_final=n_units_final_adv, # 最后隐藏层的单元数
                    ),
                    n_batches_adv_training=n_batches_adv_training, # 训练批次数
                    init_adv_model=init_adv_model, # 初始化方式
                    batch_size=mini_batch_size_adv, # 批次大小
                    n_mini_batches_per_update=n_mini_batches_per_la_per_update_adv, # 每次更新使用的小批量数
                    optim_str=optimizer_adv, # 优化器
                    loss_str=loss_adv, # 损失函数
                    lr=lr_adv, # 学习率
                    grad_norm_clipping=grad_norm_clipping_adv, # 梯度裁剪
                    device_training=device_training, # 训练设备
                    max_buffer_size=max_buffer_size_adv, # 缓冲区大小
                    lr_patience=lr_patience_adv, # 学习率耐心
                ),
                # 平均策略网络训练模块的参数
                "avrg_training": AvrgTrainingArgs(
                    avrg_net_args=AvrgNetArgs( # 使用 Average Strategy Network 结构
                        mpm_args=mpm_args_avrg, # 传递上面配置好的主扑克模块参数 (RNN 或 Flat)
                        n_units_final=n_units_final_avrg, # 最后隐藏层的单元数
                    ),
                    n_batches_avrg_training=n_batches_avrg_training, # 训练批次数
                    init_avrg_model=init_avrg_model, # 初始化方式
                    batch_size=mini_batch_size_avrg, # 批次大小
                    n_mini_batches_per_update=n_mini_batches_per_la_per_update_avrg, # 每次更新使用的小批量数
                    loss_str=loss_avrg, # 损失函数
                    optim_str=optimizer_avrg, # 优化器
                    lr=lr_avrg, # 学习率
                    grad_norm_clipping=grad_norm_clipping_avrg, # 梯度裁剪
                    device_training=device_training, # 训练设备
                    max_buffer_size=max_buffer_size_avrg, # 缓冲区大小
                    lr_patience=lr_patience_avrg, # 学习率耐心
                ),
                # 环境模块的参数
                "env": game_cls.ARGS_CLS( # 使用指定游戏类的参数类来配置环境
                    n_seats=n_seats, # 玩家数
                    starting_stack_sizes_list=[start_chips for _ in range(n_seats)], # 每个座位的初始筹码
                    bet_sizes_list_as_frac_of_pot=copy.deepcopy(agent_bet_set), # 允许的下注尺寸 (深拷贝以防意外修改)
                    stack_randomization_range=chip_randomness, # 初始筹码随机范围
                    use_simplified_headsup_obs=use_simplified_headsup_obs, # 是否用简化观测
                    uniform_action_interpolation=uniform_action_interpolation # 是否均匀插值动作
                ),
                # 可选模块的参数 (如果提供了参数对象，则启用相应模块)
                "lbr": lbr_args,
                "rlbr": rl_br_args,
                "h2h": h2h_args,
            }
        )

        # --- 将 DeepCFR 特有的参数保存为实例变量 ---
        self.nn_type = nn_type # 保存神经网络类型
        self.online = online # 保存在线模式设置
        self.n_traversals_per_iter = n_traversals_per_iter # 保存每次迭代的遍历次数
        self.iter_weighting_exponent = iter_weighting_exponent # 保存迭代加权指数
        self.sampler = sampler # 保存采样器类型
        self.n_actions_traverser_samples = n_actions_traverser_samples # 保存在遍历中采样的动作数

        # SINGLE 模式特定参数
        self.export_each_net = export_each_net # 保存是否分别导出网络
        self.eval_agent_max_strat_buf_size = eval_agent_max_strat_buf_size # 保存评估智能体策略缓冲区大小

        # --- 设置 Learner Actor 数量 (根据是否分布式) ---
        if DISTRIBUTED or CLUSTER:
            # 如果是分布式或集群模式
            print("Running with ", n_learner_actor_workers, "LearnerActor Workers.")
            self.n_learner_actors = n_learner_actor_workers # 使用指定的 Learner Actor 数量
        else:
            # 如果是本地单机模式
            self.n_learner_actors = 1 # Learner Actor 数量为 1
        self.max_n_las_sync_simultaneously = max_n_las_sync_simultaneously # 保存同时同步的 LA 最大数量

        # --- 设置参数服务器设备 ---
        assert isinstance(device_parameter_server, str), "Please pass a string (either 'cpu' or 'cuda')!" # 确认输入是字符串
        self.device_parameter_server = torch.device(device_parameter_server) # 将字符串转换为 PyTorch 设备对象