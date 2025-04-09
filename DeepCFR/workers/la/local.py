import os # 用于操作系统功能，如路径、进程信息
import pickle # 用于序列化和反序列化 Python 对象 (保存/加载 checkpoint)

import psutil # 用于获取系统信息，如内存使用情况

# 从 DeepCFR 项目导入相关类
from DeepCFR.IterationStrategy import IterationStrategy # 代表一个特定 CFR 迭代所使用的策略网络
from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR # 用于区分评估模式 (当前策略 vs 平均策略)
from DeepCFR.workers.la.buffers.AdvReservoirBuffer import AdvReservoirBuffer # 优势网络训练数据的蓄水池缓冲区
from DeepCFR.workers.la.AdvWrapper import AdvWrapper # 优势网络模型及其训练逻辑的封装器
from DeepCFR.workers.la.buffers.AvrgReservoirBuffer import AvrgReservoirBuffer # 平均策略网络训练数据的蓄水池缓冲区
from DeepCFR.workers.la.AvrgWrapper import AvrgWrapper # 平均策略网络模型及其训练逻辑的封装器
from DeepCFR.workers.la.sampling_algorithms.MultiOutcomeSampler import MultiOutcomeSampler # 一种特定的结果采样算法，用于生成训练数据
# 从 PokerRL 框架导入基础类和工具
from PokerRL.rl import rl_util # RL 相关工具函数，如获取环境构建器
from PokerRL.rl.base_cls.workers.WorkerBase import WorkerBase # PokerRL 中 Worker 的基类

# 定义 LearnerActor 类，继承自 PokerRL 的 WorkerBase
class LearnerActor(WorkerBase):
    """
    LearnerActor (LA) 是 Deep CFR / SD-CFR 框架中的一个工作单元。
    主要负责：
    1. 使用当前的策略进行自我博弈，生成训练样本。
    2. 将样本存储在缓冲区 (Advantage Buffer, 可能还有 Average Buffer)。
    3. 管理本地的神经网络副本 (AdvWrapper, AvrgWrapper)。
    4. (可选) 计算网络梯度并发送给 Parameter Server (通常通过 Chief 协调)。
    """

    def __init__(self, t_prof, worker_id, chief_handle):
        """
        初始化 LearnerActor。
        Args:
            t_prof: 训练配置 (Training Profile) 对象。
            worker_id: 当前 LA 实例的唯一标识符。
            chief_handle: Chief 工作单元的 Ray 句柄，用于通信。
        """
        # 调用父类 WorkerBase 的初始化方法
        super().__init__(t_prof=t_prof)

        # 获取优势网络 (Advantage Network) 训练相关的参数
        self._adv_args = t_prof.module_args["adv_training"]

        # 获取环境构建器 (Environment Builder)，用于创建游戏环境实例
        self._env_bldr = rl_util.get_env_builder(t_prof=t_prof)
        # 保存 Worker ID 和 Chief 句柄
        self._id = worker_id
        self._chief_handle = chief_handle

        # --- 初始化优势网络 (Advantage Network) 相关组件 ---
        # 为游戏中的每个座位 (玩家) 创建一个优势缓冲区 (AdvReservoirBuffer)
        # 用于存储 (状态, 动作) -> 优势/遗憾 值的样本
        self._adv_buffers = [
            AdvReservoirBuffer(owner=p, # 指定缓冲区所属玩家 (座位号)
                               env_bldr=self._env_bldr, # 环境构建器
                               max_size=self._adv_args.max_buffer_size, # 缓冲区最大容量
                               nn_type=t_prof.nn_type, # 神经网络类型 (影响状态表示)
                               iter_weighting_exponent=self._t_prof.iter_weighting_exponent) # 迭代加权指数 (用于样本加权)
            for p in range(self._t_prof.n_seats) # n_seats: 游戏中的座位数
        ]

        # 为每个座位创建一个优势网络封装器 (AdvWrapper)
        # 封装了优势网络模型、优化器以及训练逻辑 (如计算损失、梯度)
        self._adv_wrappers = [
            AdvWrapper(owner=p, # 所属玩家
                       env_bldr=self._env_bldr, # 环境构建器
                       adv_training_args=self._adv_args, # 优势网络训练参数
                       device=self._adv_args.device_training) # 指定训练使用的设备 (CPU/GPU)
            for p in range(self._t_prof.n_seats)
        ]

        # --- 检查是否需要处理平均策略网络 (Average Strategy Network) ---
        # _AVRG 为 True 表示配置要求计算/评估平均策略网络
        self._AVRG = EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in self._t_prof.eval_modes_of_algo
        # _SINGLE 为 True 表示配置要求计算/评估当前策略网络 (基于优势网络)
        self._SINGLE = EvalAgentDeepCFR.EVAL_MODE_SINGLE in self._t_prof.eval_modes_of_algo

        # """"""""""""""""""""""""""""
        # 平均策略网络 (Average Strategy Network) 相关组件初始化 (如果需要)
        # """"""""""""""""""""""""""""
        if self._AVRG:
            # 如果需要处理平均策略网络
            print(f"LA {self._id}: 初始化平均策略网络 (AVRG) 组件...")
            # 获取平均策略网络训练相关的参数
            self._avrg_args = t_prof.module_args["avrg_training"]

            # 为每个座位创建一个平均策略缓冲区 (AvrgReservoirBuffer)
            # 用于存储 (状态) -> 平均策略目标值 的样本
            self._avrg_buffers = [
                AvrgReservoirBuffer(owner=p, # 所属玩家
                                    env_bldr=self._env_bldr, # 环境构建器
                                    max_size=self._avrg_args.max_buffer_size, # 缓冲区最大容量
                                    nn_type=t_prof.nn_type, # 神经网络类型
                                    iter_weighting_exponent=self._t_prof.iter_weighting_exponent) # 迭代加权指数
                for p in range(self._t_prof.n_seats)
            ]

            # 为每个座位创建一个平均策略网络封装器 (AvrgWrapper)
            # 封装了平均策略网络模型、优化器及训练逻辑
            # **SD-CFR 关联:** SD-CFR 旨在避免 *训练* 平均策略网络。这里的 AvrgWrapper 可能用于：
            # 1. 在 Deep CFR 模式下训练平均网络。
            # 2. 在 SD-CFR 模式下，根据历史策略 *计算* 平均策略并存储，用于评估。
            # 3. 完全不用，只是框架兼容性代码。具体行为取决于 AvrgWrapper 的实现和 t_prof 配置。
            self._avrg_wrappers = [
                AvrgWrapper(owner=p, # 所属玩家
                            env_bldr=self._env_bldr, # 环境构建器
                            avrg_training_args=self._avrg_args, # 平均策略网络训练参数
                            device=self._avrg_args.device_training) # 指定训练设备
                for p in range(self._t_prof.n_seats)
            ]

            # --- 初始化数据采样器 (包含 AVRG Buffers) ---
            # 根据配置选择采样算法，这里只支持 MultiOutcomeSampler ("mo")
            if self._t_prof.sampler.lower() == "mo":
                self._data_sampler = MultiOutcomeSampler(
                    env_bldr=self._env_bldr, # 环境构建器
                    adv_buffers=self._adv_buffers, # 传递优势缓冲区列表
                    avrg_buffers=self._avrg_buffers, # 传递平均策略缓冲区列表
                    n_actions_traverser_samples=self._t_prof.n_actions_traverser_samples # 遍历器采样动作数
                )
            else:
                raise ValueError(f"不支持的采样器: {self._t_prof.sampler.lower()}")
        else:
            # 如果不需要处理平均策略网络 (_AVRG is False)
            print(f"LA {self._id}: 不初始化平均策略网络 (AVRG) 组件 (可能是 SD-CFR 模式)。")
            # --- 初始化数据采样器 (不包含 AVRG Buffers) ---
            if self._t_prof.sampler.lower() == "mo":
                self._data_sampler = MultiOutcomeSampler(
                    env_bldr=self._env_bldr,
                    adv_buffers=self._adv_buffers,
                    avrg_buffers=None, # 不传递平均策略缓冲区
                    n_actions_traverser_samples=self._t_prof.n_actions_traverser_samples
                )
            else:
                raise ValueError(f"不支持的采样器: {self._t_prof.sampler.lower()}")

        # --- 初始化日志记录 (如果开启了详细日志) ---
        if self._t_prof.log_verbose:
            print(f"LA {self._id}: 设置详细日志...")
            # 通过 Chief 创建用于记录内存使用的实验 (Experiment)
            self._exp_mem_usage = self._ray.get( # 使用 ray.get 等待远程调用完成并获取结果
                self._ray.remote(self._chief_handle.create_experiment, # 远程调用 Chief 的 create_experiment
                                 self._t_prof.name + "_LA" + str(worker_id) + "_Memory_Usage") # 实验名称
            )
            # 为每个玩家的优势缓冲区大小创建实验
            self._exps_adv_buffer_size = self._ray.get(
                [ # 并行创建多个实验
                    self._ray.remote(self._chief_handle.create_experiment,
                                     self._t_prof.name + "_LA" + str(worker_id) + "_P" + str(p) + "_ADV_BufSize")
                    for p in range(self._t_prof.n_seats)
                ]
            )
            # 如果需要处理平均策略网络，也为平均策略缓冲区大小创建实验
            if self._AVRG:
                self._exps_avrg_buffer_size = self._ray.get(
                    [
                        self._ray.remote(self._chief_handle.create_experiment,
                                         self._t_prof.name + "_LA" + str(worker_id) + "_P" + str(p) + "_AVRG_BufSize")
                        for p in range(self._t_prof.n_seats)
                    ]
                )

    def generate_data(self, traverser, cfr_iter):
        """
        执行一轮数据生成过程。通过自我博弈填充缓冲区。
        Args:
            traverser (int): 当前进行遍历的玩家 (座位号)。CFR 算法通常会交替遍历玩家。
            cfr_iter (int): 当前的 CFR 迭代次数。
        """
        # print(f"LA {self._id}: 开始为玩家 {traverser} 在迭代 {cfr_iter} 生成数据...")
        # --- 准备当前迭代的策略 ---
        # 为每个玩家创建一个 IterationStrategy 对象
        # 它封装了该玩家在当前迭代 cfr_iter 中使用的策略网络 (基于当前的优势网络)
        iteration_strats = [
            IterationStrategy(t_prof=self._t_prof,       # 训练配置
                              env_bldr=self._env_bldr,   # 环境构建器
                              owner=p,                   # 策略所属玩家
                              device=self._t_prof.device_inference, # 指定推理设备
                              cfr_iter=cfr_iter)         # 当前迭代次数
            for p in range(self._t_prof.n_seats)
        ]
        # 从本地的 AdvWrapper 加载最新的优势网络参数到 IterationStrategy 中
        for s in iteration_strats:
            s.load_net_state_dict(state_dict=self._adv_wrappers[s.owner].net_state_dict())

        # --- 执行数据采样 ---
        # 调用数据采样器 (_data_sampler) 的 generate 方法
        # 该方法会模拟多次游戏过程 (traversals)
        self._data_sampler.generate(n_traversals=self._t_prof.n_traversals_per_iter, # 要进行的遍历次数
                                    traverser=traverser, # 当前遍历者
                                    iteration_strats=iteration_strats, # 所有玩家当前迭代的策略
                                    cfr_iter=cfr_iter, # 当前迭代次数
                                    )
        # 在 generate 内部，采样器会与环境交互，并将收集到的 (状态, 动作, 遗憾/策略) 等数据存入对应的缓冲区

        # --- 记录详细日志 (如果启用且满足条件) ---
        # 通常在所有玩家都完成数据生成后记录一次 (这里简单用 traverser == 1 判断)
        # 并且可能不需要每个迭代都记录，这里用了 cfr_iter % 3 == 0
        if self._t_prof.log_verbose and traverser == (self._t_prof.n_seats - 1) and (cfr_iter % 3 == 0):
             # print(f"LA {self._id}: 记录缓冲区大小和内存使用情况 (迭代 {cfr_iter})...")
             for p in range(self._t_prof.n_seats):
                 # 远程调用 Chief 的 add_scalar 方法，记录优势缓冲区大小
                 self._ray.remote(self._chief_handle.add_scalar,
                                  self._exps_adv_buffer_size[p], # 对应的实验句柄
                                  "Debug/BufferSize",          # TensorBoard 中的 Tag
                                  cfr_iter,                    # 全局步数 (迭代次数)
                                  self._adv_buffers[p].size)   # 要记录的值 (缓冲区大小)
                 # 如果处理平均策略网络，也记录其缓冲区大小
                 if self._AVRG:
                     self._ray.remote(self._chief_handle.add_scalar,
                                      self._exps_avrg_buffer_size[p], "Debug/BufferSize", cfr_iter,
                                      self._avrg_buffers[p].size)

             # 使用 psutil 获取当前进程的内存使用情况 (RSS: Resident Set Size)
             process = psutil.Process(os.getpid())
             mem_usage_mb = process.memory_info().rss / (1024 * 1024) # 转换为 MB
             # 远程调用 Chief 记录内存使用
             self._ray.remote(self._chief_handle.add_scalar,
                              self._exp_mem_usage, "Debug/MemoryUsage_MB/LA", cfr_iter, # Tag 修改为包含 MB
                              mem_usage_mb)

    def update(self, adv_state_dicts=None, avrg_state_dicts=None):
        """
        用从 Parameter Server (通过 Chief/Driver) 获取的新参数更新本地的网络副本。
        Args:
            adv_state_dicts (list): 优势网络的状态字典列表 (每个玩家一个，或为 None)。
                                     如果为 None，则不更新该玩家的网络。
                                     状态字典可能是 Ray 对象 ID，需要 ray.get() 获取。
            avrg_state_dicts (list): 平均策略网络的状态字典列表 (逻辑同上)。
        """
        # print(f"LA {self._id}: 更新本地网络副本...")
        for p_id in range(self._t_prof.n_seats):
            # --- 更新优势网络 ---
            if adv_state_dicts is not None and adv_state_dicts[p_id] is not None:
                # 从 Ray 对象存储中获取实际的状态字典，并转换为 PyTorch 张量
                state_dict = self._ray.state_dict_to_torch(self._ray.get(adv_state_dicts[p_id]),
                                                           device=self._adv_wrappers[p_id].device)
                # 调用 AdvWrapper 的方法加载新的状态字典
                self._adv_wrappers[p_id].load_net_state_dict(state_dict=state_dict)

            # --- 更新平均策略网络 (如果需要) ---
            if self._AVRG and avrg_state_dicts is not None and avrg_state_dicts[p_id] is not None:
                 state_dict = self._ray.state_dict_to_torch(self._ray.get(avrg_state_dicts[p_id]),
                                                            device=self._avrg_wrappers[p_id].device)
                 # 调用 AvrgWrapper 的方法加载新的状态字典
                 self._avrg_wrappers[p_id].load_net_state_dict(state_dict=state_dict)

    # --- Getter 方法 ---
    def get_loss_last_batch_adv(self, p_id):
        """获取指定玩家优势网络上一次训练批次的损失。"""
        return self._adv_wrappers[p_id].loss_last_batch

    def get_loss_last_batch_avrg(self, p_id):
        """获取指定玩家平均策略网络上一次训练批次的损失。"""
        if self._AVRG:
            return self._avrg_wrappers[p_id].loss_last_batch
        return None # 如果不处理 AVRG 网络，返回 None

    def get_adv_grads(self, p_id):
        """
        计算并返回指定玩家优势网络的一个训练批次的梯度。
        梯度通常会被发送到 Parameter Server 进行聚合和参数更新。
        """
        # 调用 AdvWrapper 的方法，从优势缓冲区采样一批数据并计算梯度
        grads = self._adv_wrappers[p_id].get_grads_one_batch_from_buffer(buffer=self._adv_buffers[p_id])
        # 将梯度转换为 NumPy 数组 (可能为了序列化或与 Ray 更好地交互)
        return self._ray.grads_to_numpy(grads)

    def get_avrg_grads(self, p_id):
        """
        计算并返回指定玩家平均策略网络的一个训练批次的梯度。
        **SD-CFR 关联:** 如果是在 SD-CFR 模式下运行，此方法可能：
        1. 不被调用。
        2. 被调用但内部实现返回 None 或零梯度。
        3. 用于计算与平均策略相关的其他量（如果 AvrgWrapper 被重用）。
        """
        if self._AVRG:
            # 调用 AvrgWrapper 的方法，从平均策略缓冲区采样一批数据并计算梯度
            grads = self._avrg_wrappers[p_id].get_grads_one_batch_from_buffer(buffer=self._avrg_buffers[p_id])
            return self._ray.grads_to_numpy(grads)
        return None # 如果不处理 AVRG 网络，返回 None

    # --- Checkpoint 相关方法 ---
    def checkpoint(self, curr_step):
        """保存 LearnerActor 的状态到检查点文件。"""
        print(f"LA {self._id}: 保存检查点 (迭代 {curr_step})...")
        for p_id in range(self._env_bldr.N_SEATS): # N_SEATS 应该等于 t_prof.n_seats
            # 构建要保存的状态字典
            state = {
                "adv_buffer": self._adv_buffers[p_id].state_dict(),     # 保存优势缓冲区的状态
                "adv_wrappers": self._adv_wrappers[p_id].state_dict(),  # 保存优势网络封装器的状态 (含模型权重、优化器状态等)
                "p_id": p_id, # 保存玩家 ID 用于验证
            }
            # 如果处理平均策略网络，也保存其状态
            if self._AVRG:
                state["avrg_buffer"] = self._avrg_buffers[p_id].state_dict()
                state["avrg_wrappers"] = self._avrg_wrappers[p_id].state_dict()

            # 获取检查点文件的路径 (方法来自 WorkerBase)
            # 文件名包含 LA ID 和玩家 ID，确保每个玩家的状态分开保存
            chkpt_file_path = self._get_checkpoint_file_path(name=self._t_prof.name, step=curr_step,
                                                             cls=self.__class__, worker_id=str(self._id) + "_P" + str(p_id))
            # 使用 pickle 将状态字典写入文件
            with open(chkpt_file_path, "wb") as pkl_file:
                pickle.dump(obj=state, file=pkl_file, protocol=pickle.HIGHEST_PROTOCOL) # 使用最高协议以获得更好性能和兼容性

    def load_checkpoint(self, name_to_load, step):
        """从检查点文件加载 LearnerActor 的状态。"""
        print(f"LA {self._id}: 加载检查点 '{name_to_load}' (迭代 {step})...")
        for p_id in range(self._env_bldr.N_SEATS):
            # 获取检查点文件的路径
            chkpt_file_path = self._get_checkpoint_file_path(name=name_to_load, step=step,
                                                             cls=self.__class__, worker_id=str(self._id) + "_P" + str(p_id))
            # 从文件读取 pickle 数据
            with open(chkpt_file_path, "rb") as pkl_file:
                state = pickle.load(pkl_file)

                # 验证加载的状态是否属于正确的玩家
                assert state["p_id"] == p_id, f"Checkpoint p_id mismatch! Expected {p_id}, got {state['p_id']}"

                # 加载缓冲区和网络封装器的状态
                self._adv_buffers[p_id].load_state_dict(state["adv_buffer"])
                self._adv_wrappers[p_id].load_state_dict(state["adv_wrappers"])
                # 如果处理平均策略网络，也加载其状态
                if self._AVRG:
                    # 检查点可能是在没有 AVRG 的情况下保存的
                    if "avrg_buffer" in state and "avrg_wrappers" in state:
                        self._avrg_buffers[p_id].load_state_dict(state["avrg_buffer"])
                        self._avrg_wrappers[p_id].load_state_dict(state["avrg_wrappers"])
                    else:
                         print(f"警告: 检查点中缺少玩家 {p_id} 的 AVRG 状态，但当前配置需要它。AVRG 组件可能未正确初始化。")