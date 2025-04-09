import os
import pickle
import ray
import time
import threading
import torch
import numpy as np
import copy

import psutil

from DeepCFR.IterationStrategy import IterationStrategy
from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.workers.la.buffers.AdvReservoirBuffer import AdvReservoirBuffer
from DeepCFR.workers.la.AdvWrapper import AdvWrapper
from DeepCFR.workers.la.buffers.AvrgReservoirBuffer import AvrgReservoirBuffer
from DeepCFR.workers.la.AvrgWrapper import AvrgWrapper
from DeepCFR.workers.la.sampling_algorithms.MultiOutcomeSampler import MultiOutcomeSampler
from PokerRL.rl import rl_util
from PokerRL.rl.base_cls.workers.WorkerBase import WorkerBase
from DeepCFR.workers.la.buffers._ReservoirBufferBase import ReservoirBufferBase as ReservoirBuffer
from PokerRL.rl.base_cls.workers.LearnerActorBase import LearnerActorBase


class LearnerActor(LearnerActorBase):

    def __init__(self, t_prof, worker_id, chief_handle):
        super().__init__(t_prof=t_prof, worker_id=worker_id, chief_handle=chief_handle)

        self._adv_args = t_prof.module_args["adv_training"]

        self._env_bldr = rl_util.get_env_builder(t_prof=t_prof)
        self._id = worker_id
        self._chief_handle = chief_handle

        self._adv_buffers = [
            AdvReservoirBuffer(owner=p, env_bldr=self._env_bldr, max_size=self._adv_args.max_buffer_size,
                               nn_type=t_prof.nn_type,
                               iter_weighting_exponent=self._t_prof.iter_weighting_exponent)
            for p in range(self._t_prof.n_seats)
        ]

        self._adv_wrappers = [
            AdvWrapper(owner=p,
                       env_bldr=self._env_bldr,
                       adv_training_args=self._adv_args,
                       device=self._adv_args.device_training)
            for p in range(self._t_prof.n_seats)
        ]

        self._AVRG = EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in self._t_prof.eval_modes_of_algo
        self._SINGLE = EvalAgentDeepCFR.EVAL_MODE_SINGLE in self._t_prof.eval_modes_of_algo

        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._AVRG:
            self._avrg_args = t_prof.module_args["avrg_training"]

            self._avrg_buffers = [
                AvrgReservoirBuffer(owner=p, env_bldr=self._env_bldr, max_size=self._avrg_args.max_buffer_size,
                                    nn_type=t_prof.nn_type,
                                    iter_weighting_exponent=self._t_prof.iter_weighting_exponent)
                for p in range(self._t_prof.n_seats)
            ]

            self._avrg_wrappers = [
                AvrgWrapper(owner=p,
                            env_bldr=self._env_bldr,
                            avrg_training_args=self._avrg_args,
                            device=self._avrg_args.device_training)
                for p in range(self._t_prof.n_seats)
            ]

            if self._t_prof.sampler.lower() == "mo":
                self._data_sampler = MultiOutcomeSampler(
                    env_bldr=self._env_bldr,
                    adv_buffers=self._adv_buffers,
                    avrg_buffers=self._avrg_buffers,
                    n_actions_traverser_samples=self._t_prof.n_actions_traverser_samples)
            else:
                raise ValueError("Currently we don't support", self._t_prof.sampler.lower(), "sampling.")
        else:
            if self._t_prof.sampler.lower() == "mo":
                self._data_sampler = MultiOutcomeSampler(
                    env_bldr=self._env_bldr,
                    adv_buffers=self._adv_buffers,
                    avrg_buffers=None,
                    n_actions_traverser_samples=self._t_prof.n_actions_traverser_samples)
            else:
                raise ValueError("Currently we don't support", self._t_prof.sampler.lower(), "sampling.")

        if self._t_prof.log_verbose:
            self._exp_mem_usage = self._ray.get(
                self._ray.remote(self._chief_handle.create_experiment,
                                 self._t_prof.name + "_LA" + str(worker_id) + "_Memory_Usage"))
            self._exps_adv_buffer_size = self._ray.get(
                [
                    self._ray.remote(self._chief_handle.create_experiment,
                                     self._t_prof.name + "_LA" + str(worker_id) + "_P" + str(p) + "_ADV_BufSize")
                    for p in range(self._t_prof.n_seats)
                ]
            )
            if self._AVRG:
                self._exps_avrg_buffer_size = self._ray.get(
                    [
                        self._ray.remote(self._chief_handle.create_experiment,
                                         self._t_prof.name + "_LA" + str(worker_id) + "_P" + str(p) + "_AVRG_BufSize")
                        for p in range(self._t_prof.n_seats)
                    ]
                )

        if self._t_prof.nn_type == "feedforward":
            # 从module_args中获取缓冲区大小，如果不存在则使用默认值
            max_buffer_size_adv = self._t_prof.module_args["adv_training"].max_buffer_size if hasattr(self._t_prof.module_args["adv_training"], "max_buffer_size") else 3e6
            max_buffer_size_avrg = self._t_prof.module_args["avrg_training"].max_buffer_size if hasattr(self._t_prof.module_args["avrg_training"], "max_buffer_size") else 3e6

            # 使用正确的参数创建ReservoirBuffer
            self._adv_buffer = AdvReservoirBuffer(
                owner=0,  # 默认owner为0
                env_bldr=self._env_bldr,
                max_size=int(max_buffer_size_adv),
                nn_type=self._t_prof.nn_type,
                iter_weighting_exponent=self._t_prof.iter_weighting_exponent
            )

            if self._AVRG:
                self._avrg_buffer = AvrgReservoirBuffer(
                    owner=0,  # 默认owner为0
                    env_bldr=self._env_bldr,
                    max_size=int(max_buffer_size_avrg),
                    nn_type=self._t_prof.nn_type,
                    iter_weighting_exponent=self._t_prof.iter_weighting_exponent
                )
        else:
            raise NotImplementedError

        # 注释掉可能导致错误的代码
        '''
        self._adv_net_optim = self._build_opt(self._adv_net)
        self._avrg_net_optim = self._build_opt(self._avrg_net)
        '''
        self._adv_net_optim = None
        self._avrg_net_optim = None

        # 用于记录上一次批次的损失
        self._loss_last_batch_adv = [None for _ in range(self._t_prof.n_seats)]
        self._loss_last_batch_avrg = [None for _ in range(self._t_prof.n_seats)]

        # --- 异步数据生成相关变量和控制线程 ---
        self._async_threads = {}  # 存储每个玩家的后台线程 {p_id: thread}
        self._stop_events = {}    # 各线程的停止信号 {p_id: Event}

        # 为每个生成线程维护一个 buffer
        self._async_adv_buffers = {}  # {p_id: [(sample, generation_cfr_iter), ...]}

        # 当前的模型权重
        self._current_adv_weights = [None for _ in range(self._t_prof.n_seats)]
        self._current_avrg_weights = [None for _ in range(self._t_prof.n_seats)]

        # 初始化PS句柄
        self._ps_handles = []

    @ray.method(num_returns=0)
    def generate_data(self, p_id, cfr_iter):
        """同步生成数据方法"""
        has_async = hasattr(self._t_prof, "use_async_data")
        use_async = self._t_prof.use_async_data if has_async else False
        if use_async:
            # 异步模式下，此方法不做任何事
            print(f"LA {self.worker_id}: generate_data 在异步模式下不做任何事")
            return

        # --- 原始同步数据生成逻辑 ---
        print(f"LA {self.worker_id}: 为玩家 {p_id} 生成数据，迭代 {cfr_iter}")

        # 生成优势样本
        if hasattr(self, "_sample_adv") and callable(self._sample_adv):
            samples = self._sample_adv(p_id=p_id, cfr_iter=cfr_iter)
            self._adv_buffer.add_samples(samples)

        # 生成平均策略样本
        if self._AVRG and hasattr(self, "_sample_avrg") and callable(self._sample_avrg):
            if cfr_iter > 0:  # 第一次迭代跳过
                samples = self._sample_avrg(p_id=p_id, cfr_iter=cfr_iter)
                self._avrg_buffer.add_samples(samples)

    # ================================
    # === 异步数据生成相关新增方法 ===
    # ================================

    @ray.method(num_returns=0)
    def start_background_generation(self, p_id, max_staleness):
        """开始后台数据生成线程"""
        has_async = hasattr(self._t_prof, "use_async_data")
        use_async = self._t_prof.use_async_data if has_async else False
        if not use_async:
            return

        # 如果已有该玩家的线程运行，先停止
        if p_id in self._async_threads and self._async_threads[p_id].is_alive():
            self.stop_background_generation(p_id)

        # 为该玩家创建停止信号和后台生成线程
        self._stop_events[p_id] = threading.Event()

        # 创建异步缓冲区
        if p_id not in self._async_adv_buffers:
            self._async_adv_buffers[p_id] = []

        # 创建并启动生成线程
        self._async_threads[p_id] = threading.Thread(
            target=self._background_generation_worker,
            args=(p_id, max_staleness),
            daemon=True
        )
        self._async_threads[p_id].start()
        print(f"LA {self.worker_id}: 已开始为玩家 {p_id} 的后台数据生成线程")

    @ray.method(num_returns=0)
    def stop_background_generation(self, p_id=None):
        """停止后台数据生成线程(一个或所有)"""
        if p_id is None:
            # 停止所有线程
            for pid in list(self._async_threads.keys()):
                self._stop_events[pid].set()

            # 等待所有线程结束
            for pid, thread in list(self._async_threads.items()):
                if thread.is_alive():
                    thread.join(timeout=1)  # 最多等待1秒

            # 清空字典
            self._async_threads.clear()
            self._stop_events.clear()
            print(f"LA {self.worker_id}: 已停止所有后台数据生成线程")
        else:
            # 停止特定玩家的线程
            if p_id in self._stop_events:
                self._stop_events[p_id].set()

                if p_id in self._async_threads and self._async_threads[p_id].is_alive():
                    self._async_threads[p_id].join(timeout=1)

                # 移除记录
                if p_id in self._async_threads:
                    del self._async_threads[p_id]
                if p_id in self._stop_events:
                    del self._stop_events[p_id]

                print(f"LA {self.worker_id}: 已停止玩家 {p_id} 的后台数据生成线程")

    @ray.method(num_returns=0)
    def update_background_generation_model(self, p_id):
        """更新后台生成线程使用的模型权重"""
        # 此方法在父类 update 更新完网络后被调用
        if p_id < len(self._current_adv_weights):
            # 为了明确记录，可以存储当前权重的副本
            try:
                # 使用_adv_wrappers而不是_adv_net
                self._current_adv_weights[p_id] = copy.deepcopy(self._adv_wrappers[p_id].state_dict())
                if self._AVRG:
                    self._current_avrg_weights[p_id] = copy.deepcopy(self._avrg_wrappers[p_id].state_dict())
            except Exception as e:
                print(f"LA {self.worker_id}: 无法更新玩家 {p_id} 的生成模型: {e}")

            print(f"LA {self.worker_id}: 已更新玩家 {p_id} 的生成模型")

    def _background_generation_worker(self, p_id, max_staleness):
        """后台数据生成工作线程函数"""
        try:
            print(f"LA {self.worker_id}: 玩家 {p_id} 的数据生成线程开始运行")
            # 从module_args中获取缓冲区大小，如果不存在则使用默认值
            max_buffer_size = int(self._t_prof.module_args["adv_training"].max_buffer_size) if hasattr(self._t_prof.module_args["adv_training"], "max_buffer_size") else int(3e6)

            while not self._stop_events[p_id].is_set():
                try:
                    # 获取当前迭代号 - 使用 self._ray.remote 包装
                    cfr_iter_ref = self._ray.remote(self._chief_handle.get_current_cfr_iter)
                    cfr_iter = self._ray.get(cfr_iter_ref)

                    # 检查异步缓冲区大小，如果已满则清理过期数据
                    if len(self._async_adv_buffers[p_id]) >= max_buffer_size:
                        # 筛选出不过期的数据
                        if max_staleness > 0:
                            self._async_adv_buffers[p_id] = [
                                (sample, gen_iter)
                                for sample, gen_iter in self._async_adv_buffers[p_id]
                                if cfr_iter - gen_iter <= max_staleness
                            ]

                        # 如果还是满的，随机丢弃一些
                        if len(self._async_adv_buffers[p_id]) >= max_buffer_size:
                            # 保留 80% 的数据
                            keep_size = int(max_buffer_size * 0.8)
                            indices = np.random.choice(
                                len(self._async_adv_buffers[p_id]),
                                size=keep_size,
                                replace=False
                            )
                            self._async_adv_buffers[p_id] = [
                                self._async_adv_buffers[p_id][i] for i in indices
                            ]

                    # 生成一小批数据(如 100 个样本)
                    batch_size = 100

                    # 为 ADV 网络生成样本
                    if hasattr(self, "_sample_adv") and callable(self._sample_adv):
                        adv_samples = self._sample_adv(p_id=p_id, cfr_iter=cfr_iter, n_samples=batch_size)

                        # 将样本与迭代号一起存储
                        for sample in adv_samples:
                            self._async_adv_buffers[p_id].append((sample, cfr_iter))

                        print(f"LA {self.worker_id}: 玩家 {p_id} 异步生成了 {len(adv_samples)} 个 ADV 样本，当前缓冲区大小 {len(self._async_adv_buffers[p_id])}")

                    # 为 AVRG 网络生成样本 (如果需要)
                    if self._AVRG and cfr_iter > 0 and hasattr(self, "_sample_avrg") and callable(self._sample_avrg):
                        # 这里需要实现 AVRG 样本生成逻辑
                        pass

                    # 避免 CPU 占用过高，短暂休眠
                    time.sleep(0.05)

                except Exception as e:
                    print(f"LA {self.worker_id}: 玩家 {p_id} 数据生成错误: {e}")
                    time.sleep(1)  # 出错后等待一段时间再重试

        except Exception as e:
            print(f"LA {self.worker_id}: 玩家 {p_id} 生成线程崩溃: {e}")
        finally:
            print(f"LA {self.worker_id}: 玩家 {p_id} 数据生成线程已退出")

    @ray.method(num_returns=1)
    def get_adv_buffer_size(self, p_id, current_cfr_iter=None):
        """获取异步缓冲区的有效大小，应用年龄筛选"""
        has_async = hasattr(self._t_prof, "use_async_data")
        use_async = self._t_prof.use_async_data if has_async else False
        if not use_async:
            return len(self._adv_buffer)  # 同步模式返回原始缓冲区大小

        if p_id not in self._async_adv_buffers:
            return 0

        max_staleness = self._t_prof.max_data_staleness

        if current_cfr_iter is None or max_staleness <= 0:
            return len(self._async_adv_buffers[p_id])
        else:
            # 计算非过期样本数量
            valid_count = sum(1 for _, gen_iter in self._async_adv_buffers[p_id]
                             if current_cfr_iter - gen_iter <= max_staleness)
            return valid_count

    @ray.method(num_returns=2)
    def get_adv_grads(self, p_id, cfr_iter=None):
        """计算优势网络的梯度"""
        has_async = hasattr(self._t_prof, "use_async_data")
        use_async = self._t_prof.use_async_data if has_async else False
        if not use_async:
            # 原始同步模式逻辑
            grads = self._ray.grads_to_numpy(
                self._adv_wrappers[p_id].get_grads_one_batch_from_buffer(buffer=self._adv_buffers[p_id]))
            return grads, self._adv_wrappers[p_id].loss_last_batch
        else:
            # 异步模式: 从异步缓冲区获取过滤后的样本批次
            if p_id not in self._async_adv_buffers or not self._async_adv_buffers[p_id]:
                print(f"LA {self.worker_id}: 警告 - 玩家 {p_id} 没有异步样本数据")
                # 返回空梯度
                self._loss_last_batch_adv[p_id] = 0
                return [], 0

            # 筛选有效样本
            valid_samples = []
            if cfr_iter is not None and self._t_prof.max_data_staleness > 0:
                valid_samples = [
                    sample for sample, gen_iter in self._async_adv_buffers[p_id]
                    if cfr_iter - gen_iter <= self._t_prof.max_data_staleness
                ]
            else:
                valid_samples = [sample for sample, _ in self._async_adv_buffers[p_id]]

            if len(valid_samples) < self._t_prof.mini_batch_size_adv:
                print(f"LA {self.worker_id}: 警告 - 玩家 {p_id} 没有足够的有效样本 ({len(valid_samples)} < {self._t_prof.mini_batch_size_adv})")
                batch_size = max(len(valid_samples) // 2, 1)
            else:
                batch_size = self._t_prof.mini_batch_size_adv

            # 从有效样本中选择批次
            indices = np.random.choice(len(valid_samples), min(batch_size, len(valid_samples)), replace=False)
            batch = [valid_samples[i] for i in indices]

            # 将批次添加到临时缓冲区并计算梯度
            temp_buffer = AdvReservoirBuffer(
                owner=p_id,
                env_bldr=self._env_bldr,
                max_size=len(batch) + 10,  # 留一些额外空间
                nn_type=self._t_prof.nn_type,
                iter_weighting_exponent=self._t_prof.iter_weighting_exponent
            )

            for sample_data in batch:
                # 如果是元组，解包样本，元组样本包含(数据, 生成的CFR迭代号)
                if isinstance(sample_data, tuple) and len(sample_data) == 2:
                    sample = sample_data[0]
                else:
                    sample = sample_data

                # 如果样本是字典形式，直接用关键字参数添加
                if isinstance(sample, dict):
                    # 确保样本包含所有需要的关键字
                    if all(k in sample for k in ["pub_obs", "range_idx", "legal_action_mask", "adv", "iteration"]):
                        temp_buffer.add(**sample)
                    else:
                        print(f"警告: 样本缺少必需字段: {sample.keys()}")
                else:
                    # 如果不是字典形式，尝试直接添加
                    try:
                        temp_buffer.add(sample)
                    except Exception as e:
                        print(f"错误: 处理样本失败 - {e}")

            # 计算梯度
            grads = self._ray.grads_to_numpy(
                self._adv_wrappers[p_id].get_grads_one_batch_from_buffer(buffer=temp_buffer))

            self._loss_last_batch_adv[p_id] = self._adv_wrappers[p_id].loss_last_batch
            return grads, self._adv_wrappers[p_id].loss_last_batch

    @ray.method(num_returns=2)
    def get_avrg_grads(self, p_id, cfr_iter=None):
        """计算平均策略网络的梯度，兼容异步模式"""
        # 检查t_prof有没有use_async_data属性
        has_async = hasattr(self._t_prof, "use_async_data")
        use_async = self._t_prof.use_async_data if has_async else False

        if not use_async or not self._AVRG:
            # 原始同步逻辑
            return self._ray.grads_to_numpy(
                self._avrg_wrappers[p_id].get_grads_one_batch_from_buffer(buffer=self._avrg_buffers[p_id])), \
                self._avrg_wrappers[p_id].loss_last_batch
        else:
            # 异步模式的平均策略网络梯度计算逻辑可以根据需要添加
            # 目前暂时使用同步模式的逻辑
            return self._ray.grads_to_numpy(
                self._avrg_wrappers[p_id].get_grads_one_batch_from_buffer(buffer=self._avrg_buffers[p_id])), \
                self._avrg_wrappers[p_id].loss_last_batch

    @ray.method(num_returns=0)
    def update(self, adv_state_dicts=None, avrg_state_dicts=None):
        """
        Args:
            adv_state_dicts (list):         Optional. if not None:
                                                      expects a list of neural net state dicts or None for each player
                                                      in order of their seat_ids. This allows updating only some
                                                      players.

            avrg_state_dicts (list):         Optional. if not None:
                                                      expects a list of neural net state dicts or None for each player
                                                      in order of their seat_ids. This allows updating only some
                                                      players.
        """
        for p_id in range(self._t_prof.n_seats):
            if adv_state_dicts[p_id] is not None:
                self._adv_wrappers[p_id].load_net_state_dict(
                    state_dict=self._ray.state_dict_to_torch(self._ray.get(adv_state_dicts[p_id]),
                                                             device=self._adv_wrappers[p_id].device))

            if avrg_state_dicts[p_id] is not None:
                self._avrg_wrappers[p_id].load_net_state_dict(
                    state_dict=self._ray.state_dict_to_torch(self._ray.get(avrg_state_dicts[p_id]),
                                                             device=self._avrg_wrappers[p_id].device))

    @ray.method(num_returns=1)
    def get_loss_last_batch_adv(self, p_id):
        return self._loss_last_batch_adv[p_id]

    @ray.method(num_returns=1)
    def get_loss_last_batch_avrg(self, p_id):
        return self._loss_last_batch_avrg[p_id]

    @ray.method(num_returns=0)
    def checkpoint(self, curr_step):
        for p_id in range(self._env_bldr.N_SEATS):
            state = {
                "adv_buffer": self._adv_buffers[p_id].state_dict(),
                "adv_wrappers": self._adv_wrappers[p_id].state_dict(),
                "p_id": p_id,
            }
            if self._AVRG:
                state["avrg_buffer"] = self._avrg_buffers[p_id].state_dict()
                state["avrg_wrappers"] = self._avrg_wrappers[p_id].state_dict()

            with open(self._get_checkpoint_file_path(name=self._t_prof.name, step=curr_step,
                                                     cls=self.__class__, worker_id=str(self._id) + "_P" + str(p_id)),
                      "wb") as pkl_file:
                pickle.dump(obj=state, file=pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

    @ray.method(num_returns=0)
    def load_checkpoint(self, name_to_load, step):
        for p_id in range(self._env_bldr.N_SEATS):
            with open(self._get_checkpoint_file_path(name=name_to_load, step=step,
                                                     cls=self.__class__, worker_id=str(self._id) + "_P" + str(p_id)),
                      "rb") as pkl_file:
                state = pickle.load(pkl_file)

                assert state["p_id"] == p_id

                self._adv_buffers[p_id].load_state_dict(state["adv_buffer"])
                self._adv_wrappers[p_id].load_state_dict(state["adv_wrappers"])
                if self._AVRG:
                    self._avrg_buffers[p_id].load_state_dict(state["avrg_buffer"])
                    self._avrg_wrappers[p_id].load_state_dict(state["avrg_wrappers"])

    # 添加缺失的采样方法
    @ray.method(num_returns=1)
    def _sample_adv(self, p_id, cfr_iter, n_samples=None):
        """生成优势网络训练样本"""
        if n_samples is None:
            n_samples = self._t_prof.n_traversals_per_iter // self._t_prof.n_seats

        # 准备策略列表 - 直接使用包装器
        iteration_strats = []
        for s in range(self._t_prof.n_seats):
            iteration_strats.append(self._adv_wrappers[s])

        # 直接调用 MultiOutcomeSampler.generate 方法
        self._data_sampler.generate(
            n_traversals=n_samples,
            traverser=p_id,
            iteration_strats=iteration_strats,
            cfr_iter=cfr_iter
        )

        # 从优势缓冲区获取样本（generate 会填充它）
        adv_samples = self._adv_buffers[p_id].get_all()
        return adv_samples

    @ray.method(num_returns=1)
    def _sample_avrg(self, p_id, cfr_iter, n_samples=None):
        """生成平均策略网络训练样本"""
        if n_samples is None:
            n_samples = self._t_prof.n_traversals_per_iter // self._t_prof.n_seats

        # 准备策略列表 - AVRG 模式可能需要不同的策略？
        # 这里仍然使用 adv_wrappers，如果需要平均策略，需要调整
        iteration_strats = []
        for s in range(self._t_prof.n_seats):
            # 注意：这里可能需要传递 avrg_wrappers 如果 sampler 需要它
            iteration_strats.append(self._adv_wrappers[s])

        # 直接调用 MultiOutcomeSampler.generate 方法
        self._data_sampler.generate(
            n_traversals=n_samples,
            traverser=p_id,
            iteration_strats=iteration_strats,
            cfr_iter=cfr_iter
        )

        # 从平均缓冲区获取样本
        avrg_samples = []
        if self._AVRG and self._avrg_buffers and p_id < len(self._avrg_buffers):
            avrg_samples = self._avrg_buffers[p_id].get_all()
        return avrg_samples

    # 添加PS句柄设置方法
    @ray.method(num_returns=0)
    def set_ps_handles(self, *ps_handles):
        """设置参数服务器句柄"""
        self._ps_handles = list(ps_handles)

    @property
    def chief_handle(self):
        """获取Chief句柄"""
        return self._chief_handle

    @ray.method(num_returns=1)
    def get_data_generation_status(self, p_id):
        """获取数据生成状态"""
        if p_id not in self._async_threads:
            return {"active": False, "buffer_size": 0}

        return {
            "active": self._async_threads[p_id].is_alive(),
            "buffer_size": len(self._async_adv_buffers.get(p_id, []))
        }

    # 添加_build_opt方法
    def _build_opt(self, net):
        """
        构建网络优化器

        Args:
            net: 要优化的网络模型

        Returns:
            torch优化器实例
        """
        if net is None:
            return None

        try:
            import torch.optim as optim
            return optim.Adam(net.parameters(), lr=0.001)
        except Exception as e:
            print(f"创建优化器失败: {e}")
            return None
