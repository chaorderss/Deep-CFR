import os
import torch
from torch.utils.tensorboard import SummaryWriter

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.workers.driver._HighLevelAlgo import HighLevelAlgo
from PokerRL.rl.base_cls.workers.DriverBase import DriverBase


class Driver(DriverBase):

    _CHIEF_CLS = None  # 将在初始化时设置
    _EVAL_AGENT_CLS = EvalAgentDeepCFR

    def __init__(self, t_prof, eval_methods, n_iterations=None, iteration_to_import=None, name_to_import=None):
        if t_prof.DISTRIBUTED:
            from DeepCFR.workers.chief.dist import Chief
            from DeepCFR.workers.la.dist import LearnerActor
            from DeepCFR.workers.ps.dist import ParameterServer

        else:
            from DeepCFR.workers.chief.local import Chief
            from DeepCFR.workers.la.local import LearnerActor
            from DeepCFR.workers.ps.local import ParameterServer

        # 设置Chief类
        Driver._CHIEF_CLS = Chief

        # 初始化DriverBase，但不创建CrayonWrapper
        super().__init__(t_prof=t_prof, eval_methods=eval_methods, n_iterations=n_iterations,
                         iteration_to_import=iteration_to_import, name_to_import=name_to_import,
                         chief_cls=Chief, eval_agent_cls=EvalAgentDeepCFR)

        # 替换CrayonWrapper为TensorBoard
        if hasattr(self, 'crayon'):
            del self.crayon

        # 设置TensorBoard日志目录
        tb_log_dir = os.path.join("logs", t_prof.name, 'tensorboard')
        os.makedirs(tb_log_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"TensorBoard日志将保存在: {tb_log_dir}")

        if "h2h" in list(eval_methods.keys()):
            assert EvalAgentDeepCFR.EVAL_MODE_SINGLE in t_prof.eval_modes_of_algo
            assert EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in t_prof.eval_modes_of_algo
            self._ray.remote(self.eval_masters["h2h"][0].set_modes,
                             [EvalAgentDeepCFR.EVAL_MODE_SINGLE, EvalAgentDeepCFR.EVAL_MODE_AVRG_NET]
                             )

        print("Creating LAs...")
        self.la_handles = [
            self._ray.create_worker(LearnerActor,
                                    t_prof,
                                    i,
                                    self.chief_handle)
            for i in range(t_prof.n_learner_actors)
        ]

        print("Creating Parameter Servers...")
        self.ps_handles = [
            self._ray.create_worker(ParameterServer,
                                    t_prof,
                                    p,
                                    self.chief_handle)
            for p in range(t_prof.n_seats)
        ]

        self._ray.wait([
            self._ray.remote(self.chief_handle.set_ps_handle,
                             *self.ps_handles),
            self._ray.remote(self.chief_handle.set_la_handles,
                             *self.la_handles)
        ])

        print("Created and initialized Workers")

        self.algo = HighLevelAlgo(t_prof=t_prof,
                                  la_handles=self.la_handles,
                                  ps_handles=self.ps_handles,
                                  chief_handle=self.chief_handle)

        self._AVRG = EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in self._t_prof.eval_modes_of_algo
        self._SINGLE = EvalAgentDeepCFR.EVAL_MODE_SINGLE in self._t_prof.eval_modes_of_algo

        self._maybe_load_checkpoint_init()

    def save_logs(self):
        """
        重写保存日志的方法，使用TensorBoard代替CrayonWrapper
        并且处理可能缺少get_log_buffer方法的情况
        """
        try:
            # 尝试从Chief获取日志 - 正常情况
            if hasattr(self.chief_handle, 'get_log_buffer'):
                logs = self._ray.get(self._ray.remote(self.chief_handle.get_log_buffer))

                # 记录到TensorBoard
                if logs:
                    for log_group, entries in logs.items():
                        for tag, value_list in entries.items():
                            for value in value_list:
                                # value格式为[时间戳, 值]
                                self.tb_writer.add_scalar(f"{log_group}/{tag}", value[1], global_step=self._cfr_iter)

                    # 清空LogBuffer
                    if hasattr(self.chief_handle, 'flush_log_buffer'):
                        self._ray.wait([self._ray.remote(self.chief_handle.flush_log_buffer)])
            else:
                # Chief没有get_log_buffer方法，尝试使用_fetch_logged_data方法
                self._fetch_logs_from_chief()

        except Exception as e:
            print(f"日志获取错误: {e}")
            # 捕获错误但继续执行，不中断训练过程
            # 可能的替代方案：直接从workers获取日志
            self._fetch_logs_from_workers()

        # 立即将日志写入磁盘
        self.tb_writer.flush()

        # 删除过去的日志文件（保持原始功能）
        s = [self._cfr_iter]
        if self._cfr_iter > self._t_prof.log_export_freq + 1:
            s.append(self._cfr_iter - self._t_prof.log_export_freq)

        self._delete_past_log_files(steps_not_to_delete=s)

    def _fetch_logs_from_chief(self):
        """尝试使用可能的替代方法从Chief获取日志"""
        # 检查Chief是否有其他可能的日志相关方法
        possible_methods = ['_get_log_buffer', 'get_logs', 'get_logged_data', 'all_log_data']

        for method_name in possible_methods:
            if hasattr(self.chief_handle, method_name):
                try:
                    logs = self._ray.get(self._ray.remote(getattr(self.chief_handle, method_name)))

                    if logs and isinstance(logs, dict):
                        for log_group, entries in logs.items():
                            for tag, value_list in entries.items():
                                if isinstance(value_list, list):
                                    for value in value_list:
                                        if isinstance(value, (list, tuple)) and len(value) >= 2:
                                            self.tb_writer.add_scalar(f"{log_group}/{tag}", value[1], global_step=self._cfr_iter)
                    return
                except:
                    continue

    def _fetch_logs_from_workers(self):
        """直接从LAs和PS获取日志数据"""
        # 从工作节点直接获取性能指标
        for w_idx, worker in enumerate(self.la_handles):
            try:
                # 尝试获取工作节点的训练性能指标
                metrics = self._ray.get(self._ray.remote(worker.get_performance_metrics))
                if metrics and isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        self.tb_writer.add_scalar(f"worker_{w_idx}/{metric_name}", value, global_step=self._cfr_iter)
            except:
                pass

    def run(self):
        print("Setting stuff up...")

        # """"""""""""""""
        # Init globally
        # """"""""""""""""
        self.algo.init()

        print("Starting Training...")
        for _iter_nr in range(10000000 if self.n_iterations is None else self.n_iterations):
            print("Iteration: ", self._cfr_iter)

            # """"""""""""""""
            # Maybe train AVRG
            # """"""""""""""""
            avrg_times = None
            if self._AVRG and self._any_eval_needs_avrg_net():
                avrg_times = self.algo.train_average_nets(cfr_iter=_iter_nr)

            # """"""""""""""""
            # Eval
            # """"""""""""""""
            # Evaluate. Sync & Lock, then train while evaluating on other workers
            self.evaluate()

            # """"""""""""""""
            # Log
            # """"""""""""""""
            if self._cfr_iter % self._t_prof.log_export_freq == 0:
                self.save_logs()
            self.periodically_export_eval_agent()

            # """"""""""""""""
            # Iteration
            # """"""""""""""""
            iter_times = self.algo.run_one_iter_alternating_update(cfr_iter=self._cfr_iter)

            # 直接记录性能指标到TensorBoard - 这些是直接从算法获取的，不需要通过Chief
            self.tb_writer.add_scalar("performance/data_generation_time", iter_times["t_generating_data"], self._cfr_iter)
            self.tb_writer.add_scalar("performance/adv_training_time", iter_times["t_computation_adv"], self._cfr_iter)
            self.tb_writer.add_scalar("performance/adv_syncing_time", iter_times["t_syncing_adv"], self._cfr_iter)

            print(
                "Generating Data: ", str(iter_times["t_generating_data"]) + "s.",
                "  ||  Trained ADV", str(iter_times["t_computation_adv"]) + "s.",
                "  ||  Synced ADV", str(iter_times["t_syncing_adv"]) + "s.",
                "\n"
            )
            if self._AVRG and avrg_times:
                # 记录AVRG网络性能指标
                self.tb_writer.add_scalar("performance/avrg_training_time", avrg_times["t_computation_avrg"], self._cfr_iter)
                self.tb_writer.add_scalar("performance/avrg_syncing_time", avrg_times["t_syncing_avrg"], self._cfr_iter)

                print(
                    "Trained AVRG", str(avrg_times["t_computation_avrg"]) + "s.",
                    "  ||  Synced AVRG", str(avrg_times["t_syncing_avrg"]) + "s.",
                    "\n"
                )

            self._cfr_iter += 1

            # """"""""""""""""
            # Checkpoint
            # """"""""""""""""
            self.periodically_checkpoint()

        # 返回最终结果
        return {
            "iterations_completed": self._cfr_iter,
            "checkpoint_path": os.path.join(self._t_prof.path_checkpoint, self._t_prof.name, str(self._cfr_iter-1))
        }

    def _any_eval_needs_avrg_net(self):
        for e in list(self.eval_masters.values()):
            if self._cfr_iter % e[1] == 0:
                return True
        return False

    def checkpoint(self, **kwargs):
        # Call on all other workers sequentially to be safe against RAM overload
        for w in self.la_handles + self.ps_handles + [self.chief_handle]:
            self._ray.wait([
                self._ray.remote(w.checkpoint,
                                 self._cfr_iter)
            ])

        # Delete past checkpoints
        s = [self._cfr_iter]
        if self._cfr_iter > self._t_prof.checkpoint_freq + 1:
            s.append(self._cfr_iter - self._t_prof.checkpoint_freq)

        self._delete_past_checkpoints(steps_not_to_delete=s)

    def load_checkpoint(self, step, name_to_load):
        # Call on all other workers sequentially to be safe against RAM overload
        for w in self.la_handles + self.ps_handles + [self.chief_handle]:
            self._ray.wait([
                self._ray.remote(w.load_checkpoint,
                                 name_to_load, step)
            ])

    def __del__(self):
        """确保在对象销毁时关闭TensorBoard writer"""
        if hasattr(self, 'tb_writer'):
            try:
                self.tb_writer.close()
                print("TensorBoard writer已关闭")
            except:
                pass
