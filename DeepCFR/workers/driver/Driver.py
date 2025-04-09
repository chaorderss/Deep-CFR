from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.workers.driver._HighLevelAlgo import HighLevelAlgo
from PokerRL.rl.base_cls.workers.DriverBase import DriverBase
import signal
import sys


class Driver(DriverBase):

    def __init__(self, t_prof, eval_methods, n_iterations=None, iteration_to_import=None, name_to_import=None):
        if t_prof.DISTRIBUTED:
            from DeepCFR.workers.chief.dist import Chief
            from DeepCFR.workers.la.dist import LearnerActor
            from DeepCFR.workers.ps.dist import ParameterServer

        else:
            from DeepCFR.workers.chief.local import Chief
            from DeepCFR.workers.la.local import LearnerActor
            from DeepCFR.workers.ps.local import ParameterServer

        super().__init__(t_prof=t_prof, eval_methods=eval_methods, n_iterations=n_iterations,
                         iteration_to_import=iteration_to_import, name_to_import=name_to_import,
                         chief_cls=Chief, eval_agent_cls=EvalAgentDeepCFR)

        if "h2h" in list(eval_methods.keys()):
            assert EvalAgentDeepCFR.EVAL_MODE_SINGLE in t_prof.eval_modes_of_algo
            assert EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in t_prof.eval_modes_of_algo
            self.eval_masters["h2h"][0].set_modes.remote([EvalAgentDeepCFR.EVAL_MODE_SINGLE, EvalAgentDeepCFR.EVAL_MODE_AVRG_NET])

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

        # 检查是否处于分布式模式
        if hasattr(self._ray, 'runs_distributed') and self._ray.runs_distributed:
            # 分布式模式，直接使用remote调用
            self._ray.wait([
                self.chief_handle.set_ps_handle.remote(*self.ps_handles),
                self.chief_handle.set_la_handles.remote(*self.la_handles)
            ])
        else:
            # 非分布式模式，直接调用函数
            self.chief_handle.set_ps_handle(*self.ps_handles)
            self.chief_handle.set_la_handles(*self.la_handles)

        print("Created and initialized Workers")

        self.algo = HighLevelAlgo(t_prof=t_prof,
                                  la_handles=self.la_handles,
                                  ps_handles=self.ps_handles,
                                  chief_handle=self.chief_handle)

        self._AVRG = EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in self._t_prof.eval_modes_of_algo
        self._SINGLE = EvalAgentDeepCFR.EVAL_MODE_SINGLE in self._t_prof.eval_modes_of_algo

        self._maybe_load_checkpoint_init()

        # 注册信号处理，确保程序被中断时也能清理资源
        if self._t_prof.use_async_data:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        """处理程序中断信号，确保正确停止后台线程"""
        print("\n接收到中断信号，正在停止后台数据生成...")
        self._stop_background_generation()
        sys.exit(0)

    def _stop_background_generation(self):
        """停止所有LearnerActor上的后台数据生成线程"""
        if hasattr(self, 'la_handles') and self._t_prof.use_async_data:
            print("停止后台数据生成线程...")
            for la in self.la_handles:
                if hasattr(self._ray, 'runs_distributed') and self._ray.runs_distributed:
                    # 分布式模式
                    la.stop_background_generation.remote()
                else:
                    # 非分布式模式
                    la.stop_background_generation()
            print("后台数据生成线程已停止")

    def run(self):
        print("Setting stuff up...")

        # """"""""""""""""
        # Init globally
        # """"""""""""""""
        self.algo.init()

        print("Starting Training...")
        try:
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

                print(
                    "Generating Data: ", str(iter_times["t_generating_data"]) + "s.",
                    "  ||  Trained ADV", str(iter_times["t_computation_adv"]) + "s.",
                    "  ||  Synced ADV", str(iter_times["t_syncing_adv"]) + "s.",
                    "\n"
                )
                if self._AVRG and avrg_times:
                    print(
                        "Trained AVRG", str(avrg_times["t_computation_avrg"]) + "s.",
                        "  ||  Synced AVRG", str(avrg_times["t_syncing_avrg"]) + "s.",
                        "\n"
                    )

                self._cfr_iter += 1

                # 更新 Chief 的迭代计数器
                if hasattr(self._ray, 'runs_distributed') and self._ray.runs_distributed:
                    self._ray.remote(self.chief_handle.set_current_cfr_iter, self._cfr_iter)
                else:
                    self.chief_handle.set_current_cfr_iter(self._cfr_iter)

                # """"""""""""""""
                # Checkpoint
                # """"""""""""""""
                self.periodically_checkpoint()

        finally:
            # 确保在训练完成或异常退出时停止后台线程
            if self._t_prof.use_async_data:
                self._stop_background_generation()

    def _any_eval_needs_avrg_net(self):
        for e in list(self.eval_masters.values()):
            if self._cfr_iter % e[1] == 0:
                return True
        return False

    def checkpoint(self, **kwargs):
        # Call on all other workers sequentially to be safe against RAM overload
        for w in self.la_handles + self.ps_handles + [self.chief_handle]:
            self._ray.wait([
                w.checkpoint.remote(self._cfr_iter)
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
                w.load_checkpoint.remote(name_to_load, step)
            ])

    def __del__(self):
        """确保在对象销毁时也清理资源"""
        # 停止后台数据生成线程
        if hasattr(self, '_t_prof') and self._t_prof.use_async_data:
            self._stop_background_generation()
