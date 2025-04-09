import time

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from PokerRL.rl.base_cls.HighLevelAlgoBase import HighLevelAlgoBase as _HighLevelAlgoBase


class HighLevelAlgo(_HighLevelAlgoBase):

    def __init__(self, t_prof, la_handles, ps_handles, chief_handle):
        super().__init__(t_prof=t_prof, chief_handle=chief_handle, la_handles=la_handles)
        self._ps_handles = ps_handles
        self._all_p_aranged = list(range(self._t_prof.n_seats))

        self._AVRG = EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in self._t_prof.eval_modes_of_algo
        self._SINGLE = EvalAgentDeepCFR.EVAL_MODE_SINGLE in self._t_prof.eval_modes_of_algo

        self._adv_args = t_prof.module_args["adv_training"]
        if self._AVRG:
            self._avrg_args = t_prof.module_args["avrg_training"]

    def init(self):
        # """"""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""
        if self._AVRG:
            self._update_leaner_actors(update_adv_for_plyrs=self._all_p_aranged,
                                       update_avrg_for_plyrs=self._all_p_aranged)

        # """"""""""""""""""""""
        # NOT Deep CFR
        # """"""""""""""""""""""
        else:
            self._update_leaner_actors(update_adv_for_plyrs=self._all_p_aranged)

        # 如果启用异步数据生成，则在初始化时开始后台生成
        if self._t_prof.use_async_data:
            print("开始异步数据生成...")
            self._start_async_data_generation()

    def _start_async_data_generation(self):
        """开始所有Learner Actor上的异步数据生成"""
        # 检查是否为分布式模式
        is_distributed = hasattr(self._ray, 'runs_distributed') and self._ray.runs_distributed

        for p_id in range(self._t_prof.n_seats):
            for la in self._la_handles:
                # 开始该玩家的异步数据生成
                if is_distributed:
                    self._ray.wait([la.start_background_generation.remote(p_id, self._t_prof.max_data_staleness)])
                else:
                    la.start_background_generation(p_id, self._t_prof.max_data_staleness)

    def run_one_iter_alternating_update(self, cfr_iter):
        t_generating_data = 0.0
        t_computation_adv = 0.0
        t_syncing_adv = 0.0

        for p_learning in range(self._t_prof.n_seats):
            self._update_leaner_actors(update_adv_for_plyrs=self._all_p_aranged)

            if not self._t_prof.use_async_data:
                # 同步数据生成模式
                print("生成数据（同步模式）...")
                t0 = time.time()
                self._generate_traversals(p_id=p_learning, cfr_iter=cfr_iter)
                t_generating_data += time.time() - t0
            else:
                # 异步数据生成模式 - 检查缓冲区状态
                print(f"检查玩家 {p_learning} 的数据缓冲区状态（异步模式）...")
                t0 = time.time()
                # 向所有 Learner Actors 查询缓冲区大小并求和
                buffer_size_refs = [
                    self._ray.remote(la.get_adv_buffer_size, p_learning, cfr_iter)
                    for la in self._la_handles
                ]
                buffer_sizes = self._ray.get(buffer_size_refs)
                buffer_size = sum(buffer_sizes)
                print(f"玩家 {p_learning} 的总可用数据量: {buffer_size}")
                t_generating_data += time.time() - t0

                # 如果数据不足，可能需要等待
                if buffer_size < self._t_prof.min_data_for_training:
                    wait_time = 0
                    max_wait = 10  # 最多等待10秒
                    print(f"数据不足，等待收集更多样本...")
                    while buffer_size < self._t_prof.min_data_for_training and wait_time < max_wait:
                        time.sleep(1)
                        wait_time += 1
                        # 再次向所有 Learner Actors 查询缓冲区大小并求和
                        buffer_size_refs = [
                            self._ray.remote(la.get_adv_buffer_size, p_learning, cfr_iter)
                            for la in self._la_handles
                        ]
                        buffer_sizes = self._ray.get(buffer_size_refs)
                        buffer_size = sum(buffer_sizes)
                        print(f"等待 {wait_time}秒后，玩家 {p_learning} 的总可用数据量: {buffer_size}")

                    if buffer_size < self._t_prof.min_data_for_training:
                        print(f"警告：等待 {wait_time}秒后，玩家 {p_learning} 的数据量仍不足训练要求")
                        # 这里可以选择跳过训练或降低批次大小等策略

            print("训练优势网络...")
            _t_computation_adv, _t_syncing_adv = self._train_adv(p_id=p_learning, cfr_iter=cfr_iter)
            t_computation_adv += _t_computation_adv
            t_syncing_adv += _t_syncing_adv

            if self._SINGLE:
                print("将新网络推送到chief...")
                self._push_newest_adv_net_to_chief(p_id=p_learning, cfr_iter=cfr_iter)

        print("同步中...")
        self._update_leaner_actors(update_adv_for_plyrs=self._all_p_aranged)

        return {
            "t_generating_data": t_generating_data,
            "t_computation_adv": t_computation_adv,
            "t_syncing_adv": t_syncing_adv,
        }

    def train_average_nets(self, cfr_iter):
        print("训练平均策略网络...")
        t_computation_avrg = 0.0
        t_syncing_avrg = 0.0
        for p in range(self._t_prof.n_seats):
            _c, _s = self._train_avrg(p_id=p, cfr_iter=cfr_iter)
            t_computation_avrg += _c
            t_syncing_avrg += _s

        return {
            "t_computation_avrg": t_computation_avrg,
            "t_syncing_avrg": t_syncing_avrg,
        }

    def _train_adv(self, p_id, cfr_iter):
        t_computation = 0.0
        t_syncing = 0.0

        # 检查是否有足够数据进行训练（仅异步模式）
        if self._t_prof.use_async_data:
            # 向所有 Learner Actors 查询缓冲区大小并求和
            buffer_size_refs = [
                self._ray.remote(la.get_adv_buffer_size, p_id, cfr_iter)
                for la in self._la_handles
            ]
            buffer_sizes = self._ray.get(buffer_size_refs)
            buffer_size = sum(buffer_sizes)
            if buffer_size < self._t_prof.min_data_for_training:
                print(f"跳过玩家 {p_id} 的优势网络训练: 数据不足 ({buffer_size} < {self._t_prof.min_data_for_training})")
                return 0.0, 0.0

        # For logging the loss to see convergence in Tensorboard
        if self._t_prof.log_verbose:
            exp_loss_each_p = [
                self._ray.get(self._chief_handle.create_experiment.remote(
                    self._t_prof.name + "_ADVLoss_P" + str(p_id)))
                for p_id in range(self._t_prof.n_seats)
            ]

        self._ray.wait([
            self._ray.remote(self._ps_handles[p_id].reset_adv_net.remote(cfr_iter))
        ])
        self._update_leaner_actors(update_adv_for_plyrs=[p_id])

        SMOOTHING = 200
        accumulated_averaged_loss = 0.0
        for epoch_nr in range(self._adv_args.n_batches_adv_training):
            t0 = time.time()

            # 在异步模式下，传递当前迭代号以过滤老数据
            if self._t_prof.use_async_data:
                # 修改LAs的训练调用以传递当前迭代号
                grads_from_all_las, _averaged_loss = self._get_adv_gradients(p_id=p_id, cfr_iter=cfr_iter)
            else:
                # 同步模式保持原样
                grads_from_all_las, _averaged_loss = self._get_adv_gradients(p_id=p_id)

            accumulated_averaged_loss += _averaged_loss

            t_computation += time.time() - t0

            # Applying gradients
            t0 = time.time()
            self._ray.get(self._ps_handles[p_id].apply_grads_adv.remote(grads_from_all_las))

            # Step LR scheduler
            self._ray.get(self._ps_handles[p_id].step_scheduler_adv.remote(_averaged_loss))

            # update ADV on all las
            self._update_leaner_actors(update_adv_for_plyrs=[p_id])

            # log current loss
            if self._t_prof.log_verbose and ((epoch_nr + 1) % SMOOTHING == 0):
                self._ray.wait([
                    self._ray.remote(self._chief_handle.add_scalar,
                                     exp_loss_each_p[p_id], "DCFR_NN_Losses/Advantage", epoch_nr,
                                     accumulated_averaged_loss / SMOOTHING)
                ])
                accumulated_averaged_loss = 0.0

            t_syncing += time.time() - t0

        return t_computation, t_syncing

    def _get_adv_gradients(self, p_id, cfr_iter=None):
        """获取优势网络的梯度

        Args:
            p_id: 玩家ID
            cfr_iter: 当前CFR迭代号，用于在异步模式下过滤老旧数据

        Returns:
            grads: 梯度列表
            averaged_loss: 平均损失值
        """
        # 检查是否为分布式模式
        is_distributed = hasattr(self._ray, 'runs_distributed') and self._ray.runs_distributed

        if is_distributed:
            if self._t_prof.use_async_data and cfr_iter is not None:
                # 异步模式下传递CFR迭代号
                grads = [
                    la.get_adv_grads.remote(p_id, cfr_iter)
                    for la in self._la_handles
                ]
            else:
                # 同步模式保持原样
                grads = [
                    la.get_adv_grads.remote(p_id)
                    for la in self._la_handles
                ]

            self._ray.wait(grads)

            losses = self._ray.get([
                la.get_loss_last_batch_adv.remote(p_id)
                for la in self._la_handles
            ])
        else:
            if self._t_prof.use_async_data and cfr_iter is not None:
                # 异步模式下传递CFR迭代号
                grads = [
                    la.get_adv_grads(p_id, cfr_iter)
                    for la in self._la_handles
                ]
            else:
                # 同步模式保持原样
                grads = [
                    la.get_adv_grads(p_id)
                    for la in self._la_handles
                ]

            losses = [
                la.get_loss_last_batch_adv(p_id)
                for la in self._la_handles
            ]

        losses = [loss for loss in losses if loss is not None]

        n = len(losses)
        averaged_loss = sum(losses) / float(n) if n > 0 else -1

        return grads, averaged_loss

    def _generate_traversals(self, p_id, cfr_iter):
        """同步模式下的数据生成方法"""
        if self._t_prof.use_async_data:
            # 异步模式下此方法不做任何事
            print(f"异步模式下不需要同步生成数据")
            return

        # 检查是否为分布式模式
        is_distributed = hasattr(self._ray, 'runs_distributed') and self._ray.runs_distributed

        if is_distributed:
            self._ray.wait([
                la.generate_data.remote(p_id, cfr_iter)
                for la in self._la_handles
            ])
        else:
            for la in self._la_handles:
                la.generate_data(p_id, cfr_iter)

    def _update_leaner_actors(self, update_adv_for_plyrs=None, update_avrg_for_plyrs=None):
        """

        Args:
            update_adv_for_plyrs (list):         list of player_ids to update adv for
            update_avrg_for_plyrs (list):        list of player_ids to update avrg for
        """

        assert isinstance(update_adv_for_plyrs, list) or update_adv_for_plyrs is None
        assert isinstance(update_avrg_for_plyrs, list) or update_avrg_for_plyrs is None

        _update_adv_per_p = [
            True if (update_adv_for_plyrs is not None) and (p in update_adv_for_plyrs) else False
            for p in range(self._t_prof.n_seats)
        ]

        _update_avrg_per_p = [
            True if (update_avrg_for_plyrs is not None) and (p in update_avrg_for_plyrs) else False
            for p in range(self._t_prof.n_seats)
        ]

        la_batches = []
        n = len(self._la_handles)
        c = 0
        while n > c:
            s = min(n, c + self._t_prof.max_n_las_sync_simultaneously)
            la_batches.append(self._la_handles[c:s])
            if type(la_batches[-1]) is not list:
                la_batches[-1] = [la_batches[-1]]
            c = s

        w_adv = [None for _ in range(self._t_prof.n_seats)]
        w_avrg = [None for _ in range(self._t_prof.n_seats)]

        # 检查是否为分布式模式
        is_distributed = hasattr(self._ray, 'runs_distributed') and self._ray.runs_distributed

        for p_id in range(self._t_prof.n_seats):
            if not _update_adv_per_p[p_id]:
                w_adv[p_id] = None
            else:
                if is_distributed:
                    w_adv[p_id] = self._ps_handles[p_id].get_adv_weights.remote()
                else:
                    w_adv[p_id] = self._ps_handles[p_id].get_adv_weights()

            if not _update_avrg_per_p[p_id]:
                w_avrg[p_id] = None
            else:
                if is_distributed:
                    w_avrg[p_id] = self._ps_handles[p_id].get_avrg_weights.remote()
                else:
                    w_avrg[p_id] = self._ps_handles[p_id].get_avrg_weights()

        for batch in la_batches:
            if is_distributed:
                self._ray.wait([
                    la.update.remote(w_adv, w_avrg)
                    for la in batch
                ])
            else:
                for la in batch:
                    la.update(w_adv, w_avrg)

        # 在异步模式下，更新后也要通知更新的最新模型信息
        if self._t_prof.use_async_data:
            for p_id in range(self._t_prof.n_seats):
                if _update_adv_per_p[p_id]:
                    for la in self._la_handles:
                        # 通知LA更新后台生成线程使用的模型
                        if is_distributed:
                            la.update_background_generation_model.remote(p_id)
                        else:
                            la.update_background_generation_model(p_id)

    # ____________ SINGLE only
    def _push_newest_adv_net_to_chief(self, p_id, cfr_iter):
        # 检查是否为分布式模式
        is_distributed = hasattr(self._ray, 'runs_distributed') and self._ray.runs_distributed

        if is_distributed:
            weights = self._ray.get(self._ps_handles[p_id].get_adv_weights.remote())
            self._ray.wait([
                self._chief_handle.add_new_iteration_strategy_model.remote(
                    p_id, weights, cfr_iter
                )
            ])
        else:
            weights = self._ps_handles[p_id].get_adv_weights()
            self._chief_handle.add_new_iteration_strategy_model(
                p_id, weights, cfr_iter
            )

    # ____________ AVRG only
    def _get_avrg_gradients(self, p_id, cfr_iter=None):
        """获取平均策略网络的梯度

        Args:
            p_id: 玩家ID
            cfr_iter: 当前CFR迭代号，用于在异步模式下过滤老旧数据

        Returns:
            grads: 梯度列表
            averaged_loss: 平均损失值
        """
        # 检查是否为分布式模式
        is_distributed = hasattr(self._ray, 'runs_distributed') and self._ray.runs_distributed

        if is_distributed:
            if self._t_prof.use_async_data and cfr_iter is not None:
                # 异步模式下传递CFR迭代号
                grads = [
                    la.get_avrg_grads.remote(p_id, cfr_iter)
                    for la in self._la_handles
                ]
            else:
                # 同步模式保持原样
                grads = [
                    la.get_avrg_grads.remote(p_id)
                    for la in self._la_handles
                ]

            self._ray.wait(grads)

            losses = self._ray.get([
                la.get_loss_last_batch_avrg.remote(p_id)
                for la in self._la_handles
            ])
        else:
            if self._t_prof.use_async_data and cfr_iter is not None:
                # 异步模式下传递CFR迭代号
                grads = [
                    la.get_avrg_grads(p_id, cfr_iter)
                    for la in self._la_handles
                ]
            else:
                # 同步模式保持原样
                grads = [
                    la.get_avrg_grads(p_id)
                    for la in self._la_handles
                ]

            losses = [
                la.get_loss_last_batch_avrg(p_id)
                for la in self._la_handles
            ]

        losses = [loss for loss in losses if loss is not None]

        n = len(losses)
        averaged_loss = sum(losses) / float(n) if n > 0 else -1

        return grads, averaged_loss

    def _train_avrg(self, p_id, cfr_iter):
        t_computation = 0.0
        t_syncing = 0.0

        # 检查是否有足够数据进行训练（仅异步模式）
        if self._t_prof.use_async_data:
            # 假设PS有类似的方法获取AVRG缓冲区大小
            buffer_size = self._ray.get(self._ps_handles[p_id].get_avrg_buffer_size.remote(p_id, cfr_iter))
            if buffer_size < self._t_prof.min_data_for_training:
                print(f"跳过玩家 {p_id} 的平均策略网络训练: 数据不足 ({buffer_size} < {self._t_prof.min_data_for_training})")
                return 0.0, 0.0

        # For logging the loss to see convergence in Tensorboard
        if self._t_prof.log_verbose:
            exp_loss_each_p = [
                self._ray.get(self._chief_handle.create_experiment.remote(
                    self._t_prof.name + "_AVRGLoss_P" + str(p_id)))
                for p_id in range(self._t_prof.n_seats)
            ]

        self._ray.wait([
            self._ray.remote(self._ps_handles[p_id].reset_avrg_net.remote())
        ])
        self._update_leaner_actors(update_avrg_for_plyrs=[p_id])

        SMOOTHING = 200
        accumulated_averaged_loss = 0.0

        if cfr_iter > 0:
            for epoch_nr in range(self._avrg_args.n_batches_avrg_training):
                t0 = time.time()

                # 在异步模式下，传递当前迭代号以过滤老数据
                if self._t_prof.use_async_data:
                    # 修改LAs的训练调用以传递当前迭代号
                    grads_from_all_las, _averaged_loss = self._get_avrg_gradients(p_id=p_id, cfr_iter=cfr_iter)
                else:
                    # 同步模式保持原样
                    grads_from_all_las, _averaged_loss = self._get_avrg_gradients(p_id=p_id)

                accumulated_averaged_loss += _averaged_loss

                t_computation += time.time() - t0

                # Applying gradients
                t0 = time.time()
                self._ray.get(self._ps_handles[p_id].apply_grads_avrg.remote(grads_from_all_las))

                # Step LR scheduler
                self._ray.get(self._ps_handles[p_id].step_scheduler_avrg.remote(_averaged_loss))

                # update AvrgStrategyNet on all las
                self._update_leaner_actors(update_avrg_for_plyrs=[p_id])

                # log current loss
                if self._t_prof.log_verbose and ((epoch_nr + 1) % SMOOTHING == 0):
                    self._ray.wait([
                        self._ray.remote(self._chief_handle.add_scalar,
                                         exp_loss_each_p[p_id], "DCFR_NN_Losses/Average", epoch_nr,
                                         accumulated_averaged_loss / SMOOTHING)
                    ])
                    accumulated_averaged_loss = 0.0

                t_syncing += time.time() - t0

        return t_computation, t_syncing
