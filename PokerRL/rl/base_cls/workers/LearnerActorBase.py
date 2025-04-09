# Copyright (c) 2019 Eric Steinberger

from PokerRL.rl.base_cls.workers.WorkerBase import WorkerBase


class LearnerActorBase(WorkerBase):
    """
    学习器-执行器基类，扩展了WorkerBase类
    """

    def __init__(self, t_prof, worker_id, chief_handle):
        super().__init__(t_prof=t_prof)
        self.worker_id = worker_id
        self._chief_handle = chief_handle

    def update(self, *args, **kwargs):
        """更新模型"""
        raise NotImplementedError

    def get_adv_grads(self, p_id, cfr_iter=None):
        """获取优势网络梯度"""
        raise NotImplementedError

    def get_avrg_grads(self, p_id, cfr_iter=None):
        """获取平均策略网络梯度"""
        raise NotImplementedError