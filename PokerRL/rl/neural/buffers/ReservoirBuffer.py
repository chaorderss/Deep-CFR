import numpy as np
import torch


class ReservoirBuffer:
    """
    简化版的蓄水池缓冲区，用于存储样本
    """

    def __init__(self, max_size):
        """
        初始化

        Args:
            max_size: 缓冲区最大容量
        """
        self._max_size = max_size
        self._buffer = []
        self._n_entries_seen = 0

    def add_samples(self, samples):
        """
        添加多个样本到缓冲区

        Args:
            samples: 样本列表
        """
        if not samples:
            return

        for sample in samples:
            self._n_entries_seen += 1

            if len(self._buffer) < self._max_size:
                self._buffer.append(sample)
            else:
                # 使用蓄水池采样算法决定是否替换
                replace_idx = np.random.randint(0, self._n_entries_seen)
                if replace_idx < self._max_size:
                    self._buffer[replace_idx] = sample

    def sample(self, batch_size):
        """
        从缓冲区随机采样

        Args:
            batch_size: 批次大小

        Returns:
            样本列表
        """
        if len(self._buffer) == 0:
            return []

        idxs = np.random.randint(0, len(self._buffer), min(batch_size, len(self._buffer)))
        return [self._buffer[i] for i in idxs]

    def __len__(self):
        """
        获取缓冲区当前大小
        """
        return len(self._buffer)