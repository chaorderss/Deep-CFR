# 版权信息
# Copyright (c) 2019 Eric Steinberger

import torch # 导入 PyTorch 框架
import torch.nn as nn # 导入神经网络模块

# 定义 DuelingQNet 类，继承自 PyTorch 的 nn.Module
class DuelingQNet(nn.Module):
    """
    实现 Dueling Q-Network 结构。
    这个网络通常用于 Deep CFR/SD-CFR 中近似优势函数 (Advantage Function) 或累积遗憾。
    它将 Q 值分解为状态价值 V(s) 和动作优势 A(s,a)。
    """

    def __init__(self, env_bldr, q_args, device):
        """
        初始化 DuelingQNet。

        Args:
            env_bldr: 环境构建器，用于获取环境信息 (如动作空间大小)。
            q_args (DuelingQArgs): 包含网络配置参数的对象。
            device: 指定网络运行的设备 (CPU 或 GPU)。
        """
        super().__init__() # 调用父类 nn.Module 的初始化方法

        # 保存环境构建器、参数和动作数量
        self._env_bldr = env_bldr
        self._q_args = q_args
        self._n_actions = env_bldr.N_ACTIONS # 获取游戏的总动作数量

        # 定义 ReLU 激活函数，inplace=False 表示不修改输入张量
        self._relu = nn.ReLU(inplace=False)

        # --- 输入特征处理模块 (MPM: Multi-Process Module) ---
        # 从参数中动态获取 MPM 的类定义
        MPM = q_args.mpm_args.get_mpm_cls()
        # 实例化 MPM 模块，它负责处理原始的公共观察信息 (pub_obses) 和范围索引 (range_idxs)
        # 并输出一个共享的特征表示 (shared_out)
        self._mpm = MPM(env_bldr=env_bldr, device=device, mpm_args=q_args.mpm_args)

        # ____________________ 优势流 (Advantage Stream) 和价值流 (Value Stream) 的层定义 _______________________
        # 优势流的第一个线性层
        self._adv_layer = nn.Linear(in_features=self._mpm.output_units, # 输入来自 MPM 输出
                                    out_features=q_args.n_units_final) # 输出到指定的隐藏单元数
        # 价值流的第一个线性层
        self._state_v_layer = nn.Linear(in_features=self._mpm.output_units, # 输入来自 MPM 输出
                                      out_features=q_args.n_units_final) # 输出到指定的隐藏单元数

        # 优势流的最终输出层，输出每个动作的优势值
        self._adv = nn.Linear(in_features=q_args.n_units_final, out_features=self._n_actions)
        # 价值流的最终输出层，输出一个标量状态价值
        self._v = nn.Linear(in_features=q_args.n_units_final, out_features=1)

        # 将整个网络模块移动到指定的设备 (CPU/GPU)
        self.to(device)

        # 打印网络结构
        print("DuelingQNet 结构:")
        print(self)
        print("\n参数总数:", sum(p.numel() for p in self.parameters()))
        print("DuelingQNet 初始化完成")


    def forward(self, pub_obses, range_idxs, legal_action_masks):
        """
        定义网络的前向传播逻辑。

        Args:
            pub_obses: 公共观察信息 (一批)。
            range_idxs: 范围索引 (一批)。
            legal_action_masks: 合法动作掩码 (一批)，形状通常是 (batch_size, n_actions)，合法动作为1，非法为0。

        Returns:
            torch.Tensor: 每个动作的估计 Q 值 (或近似的优势/遗憾)，非法动作的值被掩码为 0。
                          形状为 (batch_size, n_actions)。
        """
        # 1. 通过共享的 MPM 模块提取特征
        shared_out = self._mpm(pub_obses=pub_obses, range_idxs=range_idxs)

        # 2. 计算中心化的优势流 (Centered Advantage) A(s, a) - mean_k[A(s, k)]
        #    调用内部辅助函数 _get_adv 完成计算
        adv = self._get_adv(shared_out=shared_out, legal_action_masks=legal_action_masks)

        # 3. 计算状态价值流 V(s)
        val_layer = self._relu(self._state_v_layer(shared_out)) # 通过价值流的第一个层和 ReLU
        val = self._v(val_layer) # 通过价值流的最终层得到 V(s)，形状 (batch_size, 1)
        # 将 V(s) 扩展 (expand) 成与优势流相同的形状 (batch_size, n_actions)
        # 这样 V(s) 对每个动作都相同
        val = val.expand_as(adv)

        # 4. 合并价值流和中心化的优势流，并应用掩码
        # Q(s, a) = V(s) + (A(s, a) - mean_k[A(s, k)])
        # 乘以 legal_action_masks 确保非法动作的最终输出为 0
        # (这对于后续策略推断很重要，非法动作的概率应为 0)
        return (val + adv) * legal_action_masks

    def get_adv(self, pub_obses, range_idxs, legal_action_masks):
        """
        公共接口：只计算并返回中心化的优势值 A(s, a) - mean_k[A(s, k)]。
        """
        # 先通过 MPM 提取共享特征
        shared_out = self._mpm(pub_obses=pub_obses, range_idxs=range_idxs)
        # 调用内部方法计算中心化优势
        return self._get_adv(shared_out=shared_out, legal_action_masks=legal_action_masks)

    def _get_adv(self, shared_out, legal_action_masks):
        """
        内部辅助函数：计算中心化的优势值 A(s, a) - mean_k[A(s, k)]。
        """
        # 1. 计算原始的优势值 A(s, a)
        y = self._relu(self._adv_layer(shared_out)) # 通过优势流的第一个层和 ReLU
        y = self._adv(y) # 通过优势流的最终层得到 A(s, a)，形状 (batch_size, n_actions)

        # 2. 中心化优势值：减去合法动作优势的均值
        # 首先，将非法动作的原始优势值置为 0，以便计算合法动作的总和
        y_masked = y * legal_action_masks

        # 计算合法动作的优势值总和 (dim=1 表示按动作维度求和)
        sum_legal_adv = y_masked.sum(dim=1)
        # 计算合法动作的数量
        n_legal_actions = legal_action_masks.sum(dim=1)
        # 避免除以零 (如果没有合法动作，理论上不应发生，但可以加个 epsilon)
        n_legal_actions = torch.clamp(n_legal_actions, min=1e-8)

        # 计算合法动作的平均优势值
        mean_legal_adv = sum_legal_adv / n_legal_actions
        # unsqueeze(1) 将形状从 (batch_size,) 变为 (batch_size, 1)
        # expand_as(y) 将形状扩展为 (batch_size, n_actions)，使均值对每个动作都相同
        mean = mean_legal_adv.unsqueeze(1).expand_as(y)

        # 3. 从原始优势值中减去平均值
        centered_adv = y - mean

        # 4. 再次应用掩码
        # 因为减去均值时，非法动作的条目也会被减（变成 0 - mean），需要重新将它们置为 0
        return centered_adv * legal_action_masks


# 定义一个简单的类来存储 DuelingQNet 的配置参数
class DuelingQArgs:
    """
    存储 DuelingQNet 配置参数的数据类。
    """
    def __init__(self, n_units_final, mpm_args):
        """
        Args:
            n_units_final (int): 价值流和优势流分支中最终隐藏层的大小。
            mpm_args: 传递给 Multi-Process Module (MPM) 的参数对象。
        """
        self.n_units_final = n_units_final
        self.mpm_args = mpm_args