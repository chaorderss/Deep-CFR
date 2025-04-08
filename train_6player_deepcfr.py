from PokerRL.game.games import NoLimitHoldem, StandardLeduc
from PokerRL.eval.lbr import LocalLBRMaster

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver

def train_deepcfr_6players(
        game_cls=NoLimitHoldem,  # 游戏类型：可选NoLimitHoldem或StandardLeduc
        n_iterations=300,        # 训练迭代次数
        n_traversals_per_iter=3000,  # 每次迭代的树遍历次数
        mini_batch_size=2048,    # 批处理大小
        max_buffer_size=2e6,     # 最大缓冲区大小
        save_path="./models/deepcfr_6p",  # 模型保存路径
        eval_freq=10,            # 评估频率（迭代次数）
        device_id=0,             # GPU设备ID，-1表示CPU
        verbose=True,            # 是否打印详细信息
        ):
    """
    训练6人桌DeepCFR模型

    参数:
        game_cls: 游戏类，默认为无限注德州扑克
        n_iterations: 训练迭代次数
        n_traversals_per_iter: 每次迭代的树遍历次数
        mini_batch_size: 训练批次大小
        max_buffer_size: 最大经验回放缓冲区大小
        save_path: 模型保存路径
        eval_freq: 每隔多少次迭代进行一次评估
        device_id: GPU设备ID，-1表示CPU
        verbose: 是否打印详细信息

    返回:
        dict: 包含训练好的模型和评估结果的字典
    """
    # 设置LBR评估参数
    lbr_args = {
        "n_workers": 10,             # LBR工作进程数
        "n_lbr_hands": 10000,        # LBR评估的手牌数量
        "lbr_mem_capacity": 80000,   # LBR内存容量
        "use_gpu": device_id >= 0,   # 是否使用GPU
        "n_traversals_per_iter": 1,  # 每次迭代的遍历次数
    }

    # 创建训练配置文件
    t_prof = TrainingProfile(
        name="DEEPCFR_6P",
        nn_type="feedforward",

        # 直接设置座位数(6人桌)
        n_seats=6,

        # 缓冲区设置
        max_buffer_size_adv=max_buffer_size,
        max_buffer_size_avrg=max_buffer_size,

        # 训练频率设置
        eval_agent_export_freq=eval_freq,
        checkpoint_freq=eval_freq*5,

        # 采样和训练批次
        n_traversals_per_iter=n_traversals_per_iter,
        n_batches_adv_training=int(n_traversals_per_iter/4),
        n_batches_avrg_training=int(n_traversals_per_iter*2),

        # 网络结构
        n_merge_and_table_layer_units_adv=128,
        n_merge_and_table_layer_units_avrg=128,
        n_units_final_adv=128,
        n_units_final_avrg=128,

        # 批处理大小
        mini_batch_size_adv=mini_batch_size,
        mini_batch_size_avrg=mini_batch_size,

        # 模型初始化
        init_adv_model="last",
        init_avrg_model="random",

        # 网络结构简化
        use_pre_layers_adv=True,
        use_pre_layers_avrg=True,

        # 游戏设置 - 正确方式
        game_cls=game_cls,
        start_chips=10000,  # 起始筹码
        chip_randomness=(0, 0),  # 筹码随机范围

        # 如果是德州扑克，需要设置下注集合
        agent_bet_set=[0.5, 1.0, 2.0] if game_cls == NoLimitHoldem else None,

        # 观察空间设置
        use_simplified_headsup_obs=False,
        uniform_action_interpolation=False,

        # 评估模式
        eval_modes_of_algo=(
            EvalAgentDeepCFR.EVAL_MODE_SINGLE,   # SD-CFR
            EvalAgentDeepCFR.EVAL_MODE_AVRG_NET, # Deep CFR
        ),

        # 设备和分布式训练
        device_inference="cuda" if device_id >= 0 else "cpu",
        device_training="cuda" if device_id >= 0 else "cpu",
        device_parameter_server="cuda" if device_id >= 0 else "cpu",
        DISTRIBUTED=False,

        # 日志设置
        log_verbose=verbose,

        # 保存路径
        path_data=save_path,

        # 添加LBR评估参数
        lbr_args=lbr_args,
    )

    # 设置评估方法 - 使用LBR评估
    eval_methods = {
        "br": eval_freq,       # 标准BR评估
        "lbr": eval_freq*2,    # LBR评估，频率比标准评估低一些
    }

    # 创建训练驱动并运行
    ctrl = Driver(
        t_prof=t_prof,
        eval_methods=eval_methods,
        n_iterations=n_iterations
    )

    # 运行训练
    results = ctrl.run()

    # 返回训练结果
    return {
        "model": ctrl.eval_agent,
        "results": results,
        "training_profile": t_prof
    }

if __name__ == '__main__':
    # 使用标准Leduc扑克进行6人桌训练（与原始示例保持一致）
    result = train_deepcfr_6players(
        game_cls=StandardLeduc,
        n_iterations=300,
        n_traversals_per_iter=1500,
        mini_batch_size=2048,
        max_buffer_size=1e6,
        save_path="./models/leduc_6p",
        eval_freq=15,  # 与原始代码中的"br": 15保持一致
        verbose=True
    )

    print("训练完成!")
    print(f"训练结果: {result['results']}")