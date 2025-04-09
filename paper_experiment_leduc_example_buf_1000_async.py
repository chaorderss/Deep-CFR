from PokerRL.game.games import StandardLeduc

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver

import torch
import os

# 设置 oneDNN 环境变量 (可选，解决之前的 TensorFlow 警告)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 自动检测并设置设备 (优先 MPS, 其次 CUDA, 最后 CPU)
if torch.backends.mps.is_available():
    device_str = "mps"
    print("检测到 MPS 后端，将使用 Metal GPU。")
elif torch.cuda.is_available():
    # 这里可以保留 device_id 逻辑，但 MPS 通常不需要 device_id
    # 如果需要特定 CUDA 设备，可以在这里细化逻辑
    device_str = "cuda"
    print("检测到 CUDA 后端，将使用 NVIDIA GPU。")
else:
    device_str = "cpu"
    print("未检测到 MPS 或 CUDA，将使用 CPU。")
if __name__ == '__main__':
    ctrl = Driver(t_prof=TrainingProfile(name="MySD-CFR2000",
                                        nn_type="feedforward",
                                        max_buffer_size_adv=1e6,
                                        max_buffer_size_avrg=1e6,

                                        n_traversals_per_iter=1500,
                                        n_batches_adv_training=750,
                                        n_batches_avrg_training=5000,
                                        n_merge_and_table_layer_units_adv=64,
                                        n_merge_and_table_layer_units_avrg=64,
                                        n_units_final_adv=64,
                                        n_units_final_avrg=64,
                                        mini_batch_size_adv=2048,
                                        mini_batch_size_avrg=2048,
                                        init_adv_model="last",
                                        init_avrg_model="random",
                                        use_pre_layers_adv=False,
                                        use_pre_layers_avrg=False,
                                        eval_agent_max_strat_buf_size=2000,

                                        game_cls=StandardLeduc,
                                        eval_agent_export_freq=100,
                                        checkpoint_freq=50,
                                        eval_modes_of_algo=(
                                            EvalAgentDeepCFR.EVAL_MODE_SINGLE,  # SD-CFR
                                        ),
                                        device_inference="cpu",
                                        device_training=device_str,
                                        device_parameter_server="cpu",
                                        DISTRIBUTED=False,
                                        log_verbose=True,
                                        path_data = "./models/MySD-CFR2000",
                                        use_async_data=False,        # 启用异步数据生成
                                        max_data_staleness=3,       # 数据最大年龄(迭代数)
                                        min_data_for_training=2048,
                                         ),
                  eval_methods={
                      "br": 10,
                  },
                  n_iterations=None)
    ctrl.run()
