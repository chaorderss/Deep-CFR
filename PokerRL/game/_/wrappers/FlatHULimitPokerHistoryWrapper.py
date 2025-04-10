# Copyright (c) 2019 Eric Steinberger


import copy

import numpy as np

from PokerRL.game.Poker import Poker
from PokerRL.game._.wrappers._Wrapper import Wrapper


class FlatHULimitPokerHistoryWrapper(Wrapper):
    """
    This wrapper only supports Heads Up games.
    This wrapper is suitable for feedforward NN architectures.

    Stores action sequence and appends to current obs similarly as is proposed in https://arxiv.org/abs/1603.01121
    Action History is *AFTER* raw env obs state in the vector
    """

    def __init__(self, env, env_bldr_that_built_me):
        """
        Initializes the wrapper.

        Args:
            env: The original poker environment instance.
            env_bldr_that_built_me: The environment builder (EnvBuilder) instance that created this environment and wrapper.
                                    The builder contains configuration information about the observation space, action space, and history vector size.
        """
        assert env.N_SEATS == 2, "FlatHULimitPokerHistoryWrapper 只支持单挑 (2人) 游戏"
        super().__init__(env=env, env_bldr_that_built_me=env_bldr_that_built_me)
        print(f"env_bldr_that_built_me: {env_bldr_that_built_me}")
        self._action_vector_size = env_bldr_that_built_me.action_vector_size
        self._action_count_this_round = None
        self._game_round_last_tick = None
        self._action_history_vector = None

    def _reset_state(self, **kwargs):
        """
        Resets the internal state of the wrapper. Usually called when the environment is reset.
        """
        self._action_count_this_round = [0, 0]  # one per player
        self._game_round_last_tick = Poker.PREFLOP
        self._action_history_vector = np.zeros(shape=self._action_vector_size, dtype=np.float32)

    def _pushback(self, env_obs=None):
        """
        Updates the action history vector after each environment step.
        This is the core logic for encoding the action sequence.
        (env_obs parameter is not used here)
        """
        # Processing logic: because we observe every transition, the last action of the old round is still in the old round.
        # Therefore, all the new_round logic has to be executed from the *next* transition onwards; this transition is handled within the old round.

        # If None, env was just reset
        _last_a_type = self.env.last_action[0]
        if _last_a_type is not None:
            _last_actor_id = self.env.last_action[2]

            idx = self.env_bldr.get_vector_idx(round_=self._game_round_last_tick,
                                               p_id=_last_actor_id,
                                               nth_action_this_round=self._action_count_this_round[_last_actor_id],
                                               action_idx=_last_a_type)

            if idx < self._action_vector_size:
                self._action_history_vector[idx] = 1.0
            else:
                print(f"警告: 动作历史索引 {idx} 超出向量大小 {self._action_vector_size}")

            self._action_count_this_round[_last_actor_id] += 1
            if self.env.current_round != self._game_round_last_tick:
                self._game_round_last_tick = self.env.current_round
                self._action_count_this_round = [0, 0]

    def print_obs(self, wrapped_obs=None):
        """Prints the wrapped observation vector, including both base observation and appended action history."""
        if wrapped_obs is None:
            wrapped_obs = self.get_current_obs()

        assert isinstance(wrapped_obs, np.ndarray), "观察值必须是 NumPy 数组"
        print()
        print()
        print("*****************************************************************************************************")
        print()
        print("________________________________________ 包装后的观察 + 历史 ________________________________________")
        print()
        base_obs_size = self.env_bldr.pub_obs_size - self._action_vector_size
        print("--- 基础环境观察 ---")
        self.env.print_obs(wrapped_obs[:base_obs_size])
        print()
        print("--- 扁平化动作序列 (历史向量) ---")
        print(wrapped_obs[base_obs_size:])
        print("*****************************************************************************************************")

    def get_current_obs(self, env_obs=None):
        """
        Gets the wrapped current observation vector.

        Args:
            env_obs (np.ndarray, optional): The base environment observation vector. If None, will get from the base environment.

        Returns:
            np.ndarray: The base observation vector with appended action history.
        """
        if env_obs is None:
            base_obs = self.env.get_current_obs(is_terminal=False)
        else:
            base_obs = env_obs

        return np.concatenate((base_obs.astype(np.float32), self._action_history_vector.astype(np.float32)), axis=0)

    def state_dict(self):
        """Returns the state dictionary of the wrapper for saving."""
        return {
            "base": super().state_dict(),
            "a_seq": np.copy(self._action_history_vector),
            "game_round_last_tick": self._game_round_last_tick,
            "action_count_this_round": copy.deepcopy(self._action_count_this_round),
        }

    def load_state_dict(self, state_dict):
        """Loads the state of the wrapper from a dictionary."""
        super().load_state_dict(state_dict=state_dict["base"])
        self._action_history_vector = np.copy(state_dict["a_seq"])
        self._game_round_last_tick = state_dict["game_round_last_tick"]
        self._action_count_this_round = copy.copy(state_dict["action_count_this_round"])

    def set_to_public_tree_node_state(self, node):
        """
        Sets the internal env wrapper to the state ""node"" is in.

        Args:
            node:                   Any node (of any type) in a PublicTree instance.
        """
        state_seq = []  # will be sorted. [0] is root.

        def add(_node):
            if _node is not None:
                if _node.p_id_acting_next != _node.tree.CHANCE_ID:
                    state_seq.insert(0, _node.env_state)
                add(_node.parent)

        add(node)  # will go from node to parent to parent of parent... to root and reverse the direction for the tree
        # fake step in the internal env to gain current rr state
        self.reset()
        self._reset_state()
        for sd in state_seq:
            self.env.load_state_dict(sd, blank_private_info=True)  # load cause _pushback uses the env's internal state
            self._pushback()
