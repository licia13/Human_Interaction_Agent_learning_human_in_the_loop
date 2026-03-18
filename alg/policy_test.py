from __future__ import annotations

import numpy as np
from stable_baselines3 import SAC
from utils.env_wrappers import reconstruct_state


def load_final_policy(path_to_saved_policy: str):
    return SAC.load(path_to_saved_policy)


def get_policy_action(state: dict, saved_policy_model) -> np.ndarray:
    flat_state = reconstruct_state(state).astype(np.float32)
    action, _ = saved_policy_model.predict(flat_state, deterministic=True)
    return action