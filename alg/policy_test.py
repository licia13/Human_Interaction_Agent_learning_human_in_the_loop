
from __future__ import annotations

import numpy as np
import torch
from stable_baselines3 import SAC
from utils.env_wrappers import reconstruct_state


def load_final_policy(path_to_saved_policy: str, algo: str = "sac"):
    
    if algo == "sac":
        return SAC.load(path_to_saved_policy)

    if algo == "awac":
        from alg.policy_learn_awac import AWACAgent
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        agent = AWACAgent(obs_dim=22, action_dim=4, device=device)
        agent.load(path_to_saved_policy)
        return agent

    raise ValueError(f"Unknown algo '{algo}'. Choose 'sac' or 'awac'.")


def get_policy_action(state: dict, saved_policy_model) -> np.ndarray:
    """Return the deterministic action for the given environment state.

    Args:
        state (dict): Environment state with keys "observation",
                      "achieved_goal", and "desired_goal".
        saved_policy_model: Model returned by load_final_policy().

    Returns:
        action (np.ndarray): Action in [-1, 1]^4.
    """
    flat_state = reconstruct_state(state).astype(np.float32)

    # SAC (SB3): use .predict()
    if isinstance(saved_policy_model, SAC):
        action, _ = saved_policy_model.predict(flat_state, deterministic=True)
        return action

    # AWAC (custom): use .get_action()
    from alg.policy_learn_awac import AWACAgent
    if isinstance(saved_policy_model, AWACAgent):
        return saved_policy_model.get_action(flat_state, deterministic=True)

    raise TypeError(f"Unrecognised model type: {type(saved_policy_model)}")
