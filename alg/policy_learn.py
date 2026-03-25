from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import SAC

from envs.task_envs import PnPNewRobotEnv
from utils.demos import prepare_demo_pool
from utils.env_wrappers import ActionNormalizer, ResetWrapper, TimeLimitWrapper, reconstruct_state
from alg.banana import feature_function

REWARD_SCALE = 20.0


def load_weights(path: Path) -> np.ndarray:
    """Read learned reward weights from feature_weights.csv."""
    weights = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            weights.append(float(row["weight"]))
    return np.array(weights, dtype=np.float32)


class FlatObsWrapper(gym.ObservationWrapper):
    """Flattens the dict observation to a 1-D vector required by SB3's MlpPolicy."""

    def __init__(self, env):
        super().__init__(env)
        obs_dim = (
            env.observation_space["observation"].shape[0]
            + env.observation_space["desired_goal"].shape[0]
        )
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def observation(self, obs):
        return reconstruct_state(obs).astype(np.float32)


def make_sac_env(render: bool = False):
    """Env used only to define SAC's observation/action spaces.  Produces flat obs."""
    env = PnPNewRobotEnv(render=render)
    env = ResetWrapper(env)
    env = ActionNormalizer(env)
    env = TimeLimitWrapper(env, max_steps=150)
    env = FlatObsWrapper(env)
    return env


def make_rollout_env(render: bool = False):
    """Env used for episodic rollouts.  Preserves the dict observation so
    feature_function can read 'observation' and 'desired_goal' directly."""
    env = PnPNewRobotEnv(render=render)
    env = ResetWrapper(env)
    env = ActionNormalizer(env)
    env = TimeLimitWrapper(env, max_steps=150)
    return env


def load_demos_into_buffer(model, demos: List[Dict], weights: np.ndarray) -> None:
    """Pre-fill the SAC replay buffer with expert demonstrations.

    For each demo episode the trajectory reward is computed post-hoc from the
    full trajectory using feature_function (matching the episodic training loop).
    """
    for demo in demos:
        states = demo["state_trajectory"]
        actions = demo["action_trajectory"]
        next_states = demo["next_state_trajectory"]
        dones = demo["done_trajectory"].flatten()
        # Distinguish true task termination (success) from time-limit truncation.
        # In this project, sparse reward is positive (+1000) only on success.
        demo_rewards = demo.get("reward_trajectory", np.zeros_like(dones)).flatten()

        T = len(actions)

        # Build traj_pairs in dict format so feature_function can parse them
        traj_pairs = []
        for t in range(T):
            obs_dict = {
                "observation": states[t, :19].astype(np.float32),
                "desired_goal": states[t, 19:22].astype(np.float32),
                "achieved_goal": states[t, 7:10].astype(np.float32),
            }
            traj_pairs.append((obs_dict, actions[t]))

        # Compute trajectory-level reward and distribute equally across steps
        features = feature_function(traj_pairs)
        trajectory_reward = float(np.dot(weights, features))
        reward_per_step = REWARD_SCALE * trajectory_reward / max(T, 1)

        # Insert every transition into the replay buffer
        for t in range(T):
            flat_obs = np.concatenate([
                states[t, :19].astype(np.float32),
                states[t, 19:22].astype(np.float32),
            ])
            flat_next_obs = np.concatenate([
                next_states[t, :19].astype(np.float32),
                states[t, 19:22].astype(np.float32),
            ])
            action = actions[t].astype(np.float32)
            done = bool(dones[t])

            model.replay_buffer.add(
                flat_obs, flat_next_obs, action,
                np.array([reward_per_step]),
                # Terminal only on true termination (success), not timeouts.
                np.array([bool(dones[t]) and float(demo_rewards[t]) > 0.0]),
                [{}],
            )


def rollout_episode(
    rollout_env, model
) -> Tuple[List, List, int]:
    """Roll out the current policy for one full episode.

    Returns:
        traj_pairs  -- list of (obs_dict, action) for feature_function
        transitions -- list of (flat_obs, flat_next_obs, action, done) for buffer
        ep_steps    -- number of steps taken
    """
    obs_dict, _ = rollout_env.reset()
    flat_obs = reconstruct_state(obs_dict).astype(np.float32)

    traj_pairs: List = []
    transitions: List = []

    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(flat_obs, deterministic=False)

        next_obs_dict, _, terminated, truncated, info = rollout_env.step(action)
        flat_next_obs = reconstruct_state(next_obs_dict).astype(np.float32)

        traj_pairs.append((obs_dict, action))
        transitions.append((
            flat_obs.copy(),
            flat_next_obs.copy(),
            action.astype(np.float32),
            terminated,
        ))

        obs_dict = next_obs_dict
        flat_obs = flat_next_obs

    return traj_pairs, transitions, len(transitions)


def evaluate(model, n_runs: int = 10) -> float:
    """Run n_runs full episodes with the deterministic policy and return success rate."""
    eval_env = make_rollout_env(render=False)
    successes = 0
    for _ in range(n_runs):
        obs_dict, _ = eval_env.reset()
        flat_obs = reconstruct_state(obs_dict).astype(np.float32)
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(flat_obs, deterministic=True)
            obs_dict, _, terminated, truncated, info = eval_env.step(action)
            flat_obs = reconstruct_state(obs_dict).astype(np.float32)
            if info.get("is_success", False):
                successes += 1
    eval_env.close()
    return successes / n_runs


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    saved_dir = repo_root / "saved"
    models_dir = saved_dir / "policy_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Training horizon (must be available before calling SB3's internal setup).
    max_steps = 500_000

    # ── Load learned reward weights ──────────────────────────────────────────
    weights = load_weights(saved_dir / "feature_weights.csv")
    print(f"Loaded weights: {weights}")

    # ── Create SAC model (uses flat-obs env to set network input size) ───────
    sac_env = make_sac_env(render=False)
    model = SAC(
        "MlpPolicy",
        sac_env,
        verbose=0,
        learning_starts=500,
        batch_size=256,
        buffer_size=300_000,
    )

    # SB3 requires `_setup_learn()` before manually calling `model.train()`.
    # It initializes `_logger` and internal counters used by learning-rate schedules.
    model._setup_learn(total_timesteps=max_steps, reset_num_timesteps=True)

    # ── Intercept SB3 logger to capture actor / critic losses ────────────────
    # SB3 records losses inside train() and clears them on dump().
    # We sniff the values just before dump() clears name_to_value.
    _sac_losses: Dict = {}
    # Some SB3 versions initialize the logger lazily, so accessing `model.logger`
    # can crash with: "AttributeError: 'SAC' object has no attribute '_logger'".
    # In that case, we just skip loss capture (success curve still works).
    try:
        _orig_dump = model.logger.dump

        def _capturing_dump(step: int = 0) -> None:
            _sac_losses.update(model.logger.name_to_value)
            _orig_dump(step)

        model.logger.dump = _capturing_dump  # type: ignore[method-assign]
    except AttributeError:
        print("[policy_learn] SB3 logger not initialized; skipping loss capture.")

    # ── Pre-fill buffer with expert demos ────────────────────────────────────
    demos = prepare_demo_pool(repo_root / "demo_data" / "PickAndPlace")
    load_demos_into_buffer(model, demos, weights)
    print(f"Loaded {len(demos)} expert demos into replay buffer")

    # ── Open metrics CSV ──────────────────────────────────────────────────────
    metrics_path = saved_dir / "metrics_sac.csv"
    metrics_file = open(metrics_path, "w", newline="")
    _fields = ["steps", "success_rate", "ep_reward", "ep_length",
               "actor_loss", "critic_loss"]
    metrics_writer = csv.DictWriter(metrics_file, fieldnames=_fields)
    metrics_writer.writeheader()

    # ── Episodic training loop ────────────────────────────────────────────────
    rollout_env = make_rollout_env(render=False)

    total_steps = 0
    eval_interval = 1_000
    steps_since_eval = 0
    success_rates: List[float] = []
    checkpoints: List[int] = []

    # Per-window accumulators (reset every eval_interval steps)
    w_ep_rewards:    List[float] = []
    w_ep_lengths:    List[float] = []
    w_actor_losses:  List[float] = []
    w_critic_losses: List[float] = []

    while total_steps < max_steps:

        # Step 1 — roll out ONE full episode with the current policy
        traj_pairs, transitions, ep_steps = rollout_episode(rollout_env, model)

        # Step 2 — recover trajectory reward post-hoc from the full trajectory
        features = feature_function(traj_pairs)
        trajectory_reward = float(np.dot(weights, features))
        reward_per_step = REWARD_SCALE * trajectory_reward / max(ep_steps, 1)

        w_ep_rewards.append(trajectory_reward)
        w_ep_lengths.append(float(ep_steps))

        # Step 3 — save all transitions to the replay buffer
        for flat_obs, flat_next_obs, action, done in transitions:
            model.replay_buffer.add(
                flat_obs, flat_next_obs, action,
                np.array([reward_per_step]), np.array([done]), [{}],
            )

        # Step 4 — update the policy (only once enough data is collected)
        total_steps += ep_steps
        model.num_timesteps = total_steps
        # Keep SB3's progress remaining consistent with our custom rollout loop.
        model._update_current_progress_remaining(
            num_timesteps=model.num_timesteps, total_timesteps=max_steps
        )
        if model.replay_buffer.size() >= model.learning_starts:
            # Cap the number of updates per episode for stability.
            gradient_steps = min(ep_steps, 64)
            model.train(batch_size=model.batch_size, gradient_steps=gradient_steps)
            # Prefer reading from SB3 logger directly; fall back to intercepted
            # values (for compatibility across SB3 versions).
            try:
                al = model.logger.name_to_value.get("train/actor_loss", np.nan)
                cl = model.logger.name_to_value.get("train/critic_loss", np.nan)
            except Exception:
                al = _sac_losses.get("train/actor_loss", np.nan)
                cl = _sac_losses.get("train/critic_loss", np.nan)
            if not np.isnan(float(al)):
                w_actor_losses.append(float(al))
            if not np.isnan(float(cl)):
                w_critic_losses.append(float(cl))

        steps_since_eval += ep_steps

        # ── Evaluate + checkpoint every eval_interval steps ──────────────────
        if steps_since_eval >= eval_interval:
            rate = evaluate(model)
            success_rates.append(rate)
            checkpoints.append(total_steps)
            print(f"[{total_steps:>7d} steps]  success rate: {rate:.2f}")
            model.save(str(models_dir / f"policy_{total_steps}"))

            metrics_writer.writerow({
                "steps":        total_steps,
                "success_rate": rate,
                "ep_reward":    np.mean(w_ep_rewards)    if w_ep_rewards    else np.nan,
                "ep_length":    np.mean(w_ep_lengths)    if w_ep_lengths    else np.nan,
                "actor_loss":   np.mean(w_actor_losses)  if w_actor_losses  else np.nan,
                "critic_loss":  np.mean(w_critic_losses) if w_critic_losses else np.nan,
            })
            metrics_file.flush()

            w_ep_rewards.clear(); w_ep_lengths.clear()
            w_actor_losses.clear(); w_critic_losses.clear()
            steps_since_eval = 0

    sac_env.close()
    rollout_env.close()
    metrics_file.close()

    # ── Plot learning curve ───────────────────────────────────────────────────
    plt.figure(figsize=(12, 5))
    plt.plot(checkpoints, success_rates)
    plt.xlabel("Environment Steps")
    plt.ylabel("Average Success Rate (10 runs)")
    plt.title("Policy Learning Curve (SAC)")
    plt.tight_layout()
    plot_path = saved_dir / "learning_curve.png"
    plt.savefig(str(plot_path))
    print(f"Learning curve saved to {plot_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
