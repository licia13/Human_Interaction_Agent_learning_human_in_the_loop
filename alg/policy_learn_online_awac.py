from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio

import torch

# APReL (preference learning)
import aprel.utils.util_functions as util_funs
from aprel.basics.environment import Environment as AprelEnvironment
from aprel.basics.trajectory import TrajectorySet
from aprel.learning.belief_models import SamplingBasedBelief
from aprel.learning.data_types import Preference, PreferenceQuery
from aprel.learning.user_models import SoftmaxUser, HumanUser
from aprel.querying.query_optimizer import QueryOptimizerDiscreteTrajectorySet

# Project code
from envs.task_envs import PnPNewRobotEnv
from utils.demos import prepare_demo_pool
from utils.env_wrappers import (
    ActionNormalizer,
    AprelGymAdapter,
    ResetWrapper,
    TimeLimitWrapper,
    TrajectoryRecord,
    reconstruct_state,
)
from alg.banana import feature_function, capture_frame
from alg.policy_learn_awac import AWACAgent, ReplayBuffer


def make_rollout_env(render: bool = False):
    """Env used for rollouts + clip capture (dict observations preserved)."""
    env = PnPNewRobotEnv(render=render)
    env = ResetWrapper(env)
    env = ActionNormalizer(env)
    env = TimeLimitWrapper(env, max_steps=150)
    return env


def save_frames_to_mp4(frames: List[np.ndarray], out_path: Path, fps: int = 30) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer_kwargs: Dict[str, Any] = dict(
        fps=fps,
        codec="libx264",
        ffmpeg_params=["-preset", "ultrafast", "-crf", "28"],
    )
    with imageio.get_writer(str(out_path), **writer_kwargs) as writer:
        for frame in frames:
            writer.append_data(frame)


@torch.no_grad()
def get_weights_from_belief(belief) -> np.ndarray:
    """Extract posterior-mean reward weights from APReL belief."""
    w = belief.mean["weights"]
    return np.array(w, dtype=np.float32)


def build_initial_traj_set(aprel_env: AprelEnvironment, saved_dir: Path) -> TrajectorySet:
    records_path = saved_dir / "trajectory_records.json"
    if not records_path.exists():
        raise FileNotFoundError(f"Missing trajectory records: {records_path}")

    records = json.loads(records_path.read_text())
    trajectories = [TrajectoryRecord.from_json(r).to_aprel(aprel_env) for r in records]
    return TrajectorySet(trajectories)


def prefill_expert_demos_into_buffer(
    replay_buffer: ReplayBuffer,
    demos: List[Dict[str, np.ndarray]],
    weights: np.ndarray,
) -> None:
    """
    Store expert transitions into replay buffer, labeling each transition with
    reward computed post-hoc from the full trajectory using current weights.
    """
    for demo in demos:
        states = demo["state_trajectory"]          # (T, 22)
        actions = demo["action_trajectory"]        # (T, 4)
        next_states = demo["next_state_trajectory"]
        dones = demo["done_trajectory"].flatten()

        T = len(actions)
        if T <= 0:
            continue

        # Build traj_pairs in dict format so feature_function can parse them.
        traj_pairs: List[Tuple[Dict[str, np.ndarray], np.ndarray]] = []
        for t in range(T):
            obs_dict = {
                "observation": states[t, :19].astype(np.float32),
                "desired_goal": states[t, 19:22].astype(np.float32),
                "achieved_goal": states[t, 7:10].astype(np.float32),
            }
            traj_pairs.append((obs_dict, actions[t]))

        features = feature_function(traj_pairs)  # (d,)
        episode_reward = float(np.dot(weights, features))
        reward_per_step = episode_reward / max(T, 1)

        for t in range(T):
            flat_obs = np.concatenate(
                [states[t, :19].astype(np.float32), states[t, 19:22].astype(np.float32)]
            )
            flat_next_obs = np.concatenate(
                [next_states[t, :19].astype(np.float32), states[t, 19:22].astype(np.float32)]
            )
        replay_buffer.add(
            flat_obs,
            flat_next_obs,
            actions[t].astype(np.float32),
            reward_per_step,
            bool(dones[t]),
        )


def rollout_one_episode_with_frames(
    env,
    agent: AWACAgent,
) -> Tuple[List[Tuple[Dict[str, np.ndarray], np.ndarray]], List[Tuple[np.ndarray, np.ndarray, np.ndarray, bool]], List[np.ndarray]]:
    """
    Roll out the current policy for ONE episode and return:
      - traj_pairs: list of (obs_dict, action) for feature_function
      - transitions: list of (flat_obs, flat_next_obs, action, done) for replay buffer
      - frames: list of RGB frames for saving an mp4 clip
    """
    obs_dict, _ = env.reset()
    flat_obs = reconstruct_state(obs_dict).astype(np.float32)

    frames: List[np.ndarray] = [capture_frame(env)]
    traj_pairs: List[Tuple[Dict[str, np.ndarray], np.ndarray]] = []
    transitions: List[Tuple[np.ndarray, np.ndarray, np.ndarray, bool]] = []

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = agent.get_action(flat_obs, deterministic=False).astype(np.float32)

        next_obs_dict, _, terminated, truncated, info = env.step(action)
        flat_next_obs = reconstruct_state(next_obs_dict).astype(np.float32)

        traj_pairs.append((obs_dict, action))
        transitions.append(
            (
                flat_obs.copy(),
                flat_next_obs.copy(),
                action.copy(),
                bool(terminated or truncated),
            )
        )

        obs_dict = next_obs_dict
        flat_obs = flat_next_obs
        frames.append(capture_frame(env))

    return traj_pairs, transitions, frames


def evaluate_success_rate(agent: AWACAgent, n_runs: int = 10) -> float:
    """Deterministic evaluation: average success over n_runs."""
    eval_env = make_rollout_env(render=False)
    successes = 0

    for _ in range(n_runs):
        obs_dict, _ = eval_env.reset()
        flat_obs = reconstruct_state(obs_dict).astype(np.float32)

        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = agent.get_action(flat_obs, deterministic=True)
            obs_dict, _, terminated, truncated, info = eval_env.step(action.astype(np.float32))
            flat_obs = reconstruct_state(obs_dict).astype(np.float32)

        if info.get("is_success", False):
            successes += 1

    eval_env.close()
    return successes / n_runs


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    saved_dir = repo_root / "saved"

    # Where to store online artifacts
    online_dir = saved_dir / "online_awac"
    clips_dir = online_dir / "clips"
    models_dir = online_dir / "policy_models"
    metrics_path = online_dir / "metrics_online_awac.csv"
    online_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Parameters you can tune ----------------
    max_steps = 500_000
    learning_starts = 2_000
    eval_interval = 1_000

    # How many preference queries?
    # If you set this low, you will only get human feedback in early episodes.
    queries_per_episode = 1
    max_total_queries = 10  # keep small unless your teacher can do more

    acquisition_function = "mutual_information"
    belief_delay = 0.5

    # AWAC dims (matches your environment + wrappers)
    OBS_DIM = 22
    ACTION_DIM = 4

    # ---------------- Build APReL environment + trajectories ----------------
    # Use an env instance for APReL (feature computation + action_space).
    # Keep render=False for APReL environment.
    aprel_base_env = make_rollout_env(render=False)
    aprel_env = AprelEnvironment(AprelGymAdapter(aprel_base_env), feature_func=feature_function)

    traj_set = build_initial_traj_set(aprel_env, saved_dir)

    dim = traj_set[0].features.shape[0]
    initial_weights = util_funs.get_random_normalized_vector(dim)

    user_model = SoftmaxUser({"weights": initial_weights})
    belief = SamplingBasedBelief(user_model, [], {"weights": initial_weights})

    optimizer = QueryOptimizerDiscreteTrajectorySet(traj_set)
    human = HumanUser(delay=belief_delay)

    # Only used to define query type and K (we will let optimizer overwrite slate).
    query_template = PreferenceQuery(traj_set[:2])

    # ---------------- Build AWAC agent + replay buffer ----------------
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    agent = AWACAgent(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        device=device,
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        lambda_awac=0.3,
        n_action_samples=16,
        batch_size=256,
    )

    replay_buffer = ReplayBuffer(OBS_DIM, ACTION_DIM, capacity=300_000)

    # Load expert demos for replay buffer warm-start (with CURRENT weights at first episode).
    demos = prepare_demo_pool(repo_root / "demo_data" / "PickAndPlace")

    # ---------------- Metrics logging ----------------
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["steps", "success_rate", "ep_reward", "ep_length"],
        )
        writer.writeheader()
        # Learning curves tracked in memory
        success_rates: List[float] = []
        checkpoints: List[int] = []
        ep_rewards_window: List[float] = []
        ep_lengths_window: List[float] = []

        rollout_env = make_rollout_env(render=False)  

        total_steps = 0
        steps_since_eval = 0
        queries_used = 0
        episode_idx = 0
        experts_prefilled = False

        while total_steps < max_steps:
            # ---------------------------------------------------------
            # 1) Ask human feedback (online preference learning)
            # ---------------------------------------------------------
            weights = get_weights_from_belief(belief)

            for _ in range(queries_per_episode):
                if queries_used >= max_total_queries:
                    break

                queries, _ = optimizer.optimize(acquisition_function, belief, query_template)
                responses = human.respond(queries[0])  # keyboard selection
                belief.update(Preference(queries[0], responses[0]))
                queries_used += 1

            weights = get_weights_from_belief(belief)

            # ---------------------------------------------------------
            # 2) Roll out current AWAC policy for one episode
            # ---------------------------------------------------------
            traj_pairs, transitions, frames = rollout_one_episode_with_frames(rollout_env, agent)
            ep_steps = len(transitions)

            # 3) Recover episode reward from the freshly updated reward weights
            features = feature_function(traj_pairs)  # (3,)
            episode_reward = float(np.dot(weights, features))
            reward_per_step = episode_reward / max(ep_steps, 1)

            # ---------------------------------------------------------
            # 4) Store transitions into replay buffer with this episode reward
            # ---------------------------------------------------------
            for flat_obs, flat_next_obs, action, done in transitions:
                replay_buffer.add(flat_obs, flat_next_obs, action, reward_per_step, done)

            # Optional deliverable-style warm start: prefill experts once (after first preference update)
            if (not experts_prefilled) and (replay_buffer.size() >= 1):
                prefill_expert_demos_into_buffer(replay_buffer, demos, weights)
                experts_prefilled = True

            # ---------------------------------------------------------
            # 5) Update the AWAC policy
            # ---------------------------------------------------------
            total_steps += ep_steps
            steps_since_eval += ep_steps

            if replay_buffer.size() >= learning_starts:
                agent.train(replay_buffer, gradient_steps=ep_steps)

            # ---------------------------------------------------------
            # 6) Add rolled-out trajectory to APReL dataset + save clip
            # ---------------------------------------------------------
            clip_path = clips_dir / f"online_awac_ep{episode_idx:04d}_step{total_steps:07d}.mp4"
            save_frames_to_mp4(frames, out_path=clip_path)

            online_record = TrajectoryRecord(
                clip_path=str(clip_path),
                features=features,
            )
            traj_set.append(online_record.to_aprel(aprel_env))

            ep_rewards_window.append(episode_reward)
            ep_lengths_window.append(float(ep_steps))

            # ---------------------------------------------------------
            # 7) Evaluate + checkpoint
            # ---------------------------------------------------------
            if steps_since_eval >= eval_interval:
                rate = evaluate_success_rate(agent, n_runs=10)
                success_rates.append(rate)
                checkpoints.append(total_steps)

                # CSV log
                writer.writerow({
                    "steps": total_steps,
                    "success_rate": rate,
                    "ep_reward": float(np.mean(ep_rewards_window)) if ep_rewards_window else np.nan,
                    "ep_length": float(np.mean(ep_lengths_window)) if ep_lengths_window else np.nan,
                })
                f.flush()

                print(f"[{total_steps:>7d} steps | ep {episode_idx}] success_rate={rate:.2f} queries_used={queries_used}")

                # Save checkpoint
                agent.save(str(models_dir / f"awac_online_{total_steps}"))

                steps_since_eval = 0
                ep_rewards_window.clear()
                ep_lengths_window.clear()

            episode_idx += 1

        rollout_env.close()
        aprel_base_env.close()

        # ---------------- Learning curve ----------------
        plt.figure(figsize=(12, 5))
        plt.plot(checkpoints, success_rates)
        plt.xlabel("Environment Steps")
        plt.ylabel("Average Success Rate (10 runs)")
        plt.title("Online AWAC (Extra Credit) Learning Curve")
        plt.tight_layout()

        plot_path = online_dir / "learning_curve_online_awac.png"
        plt.savefig(str(plot_path), dpi=150)
        plt.close()

        print(f"Online training finished. Metrics: {metrics_path}")
        print(f"Learning curve: {plot_path}")


if __name__ == "__main__":
    main()