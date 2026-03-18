from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import SAC

from envs.task_envs import PnPNewRobotEnv
from utils.demos import prepare_demo_pool
from utils.env_wrappers import ActionNormalizer, ResetWrapper, TimeLimitWrapper, reconstruct_state



def load_weights(path: Path) -> np.ndarray:
    weights = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            weights.append(float(row["weight"]))
    return np.array(weights, dtype=np.float32)


class LearnedRewardWrapper(gym.Wrapper):
    def __init__(self, env, weights: np.ndarray):
        super().__init__(env)
        self.weights = weights

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        raw = obs["observation"]
        goal = obs["desired_goal"]
        ee_pos = raw[0:3]
        banana_pos = raw[7:10]
        f1 = -float(np.linalg.norm(ee_pos - banana_pos))
        f2 = -float(np.linalg.norm(banana_pos - goal))
        f3 = f2
        reward = float(np.dot(self.weights, [f1, f2, f3]))
        if info.get("is_success", False):
            reward += 100.0
        return obs, reward, terminated, truncated, info

class FlatObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_dim = (env.observation_space["observation"].shape[0]
                   + env.observation_space["desired_goal"].shape[0])
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def observation(self, obs):
        return reconstruct_state(obs).astype(np.float32)

def make_env(weights: np.ndarray, render: bool = False):
    env = PnPNewRobotEnv(render=render)
    env = ResetWrapper(env)
    env = ActionNormalizer(env)
    env = TimeLimitWrapper(env, max_steps=150)
    env = LearnedRewardWrapper(env, weights)
    env = FlatObsWrapper(env)
    return env

def load_demos_into_buffer(model, demos: List[Dict], weights: np.ndarray) -> None:
    for demo in demos:
        states = demo["state_trajectory"]
        actions = demo["action_trajectory"]
        next_states = demo["next_state_trajectory"]
        dones = demo["done_trajectory"].flatten()

        for t in range(len(actions)):
            obs = states[t, :19].astype(np.float32)
            goal = states[t, 19:22].astype(np.float32)
            flat_obs = np.concatenate([obs, goal])

            next_obs = next_states[t, :19].astype(np.float32)
            flat_next_obs = np.concatenate([next_obs, goal])

            action = actions[t].astype(np.float32)

            ee_pos = obs[0:3]
            banana_pos = obs[6:9]
            f1 = -float(np.linalg.norm(ee_pos - banana_pos))
            f2 = -float(np.linalg.norm(banana_pos - goal))
            reward = float(np.dot(weights, [f1, f2, f2]))

            done = bool(dones[t])
            model.replay_buffer.add(
                flat_obs, flat_next_obs, action,
                np.array([reward]), np.array([done]), [{}]
            )

def evaluate(weights: np.ndarray, model, n_runs: int = 10) -> float:
    eval_env = make_env(weights, render=False)
    successes = 0
    for _ in range(n_runs):
        obs, _ = eval_env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = eval_env.step(action)
            if info.get("is_success", False):
                successes += 1
    eval_env.close()
    return successes / n_runs

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    saved_dir = repo_root / "saved"
    models_dir = saved_dir / "policy_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    weights = load_weights(saved_dir / "feature_weights.csv")
    print(f"Loaded weights: {weights}")

    env = make_env(weights, render=False)

    model = SAC(
        "MlpPolicy", env,
        verbose=0,
        learning_starts=2000,
        batch_size=256,
        buffer_size=300_000,
    )

    demos = prepare_demo_pool(repo_root / "demo_data" / "PickAndPlace")
    load_demos_into_buffer(model, demos, weights)
    print(f"Loaded {len(demos)} expert demos into replay buffer")

    total_steps = 0
    max_steps = 500_000
    eval_interval = 1_000
    success_rates = []
    checkpoints = []

    while total_steps < max_steps:
        model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
        total_steps += eval_interval

        rate = evaluate(weights, model)
        success_rates.append(rate)
        checkpoints.append(total_steps)
        print(f"[{total_steps:>7d} steps]  success rate: {rate:.2f}")

        model.save(str(models_dir / f"policy_{total_steps}"))

    env.close()

    plt.figure(figsize=(12, 5))
    plt.plot(checkpoints, success_rates)
    plt.xlabel("Environment Steps")
    plt.ylabel("Average Success Rate (10 runs)")
    plt.title("Policy Learning Curve")
    plt.tight_layout()
    plot_path = saved_dir / "learning_curve.png"
    plt.savefig(str(plot_path))
    print(f"Learning curve saved to {plot_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    saved_dir = repo_root / "saved"
    models_dir = saved_dir / "policy_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    weights = load_weights(saved_dir / "feature_weights.csv")
    print(f"Loaded weights: {weights}")

    env = make_env(weights, render=False)

    model = SAC(
        "MlpPolicy", env,
        verbose=0,
        learning_starts=2000,
        batch_size=256,
        buffer_size=300_000,
    )

    demos = prepare_demo_pool(repo_root / "demo_data" / "PickAndPlace")
    load_demos_into_buffer(model, demos, weights)
    print(f"Loaded {len(demos)} expert demos into replay buffer")

    total_steps = 0
    max_steps = 500_000
    eval_interval = 1_000
    success_rates = []
    checkpoints = []

    while total_steps < max_steps:
        model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
        total_steps += eval_interval

        rate = evaluate(weights, model)
        success_rates.append(rate)
        checkpoints.append(total_steps)
        print(f"[{total_steps:>7d} steps]  success rate: {rate:.2f}")

        model.save(str(models_dir / f"policy_{total_steps}"))

    env.close()

    plt.figure(figsize=(12, 5))
    plt.plot(checkpoints, success_rates)
    plt.xlabel("Environment Steps")
    plt.ylabel("Average Success Rate (10 runs)")
    plt.title("Policy Learning Curve")
    plt.tight_layout()
    plot_path = saved_dir / "learning_curve.png"
    plt.savefig(str(plot_path))
    print(f"Learning curve saved to {plot_path}")



def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    saved_dir = repo_root / "saved"
    models_dir = saved_dir / "policy_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    weights = load_weights(saved_dir / "feature_weights.csv")
    print(f"Loaded weights: {weights}")
    env = make_env(weights, render=False)
    model = SAC(
        "MlpPolicy", env,
        verbose=0,
        learning_starts=2000,
        batch_size=256,
        buffer_size=300_000,
    )
    demos = prepare_demo_pool(repo_root / "demo_data" / "PickAndPlace")
    load_demos_into_buffer(model, demos, weights)
    print(f"Loaded {len(demos)} expert demos into replay buffer")
    total_steps = 0
    max_steps = 500_000
    eval_interval = 1_000
    success_rates = []
    checkpoints = []
    while total_steps < max_steps:
        model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
        total_steps += eval_interval
        rate = evaluate(weights, model)
        success_rates.append(rate)
        checkpoints.append(total_steps)
        print(f"[{total_steps:>7d} steps]  success rate: {rate:.2f}")
        model.save(str(models_dir / f"policy_{total_steps}"))
    env.close()
    plt.figure(figsize=(12, 5))
    plt.plot(checkpoints, success_rates)
    plt.xlabel("Environment Steps")
    plt.ylabel("Average Success Rate (10 runs)")
    plt.title("Policy Learning Curve")
    plt.tight_layout()
    plot_path = saved_dir / "learning_curve.png"
    plt.savefig(str(plot_path))
    print(f"Learning curve saved to {plot_path}")
if __name__ == "__main__":
    main()

