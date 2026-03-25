
from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from envs.task_envs import PnPNewRobotEnv
from utils.demos import prepare_demo_pool
from utils.env_wrappers import ActionNormalizer, ResetWrapper, TimeLimitWrapper, reconstruct_state
from alg.banana import feature_function

REWARD_SCALE = 20.0


# ─────────────────────────────────────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────────────────────────────────────

LOG_STD_MIN = -5
LOG_STD_MAX = 2


def load_weights(path: Path) -> np.ndarray:
    """Read learned reward weights from feature_weights.csv."""
    weights = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            weights.append(float(row["weight"]))
    return np.array(weights, dtype=np.float32)


def make_env(render: bool = False):
    """Standard wrapper stack — preserves dict observation for feature_function."""
    env = PnPNewRobotEnv(render=render)
    env = ResetWrapper(env)
    env = ActionNormalizer(env)
    env = TimeLimitWrapper(env, max_steps=150)
    return env


# ─────────────────────────────────────────────────────────────────────────────
#  Neural networks
# ─────────────────────────────────────────────────────────────────────────────

class GaussianActor(nn.Module):
    """Squashed-Gaussian policy (tanh-bounded actions in [-1, 1])."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def _dist(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std.exp()

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return (action, log_prob)."""
        mean, std = self._dist(obs)
        dist = Normal(mean, std)
        x = dist.rsample()
        action = torch.tanh(x)
        # log-prob with tanh change-of-variables correction
        log_prob = (dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return action, log_prob

    def log_prob_of(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Log-probability of given actions under the current policy."""
        mean, std = self._dist(obs)
        dist = Normal(mean, std)
        actions_c = actions.clamp(-1 + 1e-6, 1 - 1e-6)
        raw = torch.atanh(actions_c)
        log_prob = (dist.log_prob(raw) - torch.log(1 - actions.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self._dist(obs)
        return torch.tanh(mean)


class TwinCritic(nn.Module):
    """Twin Q-networks to reduce over-estimation bias."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()

        def _mlp():
            return nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        self.q1 = _mlp()
        self.q2 = _mlp()

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)


# ─────────────────────────────────────────────────────────────────────────────
#  Replay buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Simple numpy-backed replay buffer."""

    def __init__(self, obs_dim: int, action_dim: int, capacity: int = 300_000):
        self.capacity = capacity
        self.pos = 0
        self.full = False
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, next_obs, action, reward: float, done: bool):
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def size(self) -> int:
        return self.capacity if self.full else self.pos

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size(), size=batch_size)
        return (
            torch.FloatTensor(self.obs[idx]).to(device),
            torch.FloatTensor(self.next_obs[idx]).to(device),
            torch.FloatTensor(self.actions[idx]).to(device),
            torch.FloatTensor(self.rewards[idx]).to(device),
            torch.FloatTensor(self.dones[idx]).to(device),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  AWAC agent
# ─────────────────────────────────────────────────────────────────────────────

class AWACAgent:
    """
    Advantage-Weighted Actor-Critic agent.

    Critic update: standard TD with twin Q-networks and target networks.
    Actor update:  advantage-weighted regression (no entropy term).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        lambda_awac: float = 0.3,
        n_action_samples: int = 16,
        batch_size: int = 256,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.lambda_awac = lambda_awac
        self.n_action_samples = n_action_samples
        self.batch_size = batch_size

        self.actor = GaussianActor(obs_dim, action_dim).to(device)
        self.critic = TwinCritic(obs_dim, action_dim).to(device)
        self.critic_target = deepcopy(self.critic)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    # ── Inference ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        if deterministic:
            action = self.actor.deterministic(obs_t)
        else:
            action, _ = self.actor.sample(obs_t)
        return action.squeeze(0).cpu().numpy()

    # ── Training ───────────────────────────────────────────────────────────

    def train(
        self, replay_buffer: ReplayBuffer, gradient_steps: int
    ) -> Tuple[float, float]:
        """Run `gradient_steps` updates and return (mean_actor_loss, mean_critic_loss)."""
        if replay_buffer.size() < self.batch_size:
            return float("nan"), float("nan")

        actor_losses: List[float] = []
        critic_losses: List[float] = []

        for _ in range(gradient_steps):
            obs, next_obs, actions, rewards, dones = replay_buffer.sample(
                self.batch_size, self.device
            )

            # ── Critic update ─────────────────────────────────────────────
            with torch.no_grad():
                # V(s') via Monte Carlo: sample K actions from the policy
                B = next_obs.shape[0]
                K = self.n_action_samples
                next_obs_rep = next_obs.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
                sampled_actions, _ = self.actor.sample(next_obs_rep)
                q_next = self.critic_target.q_min(next_obs_rep, sampled_actions)
                v_next = q_next.reshape(B, K, 1).mean(dim=1)          # (B, 1)
                td_target = rewards + self.gamma * (1.0 - dones) * v_next

            q1, q2 = self.critic(obs, actions)
            critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            critic_losses.append(critic_loss.item())

            # ── Actor update (AWAC) ────────────────────────────────────────
            with torch.no_grad():
                # V(s) via Monte Carlo
                obs_rep = obs.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
                sa_samples, _ = self.actor.sample(obs_rep)
                q_s = self.critic_target.q_min(obs_rep, sa_samples)
                v_s = q_s.reshape(B, K, 1).mean(dim=1)                 # (B, 1)

                # A(s,a) = Q(s,a) - V(s)
                q_sa = self.critic_target.q_min(obs, actions)
                advantage = q_sa - v_s                                  # (B, 1)

                # Advantage weights — clamp to prevent explosion
                adv_weights = torch.exp(advantage / self.lambda_awac).clamp(max=20.0)

            log_probs = self.actor.log_prob_of(obs, actions)            # (B, 1)
            actor_loss = -(log_probs * adv_weights).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            actor_losses.append(actor_loss.item())

            # ── Soft target update ─────────────────────────────────────────
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        return float(np.mean(actor_losses)), float(np.mean(critic_losses))

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, path + ".pt")

    def load(self, path: str):
        ckpt = torch.load(path + ".pt", map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])


# ─────────────────────────────────────────────────────────────────────────────
#  Demo loading
# ─────────────────────────────────────────────────────────────────────────────

def load_demos_into_buffer(
    replay_buffer: ReplayBuffer,
    demos: List[Dict],
    weights: np.ndarray,
) -> None:
    """Pre-fill replay buffer with expert demos using trajectory-level reward."""
    for demo in demos:
        states = demo["state_trajectory"]
        actions = demo["action_trajectory"]
        next_states = demo["next_state_trajectory"]
        dones = demo["done_trajectory"].flatten()
        T = len(actions)

        # Build traj_pairs with dict obs so feature_function can parse them
        traj_pairs = []
        for t in range(T):
            obs_dict = {
                "observation": states[t, :19].astype(np.float32),
                "desired_goal": states[t, 19:22].astype(np.float32),
                "achieved_goal": states[t, 7:10].astype(np.float32),
            }
            traj_pairs.append((obs_dict, actions[t]))

        # Trajectory-level reward → distributed equally across steps
        features = feature_function(traj_pairs)
        traj_reward = float(np.dot(weights, features))
        reward_per_step = REWARD_SCALE * traj_reward / max(T, 1)

        for t in range(T):
            flat_obs = np.concatenate([
                states[t, :19].astype(np.float32),
                states[t, 19:22].astype(np.float32),
            ])
            flat_next_obs = np.concatenate([
                next_states[t, :19].astype(np.float32),
                states[t, 19:22].astype(np.float32),
            ])
            replay_buffer.add(
                flat_obs, flat_next_obs,
                actions[t].astype(np.float32),
                reward_per_step, bool(dones[t]),
            )


# ─────────────────────────────────────────────────────────────────────────────
#  Episodic rollout
# ─────────────────────────────────────────────────────────────────────────────

def rollout_episode(env, agent: AWACAgent):
    """Roll out the current policy for ONE full episode.

    Returns:
        traj_pairs  -- list of (obs_dict, action) for feature_function
        transitions -- list of (flat_obs, flat_next_obs, action, done)
        ep_steps    -- number of steps
    """
    obs_dict, _ = env.reset()
    flat_obs = reconstruct_state(obs_dict).astype(np.float32)

    traj_pairs = []
    transitions = []
    terminated = truncated = False

    while not (terminated or truncated):
        action = agent.get_action(flat_obs, deterministic=False)

        next_obs_dict, _, terminated, truncated, info = env.step(action)
        flat_next_obs = reconstruct_state(next_obs_dict).astype(np.float32)

        traj_pairs.append((obs_dict, action))
        transitions.append((
            flat_obs.copy(),
            flat_next_obs.copy(),
            action.astype(np.float32),
            terminated or truncated,
        ))

        obs_dict = next_obs_dict
        flat_obs = flat_next_obs

    return traj_pairs, transitions, len(transitions)


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(agent: AWACAgent, n_runs: int = 10) -> float:
    """Run n_runs deterministic episodes and return success rate."""
    eval_env = make_env(render=False)
    successes = 0
    for _ in range(n_runs):
        obs_dict, _ = eval_env.reset()
        flat_obs = reconstruct_state(obs_dict).astype(np.float32)
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.get_action(flat_obs, deterministic=True)
            obs_dict, _, terminated, truncated, info = eval_env.step(action)
            flat_obs = reconstruct_state(obs_dict).astype(np.float32)
            if info.get("is_success", False):
                successes += 1
    eval_env.close()
    return successes / n_runs


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    saved_dir = repo_root / "saved"
    models_dir = saved_dir / "policy_models_awac"
    models_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Dimensions ───────────────────────────────────────────────────────────
    OBS_DIM = 22    # 19 obs + 3 goal
    ACTION_DIM = 4  # dx, dy, dz, gripper_cmd

    # ── Load reward weights ───────────────────────────────────────────────────
    weights = load_weights(saved_dir / "feature_weights.csv")
    print(f"Loaded weights: {weights}")

    # ── Build AWAC agent ──────────────────────────────────────────────────────
    agent = AWACAgent(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        device=device,
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        # Softer advantage weighting is more stable for sparse/indirect rewards.
        lambda_awac=1.0,
        # More action samples makes V(s) less noisy.
        n_action_samples=32,
        batch_size=256,
    )
    replay_buffer = ReplayBuffer(OBS_DIM, ACTION_DIM, capacity=300_000)

    # ── Pre-fill buffer with expert demos ────────────────────────────────────
    demos = prepare_demo_pool(repo_root / "demo_data" / "PickAndPlace")
    load_demos_into_buffer(replay_buffer, demos, weights)
    print(f"Loaded {len(demos)} expert demos into replay buffer "
          f"({replay_buffer.size()} transitions)")

    # ── Open metrics CSV ──────────────────────────────────────────────────────
    metrics_path = saved_dir / "metrics_awac.csv"
    metrics_file = open(metrics_path, "w", newline="")
    _fields = [
        "steps",
        "success_rate",
        "ep_reward",
        "ep_length",
        "actor_loss",
        "critic_loss",
        "replay_size",
        "reward_scale",
    ]
    metrics_writer = csv.DictWriter(metrics_file, fieldnames=_fields)
    metrics_writer.writeheader()

    # ── Episodic training loop ────────────────────────────────────────────────
    rollout_env = make_env(render=False)

    total_steps = 0
    max_steps = 500_000
    eval_interval = 1_000
    learning_starts = 500
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
        traj_pairs, transitions, ep_steps = rollout_episode(rollout_env, agent)

        # Step 2 — recover trajectory reward post-hoc from the full trajectory
        features = feature_function(traj_pairs)
        traj_reward = float(np.dot(weights, features))
        reward_per_step = REWARD_SCALE * traj_reward / max(ep_steps, 1)

        w_ep_rewards.append(traj_reward)
        w_ep_lengths.append(float(ep_steps))

        # Step 3 — save all transitions to the replay buffer
        for flat_obs, flat_next_obs, action, done in transitions:
            replay_buffer.add(flat_obs, flat_next_obs, action, reward_per_step, done)

        total_steps += ep_steps
        steps_since_eval += ep_steps

        # Step 4 — update the policy once enough data is collected
        if replay_buffer.size() >= learning_starts:
            # Avoid huge update bursts early in training.
            al, cl = agent.train(replay_buffer, gradient_steps=min(ep_steps, 64))
            if not np.isnan(al):
                w_actor_losses.append(al)
            if not np.isnan(cl):
                w_critic_losses.append(cl)

        # ── Evaluate + checkpoint every eval_interval steps ──────────────────
        if steps_since_eval >= eval_interval:
            rate = evaluate(agent)
            success_rates.append(rate)
            checkpoints.append(total_steps)
            print(f"[{total_steps:>7d} steps]  success rate: {rate:.2f}")
            agent.save(str(models_dir / f"awac_{total_steps}"))

            metrics_writer.writerow({
                "steps":        total_steps,
                "success_rate": rate,
                "ep_reward":    np.mean(w_ep_rewards)    if w_ep_rewards    else np.nan,
                "ep_length":    np.mean(w_ep_lengths)    if w_ep_lengths    else np.nan,
                "actor_loss":   np.mean(w_actor_losses)  if w_actor_losses  else np.nan,
                "critic_loss":  np.mean(w_critic_losses) if w_critic_losses else np.nan,
                "replay_size":  replay_buffer.size(),
                "reward_scale": REWARD_SCALE,
            })
            metrics_file.flush()

            w_ep_rewards.clear(); w_ep_lengths.clear()
            w_actor_losses.clear(); w_critic_losses.clear()
            steps_since_eval = 0

    rollout_env.close()
    metrics_file.close()

    # ── Plot learning curve ───────────────────────────────────────────────────
    plt.figure(figsize=(12, 5))
    plt.plot(checkpoints, success_rates)
    plt.xlabel("Environment Steps")
    plt.ylabel("Average Success Rate (10 runs)")
    plt.title("AWAC Policy Learning Curve")
    plt.tight_layout()
    plot_path = saved_dir / "learning_curve_awac.png"
    plt.savefig(str(plot_path))
    print(f"Learning curve saved to {plot_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
