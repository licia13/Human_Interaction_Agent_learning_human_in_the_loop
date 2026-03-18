from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np

from envs.task_envs import PnPNewRobotEnv
from utils.demos import prepare_demo_pool
from utils.env_wrappers import ActionNormalizer, ResetWrapper, TimeLimitWrapper, TrajectoryRecord


def feature_function(traj_pairs: List[Tuple[Dict[str, np.ndarray], np.ndarray]]) -> np.ndarray:
    if not traj_pairs:
        return np.zeros(3, dtype=np.float32)

    ee_to_banana_dists = []
    banana_to_plate_dists = []

    for state, action in traj_pairs:
        obs = state["observation"]
        goal = state["desired_goal"]
        ee_pos = obs[0:3]
        banana_pos = obs[7:10]
        ee_to_banana_dists.append(np.linalg.norm(ee_pos - banana_pos))
        banana_to_plate_dists.append(np.linalg.norm(banana_pos - goal))

    # Higher value = better (negative distance so closer = higher reward)
    f1 = -float(np.min(ee_to_banana_dists))       # arm approached banana
    f2 = -float(np.min(banana_to_plate_dists))     # banana got close to plate
    f3 = -float(banana_to_plate_dists[-1])         # banana ended close to plate

    return np.array([f1, f2, f3], dtype=np.float32)

def capture_frame(env: Any, width: int = 320, height: int = 240) -> np.ndarray:
    """Render the current simulation state to an image via PyBullet offscreen rendering.

    Attempts a hardware-accelerated render using ER_BULLET_HARDWARE_OPENGL.
    Falls back to a black frame of the requested dimensions if rendering fails

    Args:
        env: A gym environment whose .unwrapped.sim.physics_client
            exposes the PyBullet physics client.
        width: Output image width in pixels.
        height: Output image height in pixels.

    Returns:
        A uint8 array of shape (height, width, 3) in RGB channel order.
    """
    try:
        base = getattr(env, "unwrapped", env)
        pc = base.sim.physics_client

        view_matrix = pc.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.0, 0.0, 0.0],
            distance=1.0,
            yaw=45,
            pitch=-30,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = pc.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=float(width) / float(height),
            nearVal=0.01,
            farVal=10.0,
        )

        _, _, px, _, _ = pc.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pc.ER_BULLET_HARDWARE_OPENGL,
        )
        rgba = np.reshape(px, (height, width, 4))
        return rgba[:, :, :3].astype(np.uint8, copy=False)

    except Exception:
        return np.zeros((height, width, 3), dtype=np.uint8)


def setup_environment(*, render: bool = False) -> Any:
    env = PnPNewRobotEnv(render=render)
    env = ResetWrapper(env)
    env = ActionNormalizer(env)
    env = TimeLimitWrapper(env, max_steps=150)
    env.reset(seed=0)
    return env



def rollout(
    env: Any,
    action_seq: np.ndarray,
    *,
    options: Optional[Dict[str, Any]] = None,
    max_steps: int = 150,
) -> Tuple[List[Tuple[Dict[str, np.ndarray], np.ndarray]], List[np.ndarray]]:
    T = min(int(action_seq.shape[0]), int(max_steps))
    traj_pairs: List[Tuple[Dict[str, np.ndarray], np.ndarray]] = []
    frames: List[np.ndarray] = []

    state, _info = env.reset(seed=0, options=options)
    frames.append(capture_frame(env))

    for t in range(T):
        action = action_seq[t]
        next_state, reward, terminated, truncated, info = env.step(action)
        traj_pairs.append((state, action))
        frames.append(capture_frame(env))
        state = next_state
        if terminated or truncated:
            break

    return traj_pairs, frames


def random_rollout(
    env: Any,
    *,
    max_steps: int = 150,
) -> Tuple[List[Tuple[Dict[str, np.ndarray], np.ndarray]], List[np.ndarray]]:
    traj_pairs: List[Tuple[Dict[str, np.ndarray], np.ndarray]] = []
    frames: List[np.ndarray] = []

    state, _info = env.reset(seed=None)
    frames.append(capture_frame(env))

    for _ in range(max_steps):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        traj_pairs.append((state, action))
        frames.append(capture_frame(env))
        state = next_state
        if terminated or truncated:
            break

    return traj_pairs, frames


def main() -> None:
    """generate expert and random trajectory clips, then serialise records.

    1. Load all expert demos from ``<repo_root>/demo_data/PickAndPlace``.
    2. Roll out each demo in the environment, saving an MP4 clip and computing
       feature vectors.
    3. Generate 10 additional random-policy clips.
    4. Serialise all TrajectoryRecord objects to
       ``<repo_root>/saved/trajectory_records.json``.
    """
    env = setup_environment(render=True)

    repo_root = Path(__file__).resolve().parents[1]
    demo_dir = repo_root / "demo_data" / "PickAndPlace"
    saved_dir = repo_root / "saved"
    clips_dir = saved_dir / "clips"
    saved_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    demos = prepare_demo_pool(demo_dir, verbose=True)
    print(f"\nLoaded {len(demos)} expert demos from: {demo_dir}")

    saved_records: List[TrajectoryRecord] = []

    fps = 30
    writer_kwargs: Dict[str, Any] = dict(
        fps=fps,
        codec="libx264",
        ffmpeg_params=["-preset", "ultrafast", "-crf", "28"],
    )

    print(f"\nGenerating {len(demos)} expert clips")

    for i, demo in enumerate(demos):
        obj_pos = demo["state_trajectory"][0][7:10]
        options = {"whether_random": False, "object_pos": obj_pos}
        traj_pairs, frames = rollout(env, demo["action_trajectory"], options=options)
        features = feature_function(traj_pairs)
        clip_path = str(clips_dir / f"expert_{i:02d}.mp4")
        with imageio.get_writer(clip_path, **writer_kwargs) as writer:
            for frame in frames:
                writer.append_data(frame)
        saved_records.append(TrajectoryRecord(clip_path=clip_path, features=features))


    print(f"\nGenerating 10 random clips")
    for i in range(10):
        traj_pairs, frames = random_rollout(env)
        features = feature_function(traj_pairs)
        clip_path = str(clips_dir / f"random_{i:02d}.mp4")
        with imageio.get_writer(clip_path, **writer_kwargs) as writer:
            for frame in frames:
                writer.append_data(frame)
        saved_records.append(TrajectoryRecord(clip_path=clip_path, features=features))




    env.close()

    out_path = saved_dir / "trajectory_records.json"
    with open(out_path, "w") as f:
        json.dump([r.to_json() for r in saved_records], f, indent=2)

    print(f"Saved {len(saved_records)} trajectory records to {out_path}")


if __name__ == "__main__":
    main()