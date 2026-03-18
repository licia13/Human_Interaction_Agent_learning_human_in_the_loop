from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import aprel.utils.util_functions as util_funs
from alg.banana import feature_function
from aprel.basics.environment import Environment
from aprel.basics.trajectory import TrajectorySet
from aprel.learning.belief_models import SamplingBasedBelief
from aprel.learning.data_types import Preference, PreferenceQuery
from aprel.learning.user_models import SoftmaxUser,HumanUser
from aprel.querying.query_optimizer import QueryOptimizerDiscreteTrajectorySet
from envs.task_envs import PnPNewRobotEnv
from utils.env_wrappers import (
    ActionNormalizer,
    AprelGymAdapter,
    ResetWrapper,
    TimeLimitWrapper,
    TrajectoryRecord,
)


def setup_environment(*, render: bool = False) -> Any:
    env = PnPNewRobotEnv(render=render)
    env = ResetWrapper(env)
    env = ActionNormalizer(env)
    env = TimeLimitWrapper(env, max_steps=150)
    env.reset(seed=0)
    return env


def learn_weights(
    traj_set: TrajectorySet,
    *,
    num_queries: int = 10,
    seed: int = 0,
    acquisition_function: str = "mutual_information"
) -> np.ndarray:
    np.random.seed(seed)

    dim = traj_set[0].features.shape[0]
    initial_weights = util_funs.get_random_normalized_vector(dim)

    user_model = SoftmaxUser({"weights": initial_weights})
    belief = SamplingBasedBelief(user_model, [], {"weights": initial_weights})
    optimizer = QueryOptimizerDiscreteTrajectorySet(traj_set)
    human = HumanUser(delay=0.5)

    query_template = PreferenceQuery(traj_set[:2])

    for i in range(num_queries):
        print(f"\nQuery {i + 1} / {num_queries}  (acquisition: {acquisition_function})")
        queries, _ = optimizer.optimize(acquisition_function, belief, query_template)
        responses = human.respond(queries[0])
        belief.update(Preference(queries[0], responses[0]))

    return np.array(belief.mean["weights"], dtype=np.float32)


def save_weights(weights: np.ndarray, out_path: Path) -> None:
    """Serialise learned feature weights to a two-column CSV file.

    Args:
        weights: 1-D array of reward feature weights.
        out_path: Destination file path (will be created or overwritten).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_index", "weight"])
        for i, val in enumerate(weights.tolist()):
            writer.writerow([i, float(val)])


def main() -> None:
    """load trajectory records, run preference learning, and save weights.

    1. Load TrajectoryRecord objects from
       ``<repo_root>/saved/trajectory_records.json`` (produced by banana.py).
    2. Convert records into an APReL TrajectorySet.
    3. Run learn_weights to interactively collect human preferences and
       infer posterior mean reward weights.
    4. Save the resulting weights to ``<repo_root>/saved/feature_weights.csv``.

    Raises:
        FileNotFoundError: If ``trajectory_records.json`` does not exist.
    """
    repo_root = Path(__file__).resolve().parents[1]
    saved_dir = repo_root / "saved"

    records_path = saved_dir / "trajectory_records.json"
    if not records_path.exists():
        raise FileNotFoundError(
            f"Missing trajectory records at {records_path}. "
            "Run banana.py first to generate clips and records."
        )

    env = setup_environment(render=False)
    aprel_env = Environment(AprelGymAdapter(env), feature_func=feature_function)

    records = json.loads(records_path.read_text())
    trajectories = TrajectorySet(
        [TrajectoryRecord.from_json(r).to_aprel(aprel_env) for r in records]
    )

    out_path = saved_dir / "feature_weights.csv"

    try:
        weights = learn_weights(trajectories, num_queries=10, seed=0)
    except Exception as e:
        raise
    finally:
        env.close()

    save_weights(weights, out_path)
    print(f"\nSaved learned weights to: {out_path}")


if __name__ == "__main__":
    main()