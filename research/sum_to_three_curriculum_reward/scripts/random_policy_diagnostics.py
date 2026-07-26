"""Measure event rarity for random normalized actions on SumToThree."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np


def make_env(start_distribution: str, seed: int):
    try:
        from easydict import EasyDict
        from zoo.pooltool.sum_to_three.envs.sum_to_three_env import SumToThreeEnv
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing training dependency {exc.name!r}. Activate the LightZero Python 3.10 environment and run pip install -e ."
        ) from exc

    cfg = EasyDict(SumToThreeEnv.default_config())
    cfg.update(
        observation_type="coordinate",
        reward_algorithm="binary",
        start_distribution=start_distribution,
        curriculum_enabled=False,
        emit_step_diagnostics=True,
        raw_observation=False,
    )
    env = SumToThreeEnv(cfg)
    env.seed(seed, dynamic_seed=False)
    return env


def run_distribution(
    start_distribution: str,
    episodes: int,
    seed: int,
) -> Dict[str, object]:
    env = make_env(start_distribution, seed)
    action_rng = np.random.default_rng(seed + 10000)
    cushion_histogram: Counter[int] = Counter()
    episode_records: List[Dict[str, object]] = []
    total_contacts = 0
    total_successes = 0
    total_shots = 0

    try:
        for episode_id in range(episodes):
            initial_obs = env.reset()["observation"].tolist()
            episode_contacts = 0
            episode_successes = 0
            for _ in range(env.cfg.episode_length):
                action = action_rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
                timestep = env.step(action)
                outcome = timestep.info["shot_outcome"]
                contacted = bool(outcome["contacted_object"])
                cushion_count = int(outcome["linear_cushion_count"])
                success = bool(timestep.info["binary_reward"])
                total_shots += 1
                total_contacts += int(contacted)
                total_successes += int(success)
                episode_contacts += int(contacted)
                episode_successes += int(success)
                if contacted:
                    cushion_histogram[cushion_count] += 1

            episode_records.append(
                {
                    "episode_id": episode_id,
                    "initial_observation": initial_obs,
                    "contact_count": episode_contacts,
                    "sparse_success_count": episode_successes,
                }
            )
    finally:
        env.close()

    return {
        "start_distribution": start_distribution,
        "episodes": episodes,
        "shots": total_shots,
        "contact_rate": total_contacts / total_shots if total_shots else 0.0,
        "exact_three_success_rate": total_successes / total_shots if total_shots else 0.0,
        "cushion_histogram_given_contact": {
            str(key): cushion_histogram[key] for key in sorted(cushion_histogram)
        },
        "episode_records": episode_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=["canonical", "local", "broad"],
        default=["canonical", "local", "broad"],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = {
        "schema_version": 1,
        "policy": "uniform_random_normalized_action",
        "seed": args.seed,
        "results": [
            run_distribution(distribution, args.episodes, args.seed)
            for distribution in args.distributions
        ],
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(args.output)
    else:
        print(rendered)


if __name__ == "__main__":
    main()
