"""Evaluate a Sampled EfficientZero checkpoint with the original binary objective."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--arm", default="A_sparse_fixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", help="Exact training run ID, including attempt label")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--test-distribution",
        choices=["canonical", "local", "broad"],
        default="canonical",
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    try:
        from lzero.entry import eval_muzero
        from zoo.pooltool.sum_to_three.config.research_alpha import build_research_config
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing training dependency {exc.name!r}. Activate the LightZero Python 3.10 environment and run pip install -e ."
        ) from exc

    runtime_dir = args.output.parent / "eval_runtime" / args.output.stem
    main_config, create_config = build_research_config(
        arm=args.arm,
        seed=args.seed,
        max_env_step=1,
        exp_name=str(runtime_dir),
        cuda=not args.cpu,
    )
    main_config.env.update(
        evaluator_env_num=1,
        n_evaluator_episode=1,
        reward_algorithm="binary",
        start_distribution=args.test_distribution,
        curriculum_enabled=False,
        external_evaluation=True,
        stop_value=11,
    )

    returns_mean, returns = eval_muzero(
        [main_config, create_config],
        seed=args.seed,
        num_episodes_each_seed=args.episodes,
        print_seed_details=False,
        model_path=str(args.checkpoint.resolve()),
    )
    returns = np.asarray(returns, dtype=float)
    checkpoint_sha256 = sha256(args.checkpoint)
    attempt_label = (
        args.checkpoint.parent.parent.name
        if args.checkpoint.parent.name == "ckpt"
        else f"checkpoint-{checkpoint_sha256[:8]}"
    )
    run_id = args.run_id or f"{args.arm}-seed-{args.seed}-{attempt_label}"
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "arm": args.arm,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation": {
            "reward_algorithm": "binary",
            "test_distribution": args.test_distribution,
            "episodes": args.episodes,
            "episode_returns": returns.tolist(),
            "binary_episode_score_mean": float(returns_mean),
            "binary_episode_score_std": float(returns.std()),
            "binary_per_shot_success_rate": float(returns.mean() / 10.0),
        },
        "limitations": [
            "This evaluator exports episode returns; shot-level action diagnostics require a dedicated policy rollout collector.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
