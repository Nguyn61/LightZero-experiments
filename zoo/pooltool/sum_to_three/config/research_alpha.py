"""Student-scale SumToThree reward/curriculum experiment matrix."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple


REPO_ROOT = Path(__file__).resolve().parents[4]
PROTOCOL_PATH = (
    REPO_ROOT
    / "research"
    / "sum_to_three_curriculum_reward"
    / "configs"
    / "experiment_matrix.json"
)


def load_protocol() -> Dict[str, Any]:
    with PROTOCOL_PATH.open(encoding="utf-8") as file:
        protocol = json.load(file)
    fractions = protocol["curriculum"]
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError(f"Curriculum fractions must sum to 1: {fractions}")
    return protocol


PROTOCOL = load_protocol()
ARM_SETTINGS: Dict[str, Dict[str, object]] = {
    arm: {
        "reward_algorithm": settings["training_reward"],
        "start_distribution": settings["training_start_mode"],
        "curriculum_enabled": settings["training_start_mode"] == "curriculum",
    }
    for arm, settings in PROTOCOL["arms"].items()
}


def build_research_config(
    arm: str,
    seed: int,
    max_env_step: int,
    exp_name: str | None = None,
    cuda: bool = True,
    attempt: str = "01",
) -> Tuple[Any, Any]:
    """Build an explicit Sampled EfficientZero config for one treatment arm."""
    from easydict import EasyDict

    if arm not in ARM_SETTINGS:
        raise ValueError(f"Unknown arm {arm!r}. Available arms: {sorted(ARM_SETTINGS)}")

    collector_env_num = 8
    evaluator_env_num = 3
    treatment = deepcopy(ARM_SETTINGS[arm])
    curriculum = PROTOCOL["curriculum"]
    frozen_policy = PROTOCOL["frozen_policy"]
    if exp_name is None:
        exp_name = str(
            Path("experiments")
            / "sum_to_three_curriculum_reward"
            / arm
            / f"seed-{seed}"
            / f"attempt-{attempt}"
        )

    main_config = EasyDict(
        dict(
            exp_name=exp_name,
            env=dict(
                env_name="PoolTool-SumToThree",
                env_type="not_board_games",
                observation_type="coordinate",
                continuous=True,
                manually_discretization=False,
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
                n_evaluator_episode=20,
                stop_value=11,
                manager=dict(shared_memory=False),
                episode_length=10,
                curriculum_total_env_steps=max_env_step,
                curriculum_canonical_fraction=curriculum["canonical_fraction"],
                curriculum_local_fraction=curriculum["local_fraction"],
                local_perturbation_fraction=0.08,
                start_separation_margin=0.005,
                start_sampling_max_attempts=100,
                **treatment,
            ),
            policy=dict(
                model_path=None,
                model=dict(
                    observation_shape=4,
                    action_space_size=2,
                    continuous_action_space=True,
                    num_of_sampled_actions=frozen_policy["num_of_sampled_actions"],
                    sigma_type="conditioned",
                    model_type="mlp",
                    lstm_hidden_size=128,
                    latent_state_dim=128,
                    self_supervised_learning_loss=True,
                    res_connection_in_dynamics=True,
                    norm_type="BN",
                ),
                cuda=cuda,
                env_type="not_board_games",
                game_segment_length=10,
                update_per_collect=None,
                replay_ratio=frozen_policy["replay_ratio"],
                batch_size=256,
                optim_type="Adam",
                lr_piecewise_constant_decay=False,
                ssl_loss_weight=frozen_policy["ssl_loss_weight"],
                discount_factor=1,
                td_steps=10,
                num_unroll_steps=3,
                learning_rate=frozen_policy["learning_rate"],
                grad_clip_value=5,
                policy_entropy_loss_weight=5e-3,
                num_simulations=frozen_policy["num_simulations"],
                reanalyze_ratio=frozen_policy["reanalyze_ratio"],
                n_episode=8,
                eval_freq=5000,
                replay_buffer_size=int(1e5),
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
            ),
        )
    )

    create_config = EasyDict(
        dict(
            env=dict(
                type="pooltool_sumtothree",
                import_names=["zoo.pooltool.sum_to_three.envs.sum_to_three_env"],
            ),
            env_manager=dict(type="subprocess"),
            policy=dict(
                type="sampled_efficientzero",
                import_names=["lzero.policy.sampled_efficientzero"],
            ),
        )
    )
    return main_config, create_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARM_SETTINGS), default="A_sparse_fixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-env-step", type=int, default=PROTOCOL["primary_budget_env_steps"]
    )
    parser.add_argument("--exp-name")
    parser.add_argument("--attempt", default="01", help="Unique attempt label used in the output path")
    parser.add_argument("--cpu", action="store_true", help="Disable CUDA even when available")
    args = parser.parse_args()

    try:
        from lzero.entry import train_muzero
        main_config, create_config = build_research_config(
            arm=args.arm,
            seed=args.seed,
            max_env_step=args.max_env_step,
            exp_name=args.exp_name,
            cuda=not args.cpu,
            attempt=args.attempt,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing training dependency {exc.name!r}. Activate the LightZero Python 3.10 environment and run pip install -e ."
        ) from exc

    output_path = Path(main_config.exp_name)
    if output_path.exists() and (not output_path.is_dir() or any(output_path.iterdir())):
        raise SystemExit(
            f"Refusing to overwrite existing run directory: {output_path}. Use --attempt or --exp-name."
        )

    train_muzero(
        [main_config, create_config],
        seed=args.seed,
        model_path=main_config.policy.model_path,
        max_env_step=args.max_env_step,
    )


if __name__ == "__main__":
    main()
