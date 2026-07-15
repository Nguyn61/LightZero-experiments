"""Export TensorBoard scalar series to normalized JSON for the student notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_scalars(log_dir: Path) -> Dict[str, List[dict]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise SystemExit(
            "TensorBoard is required. Install it with: pip install tensorboard"
        ) from exc

    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = accumulator.Tags().get("scalars", [])
    return {
        tag: [
            {"wall_time": event.wall_time, "step": event.step, "value": event.value}
            for event in accumulator.Scalars(tag)
        ]
        for tag in scalar_tags
    }


def choose_primary_tag(series: Dict[str, List[dict]]) -> str | None:
    preferred_suffixes = (
        "evaluator_step/reward_mean",
        "evaluator_envstep/reward_mean",
        "evaluator_iter/reward_mean",
    )
    for suffix in preferred_suffixes:
        for tag in series:
            if tag.endswith(suffix):
                return tag
    for tag in series:
        if "evaluator" in tag.lower() and tag.lower().endswith("reward_mean"):
            return tag
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run-id", default="unknown")
    parser.add_argument("--arm", default="unknown")
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args()

    series = load_scalars(args.log_dir)
    primary_tag = choose_primary_tag(series)
    payload = {
        "schema_version": 1,
        "run_id": args.run_id,
        "arm": args.arm,
        "seed": args.seed,
        "source": {"type": "tensorboard", "log_dir": str(args.log_dir)},
        "primary_binary_series_tag": primary_tag,
        "series": series,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Exported {len(series)} scalar tags to {args.output}")
    if primary_tag is None:
        print("Warning: no evaluator reward_mean tag was identified automatically.")


if __name__ == "__main__":
    main()
