#!/usr/bin/env python3
"""Compare orchestrated model outcomes with a native benchmark JSONL file."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import List, Optional

# Keep the documented source-checkout invocation working without requiring an
# editable install first. Installed-package imports are unaffected.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ai_council.orchestration.adaptive_ranker import (
    AdaptiveHierarchyRanker,
    AdaptiveRankingConfig,
    PerformanceObservation,
)


def load_observations(path: Path) -> List[PerformanceObservation]:
    """Load anonymous outcome metrics from a JSON Lines benchmark file."""
    observations = []
    with path.open("r", encoding="utf-8") as benchmark_file:
        for line_number, line in enumerate(benchmark_file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                success = item.get("success", True)
                if not isinstance(success, bool):
                    raise TypeError("success must be a JSON boolean")
                observations.append(
                    PerformanceObservation(
                        model_id=item["model_id"],
                        quality_score=float(item["quality_score"]),
                        latency_seconds=float(item["latency_seconds"]),
                        cost=float(item["cost"]),
                        success=success,
                        recorded_at=(
                            float(item["recorded_at"])
                            if item.get("recorded_at") is not None
                            else None
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid benchmark observation on line {line_number}: {error}"
                ) from error
    if not observations:
        raise ValueError("benchmark file contains no observations")
    return observations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare adaptive orchestration with a native model baseline."
    )
    parser.add_argument("benchmark", type=Path, help="JSONL performance observations")
    parser.add_argument("--orchestrated-model", required=True)
    parser.add_argument("--native-model", required=True)
    parser.add_argument("--window-hours", type=float, default=24.0)
    parser.add_argument("--minimum-samples", type=int, default=3)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    observations = load_observations(args.benchmark)
    timestamps = [
        observation.recorded_at
        for observation in observations
        if observation.recorded_at is not None
    ]
    reference_time = max(timestamps) if timestamps else time.time()
    observations.sort(key=lambda item: item.recorded_at or reference_time)

    ranker = AdaptiveHierarchyRanker(
        AdaptiveRankingConfig(
            window_seconds=args.window_hours * 60 * 60,
            minimum_samples=args.minimum_samples,
            rerank_interval_seconds=0,
        )
    )
    ranker.pretrain(observations)
    aggregates = ranker.export_aggregates(now=reference_time)
    for model_id in (args.orchestrated_model, args.native_model):
        sample_count = aggregates.get(model_id, {}).get("sample_count", 0)
        if sample_count < args.minimum_samples:
            raise ValueError(
                f"{model_id} needs at least {args.minimum_samples} in-window samples"
            )

    comparison = ranker.compare(
        args.orchestrated_model,
        args.native_model,
        now=reference_time,
    )
    hierarchy = ranker.rank(
        [args.orchestrated_model, args.native_model],
        force=True,
        now=reference_time,
    )
    print(
        json.dumps(
            {
                "comparison": asdict(comparison),
                "hierarchy": [asdict(model) for model in hierarchy],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
