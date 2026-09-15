# Adaptive Model Hierarchy

AI Council can adjust model ordering from recent execution outcomes instead of
relying only on static capability and price metadata. The adaptive ranker keeps
anonymous model-level metrics in a configurable rolling window and blends the
result with the existing task-specific optimizer score.

## Privacy boundary

`PerformanceObservation` accepts only:

- model identifier;
- quality score;
- latency;
- cost;
- success status; and
- observation timestamp.

Prompts, responses, user identifiers, request identifiers, and arbitrary
metadata are neither accepted nor retained. `export_aggregates()` returns only
model-level averages and sample counts.

## Configure the ranking window

```python
from ai_council.orchestration import (
    AdaptiveHierarchyRanker,
    AdaptiveRankingConfig,
)

ranker = AdaptiveHierarchyRanker(
    AdaptiveRankingConfig(
        window_seconds=7 * 24 * 60 * 60,
        rerank_interval_seconds=15 * 60,
        minimum_samples=10,
    )
)
```

Models without enough observations receive a neutral score. This prevents a
new model from being penalized before there is sufficient evidence. Rankings
are deterministic: tied models are ordered by model identifier.

## Pre-train before deployment

Companies can seed the hierarchy with representative benchmark outcomes:

```python
from ai_council.orchestration import PerformanceObservation

ranker.pretrain(
    [
        PerformanceObservation(
            model_id="candidate-a",
            quality_score=0.91,
            latency_seconds=1.8,
            cost=0.03,
            success=True,
        ),
        PerformanceObservation(
            model_id="native-baseline",
            quality_score=0.82,
            latency_seconds=2.4,
            cost=0.05,
            success=True,
        ),
    ]
)

hierarchy = ranker.rank(["candidate-a", "native-baseline"], force=True)
```

Production executions feed the same metrics through
`CostOptimizer.update_performance_history()`. A new observation invalidates old
selection-cache entries so subsequent choices can use the latest evidence.

## Run the native comparison smoke test

Create a JSON Lines file containing anonymous observations. Each line follows
this schema:

```json
{"model_id":"candidate-a","quality_score":0.91,"latency_seconds":1.8,"cost":0.03,"success":true,"recorded_at":1789430400}
```

Then run:

```bash
python scripts/compare_model_performance.py benchmark.jsonl \
  --orchestrated-model candidate-a \
  --native-model native-baseline \
  --window-hours 168 \
  --minimum-samples 10
```

The JSON report contains:

- quality and success-rate deltas;
- latency and cost deltas;
- in-window sample counts; and
- the resulting adaptive hierarchy with scores.

The command fails clearly when the input is malformed or either model has
insufficient in-window observations. When timestamps are supplied, offline
benchmarks use the newest observation as their reference time so historical
benchmark fixtures remain reproducible.
