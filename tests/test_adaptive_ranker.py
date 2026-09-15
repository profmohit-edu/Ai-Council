import json
import math

import pytest

from ai_council.orchestration.adaptive_ranker import (
    AdaptiveHierarchyRanker,
    AdaptiveRankingConfig,
    PerformanceObservation,
)
from ai_council.orchestration.cost_optimizer import CostOptimizer
from ai_council.core.models import ExecutionMode, Subtask, TaskType
from scripts.compare_model_performance import load_observations, main


def observation(model_id, quality, latency, cost, timestamp, success=True):
    return PerformanceObservation(
        model_id=model_id,
        quality_score=quality,
        latency_seconds=latency,
        cost=cost,
        success=success,
        recorded_at=timestamp,
    )


def test_pretraining_ranks_models_from_anonymous_outcomes():
    ranker = AdaptiveHierarchyRanker(
        AdaptiveRankingConfig(minimum_samples=2, rerank_interval_seconds=0)
    )
    samples = [
        observation("reliable", 0.92, 1.0, 0.02, 100),
        observation("reliable", 0.90, 1.2, 0.02, 101),
        observation("unreliable", 0.70, 3.0, 0.08, 100, success=False),
        observation("unreliable", 0.75, 2.5, 0.07, 101),
    ]

    assert ranker.pretrain(samples) == 4

    ranking = ranker.rank(["unreliable", "reliable"], now=102)
    assert [item.model_id for item in ranking] == ["reliable", "unreliable"]
    assert all(item.sample_count == 2 for item in ranking)
    assert set(ranker.export_aggregates(now=102)["reliable"]) == {
        "sample_count",
        "quality",
        "success_rate",
        "latency",
        "cost",
    }


def test_window_expiry_removes_stale_observations():
    ranker = AdaptiveHierarchyRanker(
        AdaptiveRankingConfig(window_seconds=10, minimum_samples=1)
    )
    ranker.record(observation("model-a", 0.9, 1.0, 0.01, 100))
    ranker.record(observation("model-a", 0.2, 8.0, 0.20, 111))

    aggregates = ranker.export_aggregates(now=111)

    assert aggregates["model-a"]["sample_count"] == 1
    assert aggregates["model-a"]["quality"] == pytest.approx(0.2)


def test_window_expiry_invalidates_cached_ranking():
    ranker = AdaptiveHierarchyRanker(
        AdaptiveRankingConfig(
            window_seconds=10, minimum_samples=1, rerank_interval_seconds=100
        )
    )
    ranker.record(observation("a", 1.0, 0.1, 0.0, 100))
    ranker.record(observation("b", 0.9, 0.1, 0.0, 105))
    assert ranker.rank(["a", "b"], now=105)[0].model_id == "a"

    ranking = ranker.rank(["a", "b"], now=111)

    assert ranking[0].model_id == "b"
    assert ranking[0].sample_count == 1


def test_ranking_refreshes_only_after_configured_interval():
    ranker = AdaptiveHierarchyRanker(
        AdaptiveRankingConfig(
            window_seconds=100,
            minimum_samples=1,
            rerank_interval_seconds=10,
        )
    )
    ranker.record(observation("a", 0.6, 3.0, 0.10, 100))
    ranker.record(observation("b", 0.1, 5.0, 0.20, 100, success=False))
    assert ranker.rank(["a", "b"], now=100)[0].model_id == "a"

    for timestamp in (103, 104, 105):
        ranker.record(observation("b", 1.0, 0.1, 0.0, timestamp))
    cached = ranker.rank(["a", "b"], now=105)
    refreshed = ranker.rank(["a", "b"], now=111)

    assert cached[0].model_id == "a"
    assert refreshed[0].score > refreshed[1].score
    assert refreshed[0].model_id == "b"


def test_smoke_comparison_reports_orchestrated_vs_native_deltas():
    ranker = AdaptiveHierarchyRanker(
        AdaptiveRankingConfig(minimum_samples=1, window_seconds=1000)
    )
    ranker.pretrain(
        [
            observation("orchestrated", 0.9, 2.0, 0.03, 100),
            observation("native", 0.7, 1.0, 0.02, 100, success=False),
        ]
    )

    comparison = ranker.compare("orchestrated", "native", now=101)

    assert comparison.quality_delta == pytest.approx(0.2)
    assert comparison.success_rate_delta == pytest.approx(1.0)
    assert comparison.latency_delta_seconds == pytest.approx(1.0)
    assert comparison.cost_delta == pytest.approx(0.01)


@pytest.mark.parametrize(
    "bad_observation",
    [
        observation("", 0.5, 1.0, 0.01, 100),
        observation("model", 1.1, 1.0, 0.01, 100),
        observation("model", 0.5, -1.0, 0.01, 100),
        observation("model", 0.5, 1.0, -0.01, 100),
    ],
)
def test_invalid_telemetry_is_rejected(bad_observation):
    with pytest.raises(ValueError):
        AdaptiveHierarchyRanker().record(bad_observation)


def test_cost_optimizer_feeds_execution_outcomes_into_adaptive_hierarchy(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    ranker = AdaptiveHierarchyRanker(
        AdaptiveRankingConfig(minimum_samples=1, rerank_interval_seconds=0)
    )
    optimizer = CostOptimizer(model_registry=object(), adaptive_ranker=ranker)

    optimizer.update_performance_history(
        "model-a",
        actual_cost=0.02,
        quality_score=0.9,
        actual_latency=1.5,
    )

    aggregates = ranker.export_aggregates()
    assert aggregates["model-a"]["quality"] == pytest.approx(0.9)
    assert aggregates["model-a"]["latency"] == pytest.approx(1.5)
    assert optimizer.get_adaptive_hierarchy(["model-b", "model-a"], force=True) == [
        "model-a",
        "model-b",
    ]
    optimizer._optimization_cache.close()


def test_invalid_telemetry_does_not_mutate_legacy_history(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    optimizer = CostOptimizer(model_registry=object())

    with pytest.raises(ValueError):
        optimizer.update_performance_history(
            "model-a", actual_cost=0.1, quality_score=math.nan
        )

    assert "model-a" not in optimizer._performance_history
    optimizer._optimization_cache.close()


def test_expired_adaptive_observation_invalidates_selection_cache(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    ranker = AdaptiveHierarchyRanker(
        AdaptiveRankingConfig(window_seconds=1, minimum_samples=1)
    )
    ranker.record(observation("model-a", 1.0, 0.1, 0.0, 1))
    optimizer = CostOptimizer(model_registry=object(), adaptive_ranker=ranker)
    subtask = Subtask(content="test", task_type=TaskType.REASONING)
    cache_key = optimizer._create_cache_key(
        subtask, ExecutionMode.BALANCED, ["model-a"]
    )
    optimizer._optimization_cache[cache_key] = "stale"
    monkeypatch.setattr(
        optimizer,
        "_score_model_for_optimization",
        lambda *_: {
            "composite_score": 0.8,
            "cost": 0.1,
            "time": 1.0,
            "quality": 0.8,
            "reliability": 0.9,
            "confidence": 0.8,
        },
    )

    result = optimizer.optimize_model_selection(
        subtask, ExecutionMode.BALANCED, ["model-a"]
    )

    assert result.recommended_model == "model-a"
    assert result != "stale"
    optimizer._optimization_cache.close()


def test_smoke_cli_compares_jsonl_benchmarks(tmp_path, capsys):
    benchmark = tmp_path / "benchmark.jsonl"
    rows = [
        {
            "model_id": "orchestrated",
            "quality_score": 0.9,
            "latency_seconds": 1.5,
            "cost": 0.02,
            "recorded_at": 100,
        },
        {
            "model_id": "native",
            "quality_score": 0.7,
            "latency_seconds": 2.5,
            "cost": 0.04,
            "recorded_at": 100,
        },
    ]
    benchmark.write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )

    assert main(
        [
            str(benchmark),
            "--orchestrated-model",
            "orchestrated",
            "--native-model",
            "native",
            "--minimum-samples",
            "1",
        ]
    ) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["comparison"]["quality_delta"] == pytest.approx(0.2)
    assert report["hierarchy"][0]["model_id"] == "orchestrated"


def test_smoke_cli_reports_invalid_jsonl_line(tmp_path):
    benchmark = tmp_path / "bad.jsonl"
    benchmark.write_text('{"model_id": "missing metrics"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        load_observations(benchmark)


def test_smoke_cli_rejects_string_boolean_values(tmp_path):
    benchmark = tmp_path / "bad-success.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "model_id": "candidate-a",
                "quality_score": 0.9,
                "latency_seconds": 1.5,
                "cost": 0.02,
                "success": "false",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="success must be a JSON boolean"):
        load_observations(benchmark)


def test_smoke_cli_rejects_non_object_json(tmp_path):
    benchmark = tmp_path / "bad-shape.jsonl"
    benchmark.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1.*JSON object"):
        load_observations(benchmark)


@pytest.mark.parametrize("window", ["nan", "inf", "-1", "0"])
def test_smoke_cli_rejects_invalid_window(tmp_path, window):
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "model_id": "a",
                "quality_score": 0.9,
                "latency_seconds": 1.0,
                "cost": 0.1,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="window-hours"):
        main(
            [
                str(benchmark),
                "--orchestrated-model",
                "a",
                "--native-model",
                "b",
                "--window-hours",
                window,
            ]
        )


def test_smoke_cli_normalizes_missing_timestamp_to_reference_time(tmp_path, capsys):
    benchmark = tmp_path / "mixed-timestamps.jsonl"
    rows = [
        {
            "model_id": "orchestrated",
            "quality_score": 0.9,
            "latency_seconds": 1.0,
            "cost": 0.01,
            "recorded_at": 100,
        },
        {
            "model_id": "native",
            "quality_score": 0.8,
            "latency_seconds": 2.0,
            "cost": 0.02,
        },
    ]
    benchmark.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")

    assert main(
        [
            str(benchmark),
            "--orchestrated-model",
            "orchestrated",
            "--native-model",
            "native",
            "--minimum-samples",
            "1",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["comparison"]["native_samples"] == 1


@pytest.mark.parametrize(
    "config",
    [
        {"window_seconds": math.nan},
        {"window_seconds": math.inf},
        {"rerank_interval_seconds": math.nan},
        {"quality_weight": math.nan},
        {"quality_weight": -0.1, "success_weight": 0.4},
    ],
)
def test_non_finite_or_negative_ranking_config_is_rejected(config):
    with pytest.raises(ValueError):
        AdaptiveRankingConfig(**config)


@pytest.mark.parametrize(
    "bad_observation",
    [
        observation("model", 0.5, math.nan, 0.01, 100),
        observation("model", 0.5, math.inf, 0.01, 100),
        observation("model", 0.5, 1.0, math.nan, 100),
        observation("model", 0.5, 1.0, 0.01, math.nan),
    ],
)
def test_non_finite_telemetry_is_rejected(bad_observation):
    with pytest.raises(ValueError):
        AdaptiveHierarchyRanker().record(bad_observation)
