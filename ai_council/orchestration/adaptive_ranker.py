"""Privacy-preserving adaptive model ranking from recent performance telemetry."""

from dataclasses import dataclass
import math
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class AdaptiveRankingConfig:
    """Configuration for the rolling performance window and ranking cadence."""

    window_seconds: float = 24 * 60 * 60
    rerank_interval_seconds: float = 5 * 60
    minimum_samples: int = 3
    quality_weight: float = 0.45
    success_weight: float = 0.30
    latency_weight: float = 0.15
    cost_weight: float = 0.10

    def __post_init__(self) -> None:
        if not math.isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if (
            not math.isfinite(self.rerank_interval_seconds)
            or self.rerank_interval_seconds < 0
        ):
            raise ValueError("rerank_interval_seconds cannot be negative")
        if self.minimum_samples <= 0:
            raise ValueError("minimum_samples must be positive")
        weights = (
            self.quality_weight,
            self.success_weight,
            self.latency_weight,
            self.cost_weight,
        )
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("ranking weights must be finite and non-negative")
        total_weight = sum(weights)
        if abs(total_weight - 1.0) > 1e-9:
            raise ValueError("ranking weights must sum to 1.0")


@dataclass(frozen=True)
class PerformanceObservation:
    """Anonymous outcome metrics for one model execution.

    Request content, user identifiers, and response text are deliberately absent.
    """

    model_id: str
    quality_score: float
    latency_seconds: float
    cost: float
    success: bool = True
    recorded_at: Optional[float] = None


@dataclass(frozen=True)
class RankedModel:
    """A model and its adaptive score, ordered best-first by ``rank``."""

    model_id: str
    score: float
    sample_count: int


@dataclass(frozen=True)
class SmokeComparison:
    """Aggregated comparison between orchestration and a native model baseline."""

    orchestrated_model: str
    native_model: str
    quality_delta: float
    success_rate_delta: float
    latency_delta_seconds: float
    cost_delta: float
    orchestrated_samples: int
    native_samples: int


class AdaptiveHierarchyRanker:
    """Re-rank a model hierarchy using a bounded window of anonymous outcomes."""

    def __init__(self, config: Optional[AdaptiveRankingConfig] = None) -> None:
        self.config = config or AdaptiveRankingConfig()
        self._observations: Dict[str, List[PerformanceObservation]] = {}
        self._last_ranked_at: Optional[float] = None
        self._last_model_set: Tuple[str, ...] = ()
        self._cached_ranking: List[RankedModel] = []

    def record(self, observation: PerformanceObservation) -> None:
        """Record a validated performance outcome without request-level data."""
        if not observation.model_id:
            raise ValueError("model_id is required")
        if (
            not math.isfinite(observation.quality_score)
            or not 0.0 <= observation.quality_score <= 1.0
        ):
            raise ValueError("quality_score must be between 0.0 and 1.0")
        if (
            not math.isfinite(observation.latency_seconds)
            or observation.latency_seconds < 0
        ):
            raise ValueError("latency_seconds cannot be negative")
        if not math.isfinite(observation.cost) or observation.cost < 0:
            raise ValueError("cost cannot be negative")

        recorded_at = observation.recorded_at
        if recorded_at is None:
            recorded_at = time.time()
            observation = PerformanceObservation(
                model_id=observation.model_id,
                quality_score=observation.quality_score,
                latency_seconds=observation.latency_seconds,
                cost=observation.cost,
                success=observation.success,
                recorded_at=recorded_at,
            )
        elif not math.isfinite(recorded_at):
            raise ValueError("recorded_at must be finite")

        self._observations.setdefault(observation.model_id, []).append(observation)
        self._prune(recorded_at)

    def pretrain(self, observations: Iterable[PerformanceObservation]) -> int:
        """Seed the hierarchy with representative benchmark observations."""
        count = 0
        for observation in observations:
            self.record(observation)
            count += 1
        return count

    def rank(
        self,
        model_ids: Sequence[str],
        *,
        force: bool = False,
        now: Optional[float] = None,
    ) -> List[RankedModel]:
        """Return a periodically refreshed, deterministic model hierarchy."""
        current_time = time.time() if now is None else now
        model_set = tuple(sorted(set(model_ids)))
        self._prune(current_time)

        ranking_is_fresh = (
            self._last_ranked_at is not None
            and current_time - self._last_ranked_at
            < self.config.rerank_interval_seconds
        )
        if (
            not force
            and ranking_is_fresh
            and model_set == self._last_model_set
        ):
            return list(self._cached_ranking)

        ranking = [self._ranked_model(model_id) for model_id in model_set]
        ranking.sort(key=lambda item: (-item.score, item.model_id))
        self._cached_ranking = ranking
        self._last_model_set = model_set
        self._last_ranked_at = current_time
        return list(ranking)

    def score(self, model_id: str, *, now: Optional[float] = None) -> Optional[float]:
        """Return a model's adaptive score once enough observations exist."""
        current_time = time.time() if now is None else now
        self._prune(current_time)
        ranked = self._ranked_model(model_id)
        if ranked.sample_count < self.config.minimum_samples:
            return None
        return ranked.score

    def export_aggregates(self, *, now: Optional[float] = None) -> Dict[str, Dict[str, float]]:
        """Export model-level aggregates only; raw request data is never stored."""
        current_time = time.time() if now is None else now
        self._prune(current_time)
        return {
            model_id: self._aggregate(observations)
            for model_id, observations in sorted(self._observations.items())
            if observations
        }

    def compare(
        self,
        orchestrated_model: str,
        native_model: str,
        *,
        now: Optional[float] = None,
    ) -> SmokeComparison:
        """Compare recent orchestrated outcomes with a native model baseline."""
        aggregates = self.export_aggregates(now=now)
        if orchestrated_model not in aggregates or native_model not in aggregates:
            raise ValueError("both models need observations before comparison")
        orchestrated = aggregates[orchestrated_model]
        native = aggregates[native_model]
        return SmokeComparison(
            orchestrated_model=orchestrated_model,
            native_model=native_model,
            quality_delta=orchestrated["quality"] - native["quality"],
            success_rate_delta=orchestrated["success_rate"] - native["success_rate"],
            latency_delta_seconds=orchestrated["latency"] - native["latency"],
            cost_delta=orchestrated["cost"] - native["cost"],
            orchestrated_samples=int(orchestrated["sample_count"]),
            native_samples=int(native["sample_count"]),
        )

    def _ranked_model(self, model_id: str) -> RankedModel:
        observations = self._observations.get(model_id, [])
        if len(observations) < self.config.minimum_samples:
            return RankedModel(model_id=model_id, score=0.5, sample_count=len(observations))

        aggregate = self._aggregate(observations)
        latency_score = 1.0 / (1.0 + aggregate["latency"])
        cost_score = 1.0 / (1.0 + aggregate["cost"])
        score = (
            aggregate["quality"] * self.config.quality_weight
            + aggregate["success_rate"] * self.config.success_weight
            + latency_score * self.config.latency_weight
            + cost_score * self.config.cost_weight
        )
        return RankedModel(model_id=model_id, score=score, sample_count=len(observations))

    @staticmethod
    def _aggregate(observations: Sequence[PerformanceObservation]) -> Dict[str, float]:
        sample_count = len(observations)
        return {
            "sample_count": float(sample_count),
            "quality": sum(item.quality_score for item in observations) / sample_count,
            "success_rate": sum(1.0 for item in observations if item.success) / sample_count,
            "latency": sum(item.latency_seconds for item in observations) / sample_count,
            "cost": sum(item.cost for item in observations) / sample_count,
        }

    def prune_expired(self, *, now: Optional[float] = None) -> bool:
        """Drop expired observations and report whether cached inputs changed."""
        current_time = time.time() if now is None else now
        if not math.isfinite(current_time):
            raise ValueError("now must be finite")
        return self._prune(current_time)

    def _prune(self, now: float) -> bool:
        cutoff = now - self.config.window_seconds
        changed = False
        for model_id in list(self._observations):
            original = self._observations[model_id]
            retained = [
                observation
                for observation in original
                if observation.recorded_at is not None
                and observation.recorded_at >= cutoff
            ]
            changed = changed or len(retained) != len(original)
            if retained:
                self._observations[model_id] = retained
            else:
                del self._observations[model_id]
        if changed:
            self._cached_ranking = []
            self._last_model_set = ()
            self._last_ranked_at = None
        return changed
