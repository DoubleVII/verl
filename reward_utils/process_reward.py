"""Reusable key-point process reward computation for translation rollouts."""

from dataclasses import dataclass
import unicodedata
from typing import Any, Dict, Iterable, List, Optional


def normalize_match_text(value: Any, *, case_sensitive: bool = False) -> str:
    """Normalize Unicode whitespace and case for robust candidate matching."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = " ".join(text.split())
    return text if case_sensitive else text.casefold()


def _iter_key_points(key_points: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(key_points, str):
        import json
        try:
            key_points = json.loads(key_points)
        except (TypeError, ValueError):
            return []
    if not isinstance(key_points, list):
        return []
    return (item for item in key_points if isinstance(item, dict))


@dataclass
class ProcessRewardResult:
    reward: float
    key_points_hit: int
    key_points_total: int
    candidates_hit: int
    candidates_total: int

    @property
    def key_point_hit_ratio(self) -> float:
        return self.key_points_hit / self.key_points_total if self.key_points_total else 0.0


class KeyPointProcessReward:
    """Score candidate translations mentioned in a rollout's reasoning text."""

    def __init__(
        self,
        base_reward: float = 0.05,
        candidate_decay: float = 0.5,
        allow_multiple_candidates: bool = True,
        max_reward: Optional[float] = None,
        case_sensitive: bool = False,
    ) -> None:
        if base_reward < 0:
            raise ValueError("base_reward must be non-negative")
        if not 0 <= candidate_decay <= 1:
            raise ValueError("candidate_decay must be between 0 and 1")
        if max_reward is not None and max_reward < 0:
            raise ValueError("max_reward must be non-negative or None")
        self.base_reward = float(base_reward)
        self.candidate_decay = float(candidate_decay)
        self.allow_multiple_candidates = bool(allow_multiple_candidates)
        self.max_reward = None if max_reward is None else float(max_reward)
        self.case_sensitive = bool(case_sensitive)

    def score(self, reasoning_text: str, key_points: Any) -> ProcessRewardResult:
        reasoning = normalize_match_text(reasoning_text, case_sensitive=self.case_sensitive)
        reward = 0.0
        hit_points = 0
        hit_candidates = 0
        total_points = 0
        total_candidates = 0
        for point in _iter_key_points(key_points):
            translations = point.get("translations", [])
            if isinstance(translations, str):
                translations = [translations]
            if not isinstance(translations, list):
                continue
            candidates = [candidate for candidate in translations if str(candidate or "").strip()]
            if not candidates:
                continue
            total_points += 1
            total_candidates += len(candidates)
            point_hits = 0
            for candidate in candidates:
                normalized = normalize_match_text(candidate, case_sensitive=self.case_sensitive)
                if normalized and normalized in reasoning:
                    point_hits += 1
            if point_hits:
                hit_points += 1
                hit_candidates += point_hits
                count = point_hits if self.allow_multiple_candidates else 1
                reward += sum(self.base_reward * self.candidate_decay ** i for i in range(count))
        if self.max_reward is not None:
            reward = min(reward, self.max_reward)
        return ProcessRewardResult(reward, hit_points, total_points, hit_candidates, total_candidates)
