"""Ported from GRRM eval/span_proxy.py (default heuristic kept in sync by parity tests).

Source-grounded span overlap rewards, without an LLM or model dependencies.

Only quote locations matter; scores and explanations are deliberately ignored.
Repeated quotes use their best compatible source occurrence (offsets are absent
from the artifact format). Duplicate predictions still consume matching slots.
"""

import math
import re
import unicodedata
from dataclasses import dataclass


def metrics(tp, fp, fn, *, empty_item=False):
    precision = tp / (tp + fp) if tp + fp else float(empty_item)
    recall = tp / (tp + fn) if tp + fn else float(empty_item)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float(empty_item)
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


# CJK ideographs are units; other scripts use words. No external tokenizer.
_CHAR_SCRIPTS = r"\u3005\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\U00020000-\U000323af"
_UNITS = re.compile(rf"[{_CHAR_SCRIPTS}]|[^\W_{_CHAR_SCRIPTS}]+", re.UNICODE)


def _canonical(text):
    return unicodedata.normalize("NFKC", text).casefold()


def _compact(text):
    return "".join(c for c in _canonical(text) if c.isalnum())


@dataclass(frozen=True)
class GroundedSpan:
    occurrences: tuple
    char_length: float
    token_length: float


class SpanSource:
    """Cache this object when scoring multiple rollouts of one source."""

    def __init__(self, source):
        canonical = _canonical(source)
        self.text = _compact(source)
        self.weights = []
        offset = 0
        for match in _UNITS.finditer(canonical):
            token = _compact(match.group())
            if not token:
                continue
            offset += len(token)
            self.weights.extend([1 / len(token)] * len(token))
        # Python's unicode alnum includes characters outside the regex word set.
        if offset != len(self.text):
            raise ValueError("Unsupported source Unicode tokenization")
        self.prefix = [0.0]
        for weight in self.weights:
            self.prefix.append(self.prefix[-1] + weight)
        self._cache = {}

    def length(self, start, end, unit):
        return end - start if unit == "char" else self.prefix[end] - self.prefix[start]

    def ground(self, quote):
        if quote in self._cache:
            return self._cache[quote]
        compact = _compact(quote)
        occurrences = []
        start = self.text.find(compact) if compact else -1
        while start >= 0:
            end = start + len(compact)
            # Subwords are legitimate references (e.g. German morphology).
            # Partial words receive fractional token mass from source offsets.
            occurrences.append((start, end))
            start = self.text.find(compact, start + 1)
        token_length = sum(1 for _ in _UNITS.finditer(_canonical(quote)))
        grounded = GroundedSpan(tuple(occurrences), len(compact), token_length)
        self._cache[quote] = grounded
        return grounded

    def pair(self, left, right, unit="token"):
        """Return (intersection, predicted length, reference length), best occurrence."""
        if unit not in ("char", "token"):
            raise ValueError("unit must be char or token")
        best = (0.0, getattr(left, f"{unit}_length"), getattr(right, f"{unit}_length"))
        for a, b in left.occurrences:
            for c, d in right.occurrences:
                intersection = self.length(max(a, c), min(b, d), unit) if max(a, c) < min(b, d) else 0.0
                candidate = intersection, self.length(a, b, unit), self.length(c, d, unit)
                if candidate[0] > best[0]:
                    best = candidate
        return best


def pair_score(
    intersection,
    predicted,
    reference,
    *,
    method="containment",
    gamma=0.5,
    threshold=0.5,
    under_gamma=0.1,
    overlap_power=1.0,
):
    """All methods penalize excess length; containment uses a power length penalty."""
    if not intersection or not predicted or not reference:
        return 0.0
    dice = 2 * intersection / (predicted + reference)
    if method == "exact":
        return float(math.isclose(intersection, predicted) and math.isclose(intersection, reference))
    if method == "dice":
        return dice
    if method == "iou":
        return intersection / (predicted + reference - intersection)
    if method in ("containment", "asymmetric"):
        exponent = under_gamma if method == "asymmetric" and predicted < reference else gamma
        return min(
            1.0,
            (intersection / min(predicted, reference)) ** overlap_power
            * (min(predicted, reference) / max(predicted, reference)) ** exponent,
        )
    if method == "threshold":
        return float(dice >= threshold)
    raise ValueError(f"Unknown overlap method: {method}")


def maximum_weight_matching(weights):
    """Rectangular Hungarian assignment, O(min(n,m)^2 max(n,m)), pure Python."""
    if not weights or not weights[0]:
        return []
    transpose = len(weights) > len(weights[0])
    matrix = list(map(list, zip(*weights, strict=True))) if transpose else weights
    n, m = len(matrix), len(matrix[0])
    u, v, p, way = [0.0] * (n + 1), [0.0] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        distances, used = [math.inf] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], math.inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    current = -matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if current < distances[j]:
                        distances[j], way[j] = current, j0
                    if distances[j] < delta:
                        delta, j1 = distances[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    distances[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    output = []
    for j in range(1, m + 1):
        if p[j] and matrix[p[j] - 1][j - 1] > 0:
            left, right = (j - 1, p[j] - 1) if transpose else (p[j] - 1, j - 1)
            output.append((left, right, weights[left][right]))
    return sorted(output)


def score_spans(
    source,
    predicted_spans,
    reference_spans,
    *,
    unit="char",
    method="asymmetric",
    gamma=0.5,
    threshold=0.5,
    under_gamma=0.25,
    overlap_power=1.0,
):
    """Return soft TP/P/R/F1 in [0,1], plus grounding and assignment diagnostics.

    Accept source text or a cached SpanSource, and lists of quote strings or
    dictionaries with source_span. Both empty scores 1; one empty scores 0.
    Ungrounded quotes remain in denominators and receive zero credit.
    """
    if unit not in ("char", "token") or method not in (
        "exact",
        "dice",
        "iou",
        "containment",
        "asymmetric",
        "threshold",
    ):
        raise ValueError("Invalid unit or method")
    if not math.isfinite(gamma) or gamma <= 0 or not 0 < threshold <= 1:
        raise ValueError("gamma must be positive and threshold must be in (0, 1]")
    if any(not math.isfinite(value) or value <= 0 for value in (under_gamma, overlap_power)):
        raise ValueError("under_gamma and overlap_power must be positive")
    context = source if isinstance(source, SpanSource) else SpanSource(source)

    def ground(spans):
        return [
            context.ground(
                s
                if isinstance(s, str)
                else s.get("source_span", "")
                if isinstance(s, dict) and isinstance(s.get("source_span", ""), str)
                else ""
            )
            for s in spans
        ]

    left, right = ground(predicted_spans), ground(reference_spans)
    weights = [
        [
            pair_score(
                *context.pair(p, g, unit),
                method=method,
                gamma=gamma,
                threshold=threshold,
                under_gamma=under_gamma,
                overlap_power=overlap_power,
            )
            for g in right
        ]
        for p in left
    ]
    matching = maximum_weight_matching(weights)
    tp = sum(w for _, _, w in matching)
    return dict(
        metrics(tp, len(left) - tp, len(right) - tp, empty_item=not left and not right),
        matching=[list(pair) for pair in matching],
        ungrounded_prediction=sum(not p.occurrences for p in left),
        ungrounded_reference=sum(not g.occurrences for g in right),
    )
