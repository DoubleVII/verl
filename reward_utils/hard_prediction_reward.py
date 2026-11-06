"""Position-only assignment followed by reference difficulty alignment.

No judge/model calls or runtime dependency on the GRRM checkout.
"""

import json
import math
import re

from reward_utils._hard_span_proxy import score_spans


def _limits(extra_info):
    limits = {key: extra_info.get(key, default) for key, default in (("max_span_chars", 48), ("max_span_words", 8))}
    if any(type(value) is not int or value < 1 for value in limits.values()):
        raise ValueError("Span limits must be positive integers")
    return limits


def validate_annotation(value, source, *, max_span_chars=48, max_span_words=8):
    """Mirror GRRM's strict annotation contract; never repair labels or quotes."""
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("analysis"), str)
        or not value["analysis"].strip()
        or type(value.get("score")) is not int
        or not 0 <= value["score"] <= 10
        or not isinstance(value.get("spans"), list)
    ):
        return None
    spans, seen = [], set()
    for span in value["spans"]:
        if not isinstance(span, dict):
            return None
        quote = span.get("source_span")
        if (
            not isinstance(quote, str)
            or not quote.strip()
            or quote != quote.strip()
            or quote not in source
            or quote in seen
            or "\n" in quote
            or "\r" in quote
            or len(quote) > max_span_chars
            or len(quote.split()) > max_span_words
            or type(span.get("score")) is not int
            or not 1 <= span["score"] <= 10
            or not isinstance(span.get("explanation"), str)
            or not span["explanation"].strip()
        ):
            return None
        seen.add(quote)
        spans.append({key: span[key] for key in ("source_span", "score", "explanation")})
    if value["score"] != max((span["score"] for span in spans), default=0):
        return None
    return dict(analysis=value["analysis"], score=value["score"], spans=spans)


def _parse_response(text):
    if not isinstance(text, str):
        return None
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?[ \t]*\n(.*?)\n```", text, re.DOTALL)
        if not match:
            return None
        text = match[1]
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
    *,
    difficulty_weight=0.3,
    difficulty_tau=2.0,
    score_scale_factor=1.0,
):
    """Return reward and numeric diagnostics accepted by verl reward managers.

    Missing/corrupt reference metadata raises; invalid model outputs score -1.
    Error means are zero when their corresponding match count is zero.
    Group sums/counts allow pooled empty/nonempty rates without NaN metrics.
    score_scale_factor scales final scores (including invalid-output penalties)
    and group reward sums; localization and difficulty diagnostics stay unscaled.
    """
    if (
        isinstance(score_scale_factor, bool)
        or not isinstance(score_scale_factor, (int, float))
        or not math.isfinite(score_scale_factor)
        or score_scale_factor < 0
    ):
        raise ValueError("score_scale_factor must be finite and nonnegative")
    if (
        isinstance(difficulty_weight, bool)
        or not isinstance(difficulty_weight, (int, float))
        or not math.isfinite(difficulty_weight)
        or not 0 <= difficulty_weight <= 1
    ):
        raise ValueError("difficulty_weight must be finite and in [0, 1]")
    if (
        isinstance(difficulty_tau, bool)
        or not isinstance(difficulty_tau, (int, float))
        or not math.isfinite(difficulty_tau)
        or difficulty_tau <= 0
    ):
        raise ValueError("difficulty_tau must be finite and positive")
    if (
        not isinstance(extra_info, dict)
        or not isinstance(extra_info.get("src_text"), str)
        or not extra_info["src_text"].strip()
    ):
        raise ValueError("extra_info must contain nonempty src_text")
    limits = _limits(extra_info)
    try:
        reference = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid reference JSON") from error
    source = extra_info["src_text"]
    reference = validate_annotation(reference, source, **limits)
    if reference is None:
        raise ValueError("Invalid reference annotation")
    prediction = validate_annotation(_parse_response(solution_str), source, **limits)
    m = len(reference["spans"])
    result = dict(
        score=-1.0,
        valid_output=int(prediction is not None),
        span_precision=0.0,
        span_recall=0.0,
        span_f1=0.0,
        prediction_count=0,
        reference_count=m,
        match_count=0,
        difficulty_mae=0.0,
        difficulty_bias=0.0,
        difficulty_abs_error_sum=0.0,
        difficulty_error_sum=0.0,
        high_confidence_match_count=0,
        high_confidence_difficulty_mae=0.0,
        high_confidence_difficulty_bias=0.0,
        high_confidence_abs_error_sum=0.0,
        high_confidence_error_sum=0.0,
        empty_reference_count=int(m == 0),
        nonempty_reference_count=int(m > 0),
        valid_empty_reference_count=0,
        valid_nonempty_reference_count=0,
        false_positive_nonempty_count=0,
        false_negative_empty_count=0,
        empty_reference_reward_sum=0.0,
        nonempty_reference_reward_sum=0.0,
    )
    if prediction is not None:
        spans, gold = prediction["spans"], reference["spans"]
        n = len(spans)
        proxy = score_spans(source, spans, gold)
        matching = proxy["matching"]  # Only position weights select these edges.
        contribution = sum(
            w
            * (
                1
                - difficulty_weight
                + difficulty_weight * math.exp(-abs(spans[i]["score"] - gold[j]["score"]) / difficulty_tau)
            )
            for i, j, w in matching
        )
        result.update(
            score=2 * contribution / (n + m) if n + m else 1.0,
            prediction_count=n,
            match_count=len(matching),
            span_precision=proxy["precision"],
            span_recall=proxy["recall"],
            span_f1=proxy["f1"],
            valid_empty_reference_count=int(m == 0),
            valid_nonempty_reference_count=int(m > 0),
            false_positive_nonempty_count=int(m == 0 and n > 0),
            false_negative_empty_count=int(m > 0 and n == 0),
        )
        for prefix, edges in (("", matching), ("high_confidence_", [edge for edge in matching if edge[2] >= 0.8])):
            errors = [spans[i]["score"] - gold[j]["score"] for i, j, _ in edges]
            absolute, signed = sum(map(abs, errors)), sum(errors)
            result[prefix + "difficulty_mae"] = absolute / len(errors) if errors else 0.0
            result[prefix + "difficulty_bias"] = signed / len(errors) if errors else 0.0
            if prefix:
                result.update(
                    high_confidence_match_count=len(errors),
                    high_confidence_abs_error_sum=absolute,
                    high_confidence_error_sum=signed,
                )
            else:
                result.update(difficulty_abs_error_sum=absolute, difficulty_error_sum=signed)
    result["score"] *= score_scale_factor
    result["empty_reference_reward_sum" if m == 0 else "nonempty_reference_reward_sum"] = result["score"]
    return result


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    """Batch adapter; reject unequal lengths instead of silently dropping samples."""
    if len({len(values) for values in (data_sources, solution_strs, ground_truths, extra_infos)}) != 1:
        raise ValueError("Reward batch inputs must have equal lengths")
    return [
        compute_score(source, solution, truth, info, **kwargs)
        for source, solution, truth, info in zip(data_sources, solution_strs, ground_truths, extra_infos, strict=True)
    ]
