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


def _parse_response(text, response_format="json"):
    if not isinstance(text, str):
        return None
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.strip()
    if response_format == "analysis_codeblock":
        # Same explicit envelope as GRRM utils/difficulty_response.py.
        match = re.fullmatch(
            r"\s*# Step-by-step Analysis\s*\n(.+?)\n# Final JSON\s*\n"
            r"```json[ \t]*\n(.*?)\n```\s*",
            text,
            re.DOTALL,
        )
        if not match or not match[1].strip() or "```" in match[1]:
            return None
        text = match[2]
    elif text.startswith("```"):
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
    response_format=None,
    miss_penalty_weight=0.2,
):
    """Return reward and numeric diagnostics accepted by verl reward managers.

    Missing/corrupt reference metadata raises; invalid model outputs score -1.
    Error means are zero when their corresponding match count is zero.
    Group sums/counts allow pooled empty/nonempty rates without NaN metrics.
    score_scale_factor scales final scores (including invalid-output penalties)
    and group reward sums; localization and difficulty diagnostics stay unscaled.
    """
    if (
        isinstance(miss_penalty_weight, bool)
        or not isinstance(miss_penalty_weight, (int, float))
        or not math.isfinite(miss_penalty_weight)
        or miss_penalty_weight < 0
    ):
        raise ValueError("miss_penalty_weight must be finite and nonnegative")
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
    response_format = response_format if response_format is not None else extra_info.get("response_format", "json")
    if response_format not in ("json", "analysis_codeblock"):
        raise ValueError("response_format must be json or analysis_codeblock")
    try:
        reference = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid reference JSON") from error
    source = extra_info["src_text"]
    reference = validate_annotation(reference, source, **limits)
    if reference is None:
        raise ValueError("Invalid reference annotation")
    parsed = _parse_response(solution_str, response_format)
    prediction = validate_annotation(parsed, source, **limits)
    m = len(reference["spans"])
    result = dict(
        score=-1.0,
        parser_pass=int(isinstance(parsed, dict)),
        joint_reward=0.0,
        difficulty_penalty=0.0,
        miss_penalty=0.0,
        invalid_output_penalty=int(prediction is None),
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
        joint_reward = 2 * contribution / (n + m) if n + m else 1.0
        coverage = {j: w for _, j, w in matching}
        miss_penalty = (
            miss_penalty_weight
            * sum(span["score"] / 10 * (1 - coverage.get(j, 0.0)) for j, span in enumerate(gold))
            / m
            if m
            else 0.0
        )
        result.update(
            joint_reward=joint_reward,
            difficulty_penalty=proxy["f1"] - joint_reward,
            miss_penalty=miss_penalty,
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
    result["score"] = score_scale_factor * (
        result["joint_reward"] - result["miss_penalty"] - result["invalid_output_penalty"]
    )
    result["empty_reference_reward_sum" if m == 0 else "nonempty_reference_reward_sum"] = result["score"]
    return result


def _batch_summary(results, parameters):
    """Summarize this invocation only; missing denominators serialize as null."""

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    def total(key, rows=results):
        return sum(row[key] for row in rows)

    def means(rows, keys):
        return {key: ratio(total(key, rows), len(rows)) for key in keys}

    valid = [row for row in results if row["valid_output"]]
    size, parsed_count = len(results), total("parser_pass")
    summary = dict(
        batch_size=size,
        parser_pass_count=parsed_count,
        valid_output_count=len(valid),
        parser_pass_pct=ratio(100 * parsed_count, size),
        valid_output_pct=ratio(100 * len(valid), size),
        parser_failure_count=size - parsed_count,
        validation_failure_count=parsed_count - len(valid),
        parameters=parameters,
        batch_mean=means(
            results,
            ("score", "span_f1", "joint_reward", "difficulty_penalty", "miss_penalty", "invalid_output_penalty"),
        ),
        valid_mean=means(
            valid,
            (
                "span_precision",
                "span_recall",
                "span_f1",
                "joint_reward",
                "difficulty_penalty",
                "miss_penalty",
                "prediction_count",
                "reference_count",
            ),
        ),
    )
    for prefix, count_key, abs_key, error_key in (
        ("matched", "match_count", "difficulty_abs_error_sum", "difficulty_error_sum"),
        (
            "high_confidence",
            "high_confidence_match_count",
            "high_confidence_abs_error_sum",
            "high_confidence_error_sum",
        ),
    ):
        count = total(count_key)
        summary[prefix] = dict(
            count=count, difficulty_mae=ratio(total(abs_key), count), difficulty_bias=ratio(total(error_key), count)
        )
    for group, error_key in (
        ("empty_reference", "false_positive_nonempty_count"),
        ("nonempty_reference", "false_negative_empty_count"),
    ):
        count, valid_count = total(group + "_count"), total("valid_" + group + "_count")
        summary[group] = dict(
            count=count,
            mean_reward=ratio(total(group + "_reward_sum"), count),
            valid_count=valid_count,
            sentence_error_count=total(error_key),
            sentence_error_rate=ratio(total(error_key), valid_count),
        )
    return summary


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, *, print_batch_summary=True, **kwargs):
    """Batch adapter; reject unequal lengths instead of silently dropping samples."""
    if len({len(values) for values in (data_sources, solution_strs, ground_truths, extra_infos)}) != 1:
        raise ValueError("Reward batch inputs must have equal lengths")
    if type(print_batch_summary) is not bool:
        raise ValueError("print_batch_summary must be a boolean")
    results = [
        compute_score(source, solution, truth, info, **kwargs)
        for source, solution, truth, info in zip(data_sources, solution_strs, ground_truths, extra_infos, strict=True)
    ]
    if print_batch_summary:
        parameters = {
            key: kwargs.get(key, default)
            for key, default in (
                ("miss_penalty_weight", 0.2),
                ("difficulty_weight", 0.3),
                ("difficulty_tau", 2.0),
                ("score_scale_factor", 1.0),
            )
        }
        print(
            "[hard_prediction_reward] " + json.dumps(_batch_summary(results, parameters), allow_nan=False), flush=True
        )
    return results
