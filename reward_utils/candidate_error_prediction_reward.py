"""Position-only source-span assignment and soft-target confidence calibration."""

import json
import math
import re

from reward_utils._candidate_error_response import parse_response, validate_prediction
from reward_utils._hard_span_proxy import score_spans
from reward_utils.hard_prediction_reward import _validate_think_tag


def _reference(ground_truth, source):
    try:
        value = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid reference JSON") from error
    if not isinstance(value, dict) or not isinstance(value.get("spans"), list):
        raise ValueError("Invalid reference annotation")
    for span in value["spans"]:
        if not isinstance(span, dict):
            raise ValueError("Invalid reference span")
        quote = span.get("source_span")
        if (
            not isinstance(quote, str)
            or not quote.strip()
            or quote not in source
            or span.get("severity", "major") != "major"
        ):
            raise ValueError("Reference must contain only grounded major spans")
        # Repeated reference quotes may identify a particular annotated occurrence.
        start, end = span.get("source_start"), span.get("source_end")
        if start is not None or end is not None:
            if (
                type(start) is not int
                or type(end) is not int
                or start < 0
                or end != start + len(quote)
                or source[start:end] != quote
            ):
                raise ValueError("Invalid reference offsets")
    return value["spans"]


def _parse(text, think_tag):
    if not isinstance(text, str):
        return None
    text = text.strip()
    if re.search(r"<\s*/?\s*tool_call\b", text, re.I):
        return None
    opening, closing = f"<{think_tag}>", f"</{think_tag}>"
    pattern = rf"<\s*/?\s*{re.escape(think_tag)}(?![A-Za-z0-9_-])"
    if re.search(pattern, text, re.I):
        match = re.fullmatch(rf"{re.escape(opening)}(.*?){re.escape(closing)}\s*(.+)", text, re.S)
        if not match or not match[1].strip() or any(re.search(pattern, p, re.I) for p in match.groups()):
            return None
        text = match[2]
    return parse_response(text)


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
    *,
    calibration_weight=0.3,
    miss_penalty_weight=0.2,
    score_scale_factor=1.0,
    think_tag=None,
):
    """Return a scaled reward and unscaled numeric diagnostics.

    Reference/configuration errors raise; malformed predictions receive -1 * scale.
    Calibration sum uses a fixed reference denominator so extra zero-score predictions
    cannot dilute existing calibration errors. Valid rewards have no fixed lower bound.
    """
    for name, value in (
        ("calibration_weight", calibration_weight),
        ("miss_penalty_weight", miss_penalty_weight),
        ("score_scale_factor", score_scale_factor),
    ):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if (
        not isinstance(extra_info, dict)
        or not isinstance(extra_info.get("src_text"), str)
        or not extra_info["src_text"].strip()
    ):
        raise ValueError("extra_info must contain nonempty src_text")
    source = extra_info["src_text"]
    think_tag = _validate_think_tag(think_tag if think_tag is not None else extra_info.get("think_tag", "think"))
    gold = _reference(ground_truth, source)
    parsed = _parse(solution_str, think_tag)
    prediction = validate_prediction(parsed, source)
    m = len(gold)
    result = dict(
        score=0.0,
        parser_pass=int(isinstance(parsed, dict)),
        valid_output=int(prediction is not None),
        invalid_output_penalty=int(prediction is None),
        span_precision=0.0,
        span_recall=0.0,
        span_f1=0.0,
        prediction_count=0,
        reference_count=m,
        match_count=0,
        calibration_loss=0.0,
        calibration_penalty=0.0,
        calibration_error_sum=0.0,
        calibration_count=0,
        miss_loss=0.0,
        miss_penalty=0.0,
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
        spans = prediction["spans"]
        n = len(spans)
        proxy = score_spans(source, spans, gold)
        targets = {i: w for i, _, w in proxy["matching"]}
        coverage = {j: w for _, j, w in proxy["matching"]}
        errors = sum((span["confidence"] - targets.get(i, 0.0)) ** 2 for i, span in enumerate(spans))
        calibration = errors / max(m, 1)
        miss = sum(1 - coverage.get(j, 0.0) for j in range(m)) / m if m else 0.0
        result.update(
            span_precision=proxy["precision"],
            span_recall=proxy["recall"],
            span_f1=proxy["f1"],
            prediction_count=n,
            match_count=len(proxy["matching"]),
            calibration_loss=calibration,
            calibration_penalty=calibration_weight * calibration,
            calibration_error_sum=errors,
            calibration_count=n,
            miss_loss=miss,
            miss_penalty=miss_penalty_weight * miss,
            valid_empty_reference_count=int(m == 0),
            valid_nonempty_reference_count=int(m > 0),
            false_positive_nonempty_count=int(m == 0 and n > 0),
            false_negative_empty_count=int(m > 0 and n == 0),
        )
    result["score"] = score_scale_factor * (
        result["span_f1"] - result["calibration_penalty"] - result["miss_penalty"] - result["invalid_output_penalty"]
    )
    result["empty_reference_reward_sum" if m == 0 else "nonempty_reference_reward_sum"] = result["score"]
    return result


def _batch_summary(results, parameters):
    def total(key):
        return sum(row[key] for row in results)

    def ratio(a, b):
        return a / b if b else None

    size = len(results)
    summary = dict(
        batch_size=size,
        parameters=parameters,
        parser_pass_count=total("parser_pass"),
        valid_output_count=total("valid_output"),
        parser_pass_pct=ratio(100 * total("parser_pass"), size),
        valid_output_pct=ratio(100 * total("valid_output"), size),
        calibration_error_sum=total("calibration_error_sum"),
        calibration_count=total("calibration_count"),
        calibration_mse=ratio(total("calibration_error_sum"), total("calibration_count")),
        batch_mean={
            key: ratio(total(key), size)
            for key in (
                "score",
                "span_f1",
                "calibration_loss",
                "calibration_penalty",
                "miss_penalty",
                "invalid_output_penalty",
            )
        },
    )
    for group, error in (
        ("empty_reference", "false_positive_nonempty_count"),
        ("nonempty_reference", "false_negative_empty_count"),
    ):
        count, valid = total(group + "_count"), total("valid_" + group + "_count")
        summary[group] = dict(
            count=count,
            valid_count=valid,
            mean_reward=ratio(total(group + "_reward_sum"), count),
            sentence_error_count=total(error),
            sentence_error_rate=ratio(total(error), valid),
        )
    return summary


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, *, print_batch_summary=True, **kwargs):
    if len({len(values) for values in (data_sources, solution_strs, ground_truths, extra_infos)}) != 1:
        raise ValueError("Reward batch inputs must have equal lengths")
    if type(print_batch_summary) is not bool:
        raise ValueError("print_batch_summary must be a boolean")
    results = [
        compute_score(*args, **kwargs)
        for args in zip(data_sources, solution_strs, ground_truths, extra_infos, strict=True)
    ]
    if print_batch_summary:
        parameters = {
            key: kwargs.get(key, default)
            for key, default in (("calibration_weight", 0.3), ("miss_penalty_weight", 0.2), ("score_scale_factor", 1.0))
        }
        print(
            "[candidate_error_prediction_reward] " + json.dumps(_batch_summary(results, parameters), allow_nan=False),
            flush=True,
        )
    return results
