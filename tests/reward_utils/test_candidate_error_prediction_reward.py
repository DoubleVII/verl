import json
import math
from pathlib import Path

import pytest

from reward_utils._candidate_error_response import parse_response, validate_prediction
from reward_utils._hard_span_proxy import score_spans
from reward_utils.candidate_error_prediction_reward import compute_score, compute_score_batch

SOURCE = "one two three four"


def pred(*quotes, score=10, severity="major"):
    return dict(spans=[dict(source_span=q, score=score, severity=severity, reason="mismatch") for q in quotes])


def gold(*quotes):
    return dict(spans=[dict(source_span=q, severity="major") for q in quotes])


def reward(p, g=None, **kwargs):
    return compute_score(
        "candidate_error_prediction",
        json.dumps(p) if isinstance(p, dict) else p,
        gold("one") if g is None else g,
        dict(src_text=SOURCE),
        **kwargs,
    )


@pytest.mark.parametrize("score", range(11))
def test_exact_and_unmatched_calibration(score):
    p = score / 10
    exact = reward(pred("one", score=score))
    assert exact["score"] == pytest.approx(1 - 0.3 * (p - 1) ** 2)
    unmatched = reward(pred("four", score=score))
    assert unmatched["score"] == pytest.approx(-0.2 - 0.3 * p**2)
    assert unmatched["match_count"] == 0


def test_partial_soft_target_and_position_only_matching():
    results = [reward(pred("one two", score=s)) for s in (0, 5, 10)]
    w = math.sqrt(0.5)
    for s, row in zip((0, 5, 10), results, strict=True):
        assert row["span_f1"] == pytest.approx(w)
        assert row["calibration_loss"] == pytest.approx((s / 10 - w) ** 2)
        assert row["miss_penalty"] == pytest.approx(0.2 * (1 - w))
        assert row["match_count"] == 1
    assert results[1]["score"] > results[0]["score"]
    assert reward(pred("one", severity="minor")) == reward(pred("one"))


@pytest.mark.parametrize(
    "p,g,expected",
    [(pred(), gold(), 1), (pred(), gold("one"), -0.2), (pred("one", score=0), gold(), 0), (pred("one"), gold(), -0.3)],
)
def test_empty_cases(p, g, expected):
    assert reward(p, g)["score"] == pytest.approx(expected)


def test_duplicates_and_zero_confidence_cannot_dilute_calibration():
    row = reward(pred("one", "one"))
    assert row["match_count"] == 1
    assert row["span_f1"] == pytest.approx(2 / 3)
    assert row["calibration_loss"] == 1
    p = pred("one", score=5)
    original = reward(p)
    p["spans"] += pred("two", score=0)["spans"]
    extended = reward(p)
    assert extended["calibration_loss"] == original["calibration_loss"]
    assert extended["score"] < original["score"]


@pytest.mark.parametrize(
    "p",
    [
        "bad",
        "{}",
        "[]",
        "<think>unfinished",
        '<think>a</think><think>b</think>{"spans":[]}',
        pred("absent"),
        pred("one", score=True),
        pred("one", score=1.2),
        pred("one", score=float("nan")),
        pred("one", severity="other"),
    ],
)
def test_invalid_outputs(p):
    row = reward(p, score_scale_factor=3)
    assert row["score"] == -3
    assert row["invalid_output_penalty"] == 1
    assert row["miss_penalty"] == row["calibration_loss"] == 0


def test_envelopes_and_normalization():
    content = json.dumps(pred("ONE", score="8"))
    wrapped = (
        "<thinking>Compare meanings.</thinking>\n# Analysis\nCompare.\n# Final JSON\n```json\n" + content + "\n```"
    )
    row = reward(wrapped, think_tag="thinking")
    assert row["valid_output"] == 1
    assert row["score"] == pytest.approx(0.988)
    assert validate_prediction(pred("one"), "one one") is None
    assert reward("```json\n{}\n```\n```broken")["valid_output"] == 0


@pytest.mark.parametrize("name", ["calibration_weight", "miss_penalty_weight", "score_scale_factor"])
@pytest.mark.parametrize("value", [-1, True, float("nan"), float("inf")])
def test_invalid_parameters(name, value):
    with pytest.raises(ValueError):
        reward(pred(), **{name: value})


@pytest.mark.parametrize(
    "g",
    [
        "broken",
        {},
        gold("absent"),
        {"spans": [{"source_span": None}]},
        {"spans": [{"source_span": "one", "severity": "minor"}]},
    ],
)
def test_bad_references_raise(g):
    with pytest.raises(ValueError):
        reward(pred(), g)


def test_batch_and_decomposition(capsys):
    ps = [pred("one", score=8), pred(), pred("two"), "bad"]
    gs = [gold("one"), gold("one"), gold(), gold()]
    strings = [json.dumps(p) if isinstance(p, dict) else p for p in ps]
    rows = compute_score_batch(["x"] * 4, strings, gs, [dict(src_text=SOURCE)] * 4, score_scale_factor=2)
    assert rows == [reward(p, g, score_scale_factor=2) for p, g in zip(ps, gs, strict=True)]
    for row in rows:
        assert row["score"] == pytest.approx(
            2 * (row["span_f1"] - row["calibration_penalty"] - row["miss_penalty"] - row["invalid_output_penalty"])
        )
    summary = json.loads(capsys.readouterr().out.split(" ", 1)[1])
    assert summary["valid_output_count"] == 3
    assert summary["calibration_count"] == 2
    assert summary["empty_reference"]["sentence_error_rate"] == 1
    assert compute_score_batch([], [], [], []) == []
    assert json.loads(capsys.readouterr().out.split(" ", 1)[1])["calibration_mse"] is None
    compute_score_batch([], [], [], [], print_batch_summary=False)
    assert capsys.readouterr().out == ""
    with pytest.raises(ValueError):
        compute_score_batch(["x"], [], [], [])


def test_grrm_parser_and_validation_parity():
    root = Path("/home/yangs/repo/GRRM")
    if not root.exists():
        pytest.skip("Optional GRRM checkout unavailable")
    # Load the pure response module with its protocol import removed, to avoid
    # changing sys.path or importing the GRRM application/dependencies.
    namespace = {}
    protocol = {}
    exec(compile((root / "utils/candidate_error_protocol.py").read_text(), "protocol", "exec"), protocol)
    namespace["validate"] = protocol["validate"]
    exec(
        compile(
            (root / "utils/candidate_error_response.py")
            .read_text()
            .replace("from utils.candidate_error_protocol import validate", ""),
            "response",
            "exec",
        ),
        namespace,
    )
    for p in (pred(), pred("one"), pred("ONE", score="8"), pred("absent"), pred("one", "one")):
        text = "# Analysis\nCheck.\n```json\n" + json.dumps(p) + "\n```"
        assert parse_response(text) == namespace["parse_response"](text)
        local = validate_prediction(parse_response(text), SOURCE)
        remote = namespace["validate_compact"](
            namespace["parse_response"](text), dict(src_text=SOURCE), allow_minor=True
        )
        assert (local is None) == (remote is None)
        if local is not None:
            assert [(s["source_span"], s["confidence"], s["severity"]) for s in local["spans"]] == [
                (s["source_span"], s["confidence"], s["severity"]) for s in remote["spans"]
            ]
    assert (
        reward(pred("one"), calibration_weight=0, miss_penalty_weight=0)["score"]
        == score_spans(SOURCE, pred("one")["spans"], gold("one")["spans"])["f1"]
    )


@pytest.mark.parametrize("metric", ["f1", "f2"])
@pytest.mark.parametrize(
    "p,g,f1,f2",
    [
        (pred(), gold(), 1, 1),
        (pred(), gold("one"), 0, 0),
        (pred("one"), gold(), 0, 0),
        (pred("one"), gold("one"), 1, 1),
        (pred("one"), gold("four"), 0, 0),
        (pred("one"), gold("one", "two"), 2 / 3, 5 / 9),
        (pred("one", "two"), gold("one"), 2 / 3, 5 / 6),
        (pred("one", "one"), gold("one"), 2 / 3, 5 / 6),
        (pred("one two"), gold("one", "four"), 2 * math.sqrt(0.5) / 3, 5 * math.sqrt(0.5) / 9),
    ],
)
def test_localization_metric_and_penalties(metric, p, g, f1, f2):
    row = reward(p, g, localization_metric=metric, miss_penalty_weight=0.6, score_scale_factor=2)
    baseline = reward(p, g, miss_penalty_weight=0.6, score_scale_factor=2)
    assert row["span_f1"] == pytest.approx(f1)
    assert row["span_f2"] == pytest.approx(f2)
    assert row["localization_reward"] == pytest.approx(f1 if metric == "f1" else f2)
    for key in ("calibration_loss", "calibration_penalty", "miss_penalty", "match_count"):
        assert row[key] == baseline[key]
    assert row["score"] == pytest.approx(
        2 * (row["localization_reward"] - row["calibration_penalty"] - row["miss_penalty"])
    )
    if not p["spans"] and g["spans"]:
        assert row["score"] == -1.2
    if metric == "f1":
        assert row == baseline


@pytest.mark.parametrize("metric", [None, True, 2, "F2", "f3", []])
def test_invalid_localization_metric(metric):
    with pytest.raises(ValueError, match="localization_metric"):
        reward(pred(), localization_metric=metric)


@pytest.mark.parametrize("metric", ["f1", "f2"])
def test_localization_batch_and_invalid_output(metric, capsys):
    ps = [pred("one"), pred("one", "two"), "bad"]
    gs = [gold("one", "two"), gold("one"), gold("one")]
    rows = compute_score_batch(
        ["x"] * 3,
        [json.dumps(p) if isinstance(p, dict) else p for p in ps],
        gs,
        [dict(src_text=SOURCE)] * 3,
        localization_metric=metric,
        miss_penalty_weight=0.6,
    )
    assert rows == [
        reward(p, g, localization_metric=metric, miss_penalty_weight=0.6) for p, g in zip(ps, gs, strict=True)
    ]
    assert rows[-1]["score"] == -1
    assert rows[-1]["localization_reward"] == rows[-1]["span_f2"] == 0
    summary = json.loads(capsys.readouterr().out.split(" ", 1)[1])
    assert summary["parameters"]["localization_metric"] == metric
    assert summary["parameters"]["miss_penalty_weight"] == 0.6
    means = summary["batch_mean"]
    assert means["score"] == pytest.approx(
        means["localization_reward"]
        - means["calibration_penalty"]
        - means["miss_penalty"]
        - means["invalid_output_penalty"]
    )
