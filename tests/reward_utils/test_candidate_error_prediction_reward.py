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
