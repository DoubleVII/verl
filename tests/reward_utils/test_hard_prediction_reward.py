import importlib.util
import itertools
import json
import math
import random
import sys
from pathlib import Path

import pytest

from reward_utils._hard_span_proxy import maximum_weight_matching, score_spans
from reward_utils.hard_prediction_reward import compute_score, compute_score_batch


def annotation(*quotes, score=5):
    return dict(
        analysis="difficulty",
        score=score if quotes else 0,
        spans=[dict(source_span=q, score=score, explanation="decision") for q in quotes],
    )


def reward(prediction, reference, source="one two three four", **kwargs):
    text = prediction if isinstance(prediction, str) else json.dumps(prediction)
    return compute_score("hard_prediction", text, json.dumps(reference), dict(src_text=source), **kwargs)


@pytest.mark.parametrize("error", [0, 1, 2, 4])
def test_calibration_curve(error):
    result = reward(annotation("one", score=5 + error), annotation("one"))
    assert result["score"] == pytest.approx(0.7 + 0.3 * math.exp(-error / 2))
    assert result["span_f1"] == 1
    assert result["difficulty_mae"] == error
    assert result["difficulty_bias"] == error
    assert result["high_confidence_match_count"] == 1


@pytest.mark.parametrize("p,g,expected", [((), (), 1), (("one",), (), 0), ((), ("one",), -0.1)])
def test_empty_transitions(p, g, expected):
    result = reward(annotation(*p), annotation(*g))
    assert result["score"] == expected
    assert result["empty_reference_count"] == int(not g)
    assert result["nonempty_reference_count"] == int(bool(g))
    assert result["false_positive_nonempty_count"] == int(bool(p) and not g)
    assert result["false_negative_empty_count"] == int(not p and bool(g))


def test_missed_extra_and_wide_spans():
    assert reward(annotation("one"), annotation("one", "two"))["score"] == pytest.approx(2 / 3 - 0.05)
    assert reward(annotation("one", "two"), annotation("one"))["score"] == pytest.approx(2 / 3)
    assert reward(annotation("one two"), annotation("one"))["score"] == pytest.approx(
        (3 / 6) ** 0.5 - 0.1 * (1 - (3 / 6) ** 0.5)
    )
    assert reward(annotation("one"), annotation("four"))["score"] == pytest.approx(-0.1)
    pred, gold = annotation("one two", score=10), annotation("one", "two", score=1)
    result = reward(pred, gold, difficulty_weight=0, miss_penalty_weight=0)
    assert result["score"] == score_spans("one two three four", pred["spans"], gold["spans"])["f1"]


@pytest.mark.parametrize("difficulty,expected", [(1, -0.02), (9, -0.18), (10, -0.2)])
def test_difficulty_weighted_empty_prediction(difficulty, expected):
    result = reward(annotation(), annotation("one", score=difficulty))
    assert result["score"] == pytest.approx(expected)
    assert result["joint_reward"] == 0
    assert result["miss_penalty"] == pytest.approx(-expected)


def test_missed_difficulty_priority_and_coverage():
    gold = annotation("one", "two", score=1)
    gold["spans"][1]["score"] = gold["score"] = 9
    misses_high = reward(annotation("one", score=1), gold)
    misses_low = reward(annotation("two", score=9), gold)
    assert misses_high["miss_penalty"] == pytest.approx(0.09)
    assert misses_low["miss_penalty"] == pytest.approx(0.01)
    assert misses_low["score"] > misses_high["score"]
    assert reward(annotation("one", score=10), gold)["miss_penalty"] == misses_high["miss_penalty"]
    assert reward(gold, gold)["miss_penalty"] == 0
    assert reward(annotation(), gold, miss_penalty_weight=0)["score"] == 0
    partial = reward(annotation("one two"), annotation("one"))
    assert partial["miss_penalty"] == pytest.approx(0.1 * (1 - math.sqrt(0.5)))
    assert reward("bad", gold)["miss_penalty"] == 0


@pytest.mark.parametrize("prediction", [annotation(), annotation("one", score=9), "bad", "{}", "[]"])
def test_reward_decomposition(prediction):
    result = reward(prediction, annotation("one"), score_scale_factor=3)
    assert result["joint_reward"] == pytest.approx(result["span_f1"] - result["difficulty_penalty"])
    assert result["score"] == pytest.approx(
        3 * (result["joint_reward"] - result["miss_penalty"] - result["invalid_output_penalty"])
    )












def test_scores_cannot_change_position_assignment_including_ties():
    # Both predicted quotes contain each golden quote, creating tied assignments.
    pred = annotation("one two", "one two three", score=1)
    gold = annotation("one", "two", score=1)
    gold["spans"][1]["score"] = gold["score"] = 10
    source = "one two three four"
    matching = score_spans(source, pred["spans"], gold["spans"])["matching"]
    for scores in [(1, 10), (10, 1)]:
        for span, score in zip(pred["spans"], scores, strict=True):
            span["score"] = score
        pred["score"] = max(scores)
        expected = (
            2
            * sum(w * (0.7 + 0.3 * math.exp(-abs(scores[i] - gold["spans"][j]["score"]) / 2)) for i, j, w in matching)
            / 4
        )
        assert reward(pred, gold, miss_penalty_weight=0)["score"] == pytest.approx(expected)


@pytest.mark.parametrize("bad", [True, 5.0, 0, 11, "5"])
def test_invalid_span_scores(bad):
    pred = annotation("one")
    pred["spans"][0]["score"] = bad
    assert reward(pred, annotation("one"))["score"] == -1


@pytest.mark.parametrize(
    "change",
    [
        {"analysis": ""},
        {"score": True},
        {"score": 5.0},
        {"score": 4},
        {"spans": ["one"]},
        {"spans": [dict(source_span="absent", score=5, explanation="decision")]},
        {"spans": [dict(source_span=" one", score=5, explanation="decision")]},
        {"spans": [dict(source_span="one", score=5, explanation="")]},
        {"spans": annotation("one", "one")["spans"]},
    ],
)
def test_invalid_annotations(change):
    pred = annotation("one")
    pred.update(change)
    result = reward(pred, annotation("one"))
    assert result["score"] == -1
    assert result["valid_output"] == 0
    assert result["nonempty_reference_reward_sum"] == -1
    assert result["valid_nonempty_reference_count"] == 0


def test_limits():
    for source in ["a" * 49, "a b c d e f g h i", "one\ntwo"]:
        assert reward(annotation(source), annotation(), source)["score"] == -1
    source = "a" * 49
    result = compute_score(
        "x", json.dumps(annotation(source)), annotation(source), dict(src_text=source, max_span_chars=49)
    )
    assert result["score"] == 1


@pytest.mark.parametrize(
    "envelope",
    [
        "{}",
        "```json\n{}\n```",
        "```\n{}\n```",
        "<think>reasoning</think>{}",
        "reasoning</think>{}",
        "<think>reasoning</think>```json\n{}\n```",
    ],
)
def test_response_envelopes(envelope):
    text = envelope.replace("{}", json.dumps(annotation("one")))
    assert reward(text, annotation("one"))["score"] == 1


@pytest.mark.parametrize("text", ["", "{}", "[]", "not JSON", "```json\n{}", "text {}", "{} trailing"])
def test_no_json_repair(text):
    assert reward(text, annotation())["score"] == -1


def test_analysis_codeblock_parser_selection():
    truth = annotation("one")
    text = "# Step-by-step Analysis\n\nSpecific reasoning.\n\n# Final JSON\n\n```json\n" + json.dumps(truth) + "\n```"
    info = dict(src_text="one", response_format="analysis_codeblock")
    assert compute_score("x", text, truth, info)["score"] == 1
    assert reward(text, truth, response_format="analysis_codeblock")["score"] == 1
    assert reward("<think>reasoning</think>" + text, truth, response_format="analysis_codeblock")["score"] == 1
    assert compute_score("x", text, truth, info, response_format="json")["score"] == -1
    assert reward(json.dumps(truth), truth, response_format="analysis_codeblock")["score"] == -1
    for invalid in [
        text + "trailing",
        text.replace("Specific reasoning.", ""),
        text.replace("Specific reasoning.", "```json\n{}\n```"),
        text.replace("# Final JSON", "# Answer"),
        text[:-3],
    ]:
        assert reward(invalid, truth, response_format="analysis_codeblock")["score"] == -1
    result = compute_score_batch(["x"], [text], [truth], [info], response_format="analysis_codeblock")
    assert result[0]["score"] == 1
    with pytest.raises(ValueError, match="response_format"):
        reward(text, truth, response_format="unknown")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(difficulty_weight=-0.1),
        dict(difficulty_weight=1.1),
        dict(difficulty_weight=float("nan")),
        dict(difficulty_weight=True),
        dict(difficulty_tau=0),
        dict(difficulty_tau=float("inf")),
        dict(score_scale_factor=-1),
        dict(score_scale_factor=float("nan")),
        dict(score_scale_factor=float("inf")),
        dict(score_scale_factor=True),
        dict(score_scale_factor="2"),
        dict(miss_penalty_weight=-1),
        dict(miss_penalty_weight=True),
        dict(miss_penalty_weight=float("nan")),
        dict(miss_penalty_weight=float("inf")),
        dict(miss_penalty_weight="0.2"),
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        reward(annotation(), annotation(), **kwargs)


@pytest.mark.parametrize("scale", [0, 0.5, 2])
def test_final_reward_scaling(scale):
    predictions = [annotation("one", score=7), annotation(), annotation("one"), "bad", annotation()]
    references = [annotation("one"), annotation(), annotation(), annotation("one"), annotation("one", score=9)]
    texts = [json.dumps(p) if isinstance(p, dict) else p for p in predictions]
    scaled = compute_score_batch(
        ["hard_prediction"] * 5,
        texts,
        references,
        [dict(src_text="one")] * 5,
        score_scale_factor=scale,
    )
    for pred, ref, result in zip(predictions, references, scaled, strict=True):
        baseline = reward(pred, ref, source="one")
        for key in baseline:
            expected = (
                baseline[key] * scale
                if key
                in (
                    "score",
                    "empty_reference_reward_sum",
                    "nonempty_reference_reward_sum",
                )
                else baseline[key]
            )
            assert result[key] == pytest.approx(expected)


def test_reference_errors_and_batch_shape():
    for truth, info in [
        ("bad", dict(src_text="one")),
        (annotation("absent"), dict(src_text="one")),
        (annotation(), {}),
        (annotation(), dict(src_text="one", max_span_chars=0)),
    ]:
        with pytest.raises(ValueError):
            compute_score("x", "{}", truth, info)
    with pytest.raises(ValueError):
        compute_score_batch(["x"], [], [], [])
    assert compute_score_batch([], [], [], []) == []
    outputs = compute_score_batch(
        ["x"] * 2, [json.dumps(annotation("one")), "bad"], [annotation("one")] * 2, [dict(src_text="one")] * 2
    )
    assert [out["score"] for out in outputs] == [1, -1]
    assert outputs[0].keys() == outputs[1].keys()
    assert all(math.isfinite(v) for out in outputs for v in out.values())


def test_assignment_against_exhaustive_search():
    rng = random.Random(42)
    for n, m in itertools.product(range(1, 5), repeat=2):
        weights = [[rng.random() for _ in range(m)] for _ in range(n)]
        matrix = weights if n <= m else list(zip(*weights, strict=True))
        expected = max(
            sum(matrix[i][j] for i, j in enumerate(assignment))
            for assignment in itertools.permutations(range(len(matrix[0])), len(matrix))
        )
        assert sum(w for _, _, w in maximum_weight_matching(weights)) == pytest.approx(expected)


def test_grrm_parity_when_available(monkeypatch):
    root = Path(__file__).resolve().parents[2].parent / "GRRM"
    if not (root / "eval/span_proxy.py").is_file():
        pytest.skip("Sibling GRRM checkout unavailable")

    # Load only the two pure Python evaluator modules; avoid utils namespace conflicts.
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    load("eval.span_matching", root / "eval/span_matching.py")
    proxy = load("grrm_span_proxy", root / "eval/span_proxy.py")
    cases = [
        ("one two three", ["one two"], ["one", "two"]),
        ("cat and cat dog", ["cat"], ["cat dog"]),
        ("折扣与折旧", ["折扣与"], ["与折旧"]),
        ("ＡＢＣ — 中国", ["ＡＢＣ"], ["中国"]),
        ("お金壊してきてちょうだい", ["壊してきて"], ["壊してきて"]),
        ("a", [], []),
        ("a", ["a"], []),
    ]
    for source, pred, gold in cases:
        assert score_spans(source, pred, gold) == proxy.score_spans(source, pred, gold)
