import json

import pytest

from reward_utils.ranking_score_reward import (
    _extract_gqmpe_ranking_score,
    gqmpe_ranking_score_reward_fn,
)


GROUND_TRUTH = json.dumps({"A": 4, "B": 9, "C": 8})


def _response(
    ranking="B > C > A",
    scores="B: 9, C: 8, A: 4",
    post_edit_analysis="Translation B is the best base, with one minor correction.",
    translation="最终译文。",
    language_tag="text",
):
    return f"""Detailed candidate analysis.

### Final Ranking:

{ranking}

### Scores:

{scores}

# Post-edit Analysis
{post_edit_analysis}

# Final post-edited translation
```{language_tag}
{translation}
```"""


def test_gqmpe_ranking_score_reward_scores_ranking_and_requires_post_edit():
    result = gqmpe_ranking_score_reward_fn(
        "TowerBlocks-GQMPE-Ranking.ranking_score",
        _response(),
        GROUND_TRUTH,
        score_scale_factor=0.5,
    )

    assert result == {
        "score": pytest.approx(1.0),
        "valid_answer": 1,
        "ranking_reward": pytest.approx(1.0),
        "score_reward": pytest.approx(1.0),
    }


def test_gqmpe_extractor_returns_post_edit_without_scoring_it():
    extracted = _extract_gqmpe_ranking_score(
        _response(translation="组合后的最终译文。", language_tag="")
    )

    assert extracted["ranking_text"] == "B > C > A"
    assert extracted["score_text"] == "B: 9, C: 8, A: 4"
    assert extracted["post_edit_mt"] == "组合后的最终译文。"


@pytest.mark.parametrize(
    "response",
    [
        _response(translation=""),
        _response(post_edit_analysis=""),
        _response()[:-3],
        _response() + "\nextra text",
        _response().replace("### Scores:", "### Candidate Scores:"),
    ],
)
def test_gqmpe_reward_rejects_invalid_or_missing_post_edit_format(response):
    result = gqmpe_ranking_score_reward_fn(
        "TowerBlocks-GQMPE-Ranking.ranking_score",
        response,
        GROUND_TRUTH,
    )

    assert result == {
        "score": 0,
        "valid_answer": 0,
        "ranking_reward": 0,
        "score_reward": 0,
    }


def test_gqmpe_reward_preserves_existing_score_semantics():
    result = gqmpe_ranking_score_reward_fn(
        "TowerBlocks-GQMPE-Ranking.ranking_score",
        _response(ranking="B > A > C", scores="B: 9, A: 8, C: 4"),
        GROUND_TRUTH,
    )

    assert result["valid_answer"] == 1
    assert result["score"] < 2
    assert result["ranking_reward"] < 1
    assert result["score_reward"] < 1
