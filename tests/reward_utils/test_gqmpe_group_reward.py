from types import SimpleNamespace

import pytest
import torch

from reward_utils.helpers import group_extract_scores
from reward_utils.prompts import get_GQM_prompt
from reward_utils.rm_lib import MultiTaskSelfRewardProcessor


class _Batch:
    def __init__(self, size):
        self._items = {
            "responses": torch.tensor([[i + 1, 0, 0] for i in range(size)], dtype=torch.long),
            "attention_mask": torch.ones((size, 3), dtype=torch.long),
        }
        self.batch_size = (size,)

    def __getitem__(self, key):
        return self._items[key]


class _Data:
    def __init__(self, decoded_responses, extra_info, uids, abilities):
        self.batch = _Batch(len(decoded_responses))
        self.non_tensor_batch = {
            "extra_info": extra_info,
            "uid": uids,
            "ability": abilities,
        }


class _Tokenizer:
    eos_token = "<eos>"

    def __init__(self, decoded=None):
        self.decoded = list(decoded or [])
        self.encoded_texts = []

    def decode(self, ids, skip_special_tokens=True):
        return self.decoded[int(ids[0]) - 1]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]

    def encode(self, text, add_special_tokens=False):
        self.encoded_texts.append(text)
        return list(range(max(1, len(text.split()))))


def _output(text):
    return SimpleNamespace(outputs=[SimpleNamespace(text=text)])


def _gqmpe_response(
    ranking="B > A",
    scores="B: 8, A: 2",
    post_edit_analysis="Use B as the base and correct its terminology.",
    translation="final translation",
):
    return f"""Detailed candidate analysis.

### Final Ranking:

{ranking}

### Scores:

{scores}

# Post-edit Analysis
{post_edit_analysis}

# Final post-edited translation
```text
{translation}
```"""


def _config(**overrides):
    values = {
        "prompt_length": 1000,
        "custom_processor": {"extractor_type": "none"},
        "group_prompt_type": "ranking_score",
        "group_add_example": False,
        "score_scale_factor": 0.1,
        "mt_score_scale_factor": 0.1,
        "gpe_score_scale_factor": 0.1,
        "ranking_score_scale_factor": 0.1,
        "default_reward": -1.0,
        "rm_max_candidates": 4,
        "group_post_edit_score_mode": "mt_group_advantage",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _extra(mt_texts=None, **overrides):
    values = {
        "src_text": "source",
        "src_lang": "en",
        "trg_lang": "zh",
        "mt_texts": list(mt_texts or []),
    }
    values.update(overrides)
    return values


def test_get_gqm_prompt_builds_gqmpe_prompt_with_mapped_languages_and_ordered_candidates():
    prompt = get_GQM_prompt(
        "en",
        "zh",
        "hello",
        ["first", "second"],
        "ranking_score",
        task_type="gqmpe",
    )

    assert prompt.startswith(
        "Given a source text in English and multiple translation candidates in Chinese. "
        "Perform a step by step analysis and comparison of the translation quality for the candidates. "
        "Finally, rank and score the candidates with integer scores on a scale from 0 to 10. "
        "Then provide a detailed post-edit analysis and a final improved translation in Chinese"
    )
    assert (
        prompt.index("Translation A:") < prompt.index("first") < prompt.index("Translation B:") < prompt.index("second")
    )
    assert "Notes:" not in prompt
    assert "reference:" not in prompt


@pytest.mark.parametrize(
    "kwargs",
    [
        {"add_example": True},
        {"notes": "a note"},
        {"ref_text": "reference", "ref_lang": "en"},
        {"ref_lang": "en"},
    ],
)
def test_get_gqm_prompt_rejects_gqmpe_auxiliary_inputs(kwargs):
    with pytest.raises(ValueError, match="does not support examples, notes, or references"):
        get_GQM_prompt(
            "en",
            "zh",
            "hello",
            ["first", "second"],
            "ranking_score",
            task_type="gqmpe",
            **kwargs,
        )


def test_group_extract_scores_keeps_legacy_gqm_last_line_behavior():
    assert group_extract_scores("analysis\nB: 8, A: 2", "ranking_score", 2) == [2, 8]


@pytest.mark.parametrize(
    ("prompt_type", "response", "expected"),
    [
        (
            "score",
            "Detailed candidate analysis.\nB: 8, A: 2\n\n# Post-edit Analysis\nUse B.\n\n"
            "# Final post-edited translation\n```text\nfinal translation\n```",
            [2, 8],
        ),
        (
            "ranking",
            "Detailed candidate analysis.\n\n### Final Ranking:\n\nB > A\n\n"
            "# Post-edit Analysis\nUse B.\n\n# Final post-edited translation\n```text\nfinal translation\n```",
            [0, 1],
        ),
        ("ranking_score", _gqmpe_response(), [2, 8]),
    ],
)
def test_group_extract_scores_parses_complete_gqmpe_outputs(prompt_type, response, expected):
    assert group_extract_scores(response, prompt_type, 2, task_type="gqmpe") == expected


@pytest.mark.parametrize(
    "response",
    [
        _gqmpe_response().replace("# Post-edit Analysis", "# Editing Analysis"),
        _gqmpe_response(scores="B: 8"),
        _gqmpe_response(ranking="A > B"),
        _gqmpe_response(post_edit_analysis=""),
        _gqmpe_response()[:-3],
        _gqmpe_response() + "\nextra text",
    ],
)
def test_group_extract_scores_rejects_invalid_gqmpe_outputs(response):
    assert group_extract_scores(response, "ranking_score", 2, task_type="gqmpe") is None


def test_multitask_translation_uses_gqmpe_prompt_and_scores():
    tokenizer = _Tokenizer()
    input_tokenizer = _Tokenizer(["candidate one", "candidate two"])
    data = _Data(
        ["candidate one", "candidate two"],
        [_extra(), _extra()],
        ["translation-group", "translation-group"],
        ["translation", "translation"],
    )
    calls = []

    def generate_fn(prompts):
        calls.append(prompts)
        return [_output(_gqmpe_response())] if prompts else []

    processor = MultiTaskSelfRewardProcessor(
        config=_config(group_task_type="gqmpe"),
        tokenizer=tokenizer,
        input_tokenizer=input_tokenizer,
    )

    assert processor.compute_scores(data, generate_fn) == [pytest.approx(0.2), pytest.approx(0.8)]
    assert [len(prompts) for prompts in calls] == [1, 0, 0]
    assert "final improved translation in Chinese" in tokenizer.encoded_texts[0]


def test_multitask_group_post_edit_uses_gqmpe_prompt_and_relative_score():
    tokenizer = _Tokenizer()
    input_tokenizer = _Tokenizer(["post edit"])
    data = _Data(
        ["post edit"],
        [_extra(["baseline one", "baseline two"])],
        ["post-edit-group"],
        ["group_post_edit"],
    )
    response = _gqmpe_response(ranking="C > B > A", scores="C: 8, B: 4, A: 2")
    calls = []

    def generate_fn(prompts):
        calls.append(prompts)
        return [_output(response)] if prompts else []

    processor = MultiTaskSelfRewardProcessor(
        config=_config(group_task_type="gqmpe"),
        tokenizer=tokenizer,
        input_tokenizer=input_tokenizer,
    )

    assert processor.compute_scores(data, generate_fn) == [pytest.approx((8 - (2 + 4) / 2) * 0.1)]
    assert [len(prompts) for prompts in calls] == [0, 1, 0]
    assert "Translation C:" in tokenizer.encoded_texts[0]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"group_task_type": "unknown"}, "group_task_type"),
        ({"group_task_type": "gqmpe", "group_add_example": True}, "group_add_example"),
    ],
)
def test_multitask_rejects_invalid_gqmpe_configuration(overrides, message):
    with pytest.raises(ValueError, match=message):
        MultiTaskSelfRewardProcessor(
            config=_config(**overrides),
            tokenizer=_Tokenizer(),
            input_tokenizer=_Tokenizer(),
        )
