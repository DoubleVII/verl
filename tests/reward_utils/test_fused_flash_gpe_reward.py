import json
from types import SimpleNamespace

import pytest
import torch

from reward_utils.rm_lib import (
    REWARD_MODEL_PROMPTS_KEY,
    REWARD_MODEL_RESPONSES_KEY,
    FusedFlashGPERewardModelProcessor,
    RewardProcessorOutput,
    _myers_insert_delete_distance,
)


class _Batch:
    def __init__(self, size):
        self._items = {
            "responses": torch.tensor([[idx + 1, 0, 0] for idx in range(size)], dtype=torch.long),
            "attention_mask": torch.ones((size, 3), dtype=torch.long),
        }
        self.batch_size = (size,)

    def __getitem__(self, key):
        return self._items[key]


class _Data:
    def __init__(self, responses, extra_info, uids=None):
        self.batch = _Batch(len(responses))
        self.non_tensor_batch = {
            "extra_info": extra_info,
            "uid": uids or ["group"] * len(responses),
        }


class _Tokenizer:
    eos_token = "<eos>"

    def __init__(self, decoded=None, candidate_tokens=None):
        self.decoded = list(decoded or [])
        self.candidate_tokens = candidate_tokens or {}
        self.encoded_texts = []

    def decode(self, ids, skip_special_tokens=True):
        return self.decoded[int(ids[0]) - 1]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]

    def encode(self, text, add_special_tokens=False):
        self.encoded_texts.append(text)
        if text in self.candidate_tokens:
            return self.candidate_tokens[text]
        return list(range(max(1, len(text.split()))))


def _output(text):
    return SimpleNamespace(outputs=[SimpleNamespace(text=text)])


def _config(**custom_overrides):
    custom_processor = {
        "extractor_type": "line",
        "diversity_algorithm": "none",
        "diversity_penalty_weight": 1.0,
    }
    custom_processor.update(custom_overrides)
    return SimpleNamespace(
        prompt_length=1000,
        custom_processor=custom_processor,
        group_prompt_type="ranking_score",
        group_add_example=False,
        score_scale_factor=0.1,
        default_reward=-1.0,
    )


def _extra(prompt_type="fixed_4", max_candidates=4):
    return {
        "src_text": "source",
        "src_lang": "en",
        "trg_lang": "zh",
        "prompt_type": prompt_type,
        "max_candidates": max_candidates,
    }


def _fused(candidates, final_translation="final translation"):
    candidate_json = json.dumps({"translations": candidates}, ensure_ascii=False)
    return f"""<thinking>
generate candidates
</thinking>
<response>
{candidate_json}
</response>
<thinking>
compare candidates
</thinking>
<response>
```text
{final_translation}
```
</response>"""


def _processor(responses, config=None, candidate_tokens=None):
    input_tokenizer = _Tokenizer(responses, candidate_tokens=candidate_tokens)
    processor = FusedFlashGPERewardModelProcessor(
        config=config or _config(),
        tokenizer=_Tokenizer(),
        input_tokenizer=input_tokenizer,
    )
    return processor


@pytest.mark.parametrize(
    ("prompt_type", "max_candidates", "candidate_count"),
    [("fixed_4", 4, 4), ("fixed_16", 16, 16), ("adaptive", 8, 2)],
)
def test_valid_candidate_counts_score_only_final_translation(prompt_type, max_candidates, candidate_count):
    responses = [
        _fused([f"candidate {idx}" for idx in range(candidate_count)], "final one"),
        _fused([f"alternative {idx}" for idx in range(candidate_count)], "final two"),
    ]
    data = _Data(responses, [_extra(prompt_type, max_candidates), _extra(prompt_type, max_candidates)])
    processor = _processor(responses)

    scores = processor.compute_scores(data, lambda prompts: [_output("A: 2, B: 8")])

    assert scores == [pytest.approx(0.2), pytest.approx(0.8)]
    rm_prompt = "\n".join(processor.tokenizer.encoded_texts)
    assert "final one" in rm_prompt
    assert "final two" in rm_prompt
    assert "candidate 0" not in rm_prompt


@pytest.mark.parametrize(
    "response",
    [
        "<thinking>x</thinking><response>{}</response>",
        _fused(["a", "b"]).replace('{"translations": ["a", "b"]}', "not json"),
        _fused(["a", "b"]).replace("translations", "candidates"),
        _fused(["a", ""]),
        _fused(["a", "b"]).replace("```text\nfinal translation\n```", "final translation"),
    ],
)
def test_malformed_fused_output_returns_default_without_rm_prompt(response):
    calls = []
    data = _Data([response], [_extra("adaptive", 4)])
    processor = _processor([response])

    scores = processor.compute_scores(data, lambda prompts: calls.append(prompts) or [])

    assert scores == [-1.0]
    assert calls == [[]]


@pytest.mark.parametrize(
    ("prompt_type", "max_candidates", "candidate_count"),
    [
        ("fixed_4", 4, 3),
        ("fixed_4", 8, 4),
        ("fixed_16", 16, 15),
        ("adaptive", 8, 1),
        ("adaptive", 2, 3),
        ("unknown", 4, 4),
        (None, 4, 4),
    ],
)
def test_invalid_candidate_count_or_config_returns_default(
    prompt_type, max_candidates, candidate_count
):
    response = _fused([f"candidate {idx}" for idx in range(candidate_count)])
    calls = []
    data = _Data([response], [_extra(prompt_type, max_candidates)])
    processor = _processor([response])

    scores = processor.compute_scores(data, lambda prompts: calls.append(prompts) or [])

    assert scores == [-1.0]
    assert calls == [[]]


def test_none_diversity_keeps_group_mt_reward():
    responses = [
        _fused(["Same", " same ", "third", "fourth"], "final one"),
        _fused(["a", "b", "c", "d"], "final two"),
    ]
    processor = _processor(responses, _config(diversity_algorithm="none"))
    data = _Data(responses, [_extra(), _extra()])

    scores = processor.compute_scores(data, lambda prompts: [_output("A: 2, B: 8")])

    assert scores == [pytest.approx(0.2), pytest.approx(0.8)]


def test_exact_match_normalizes_whitespace_and_case_and_stacks_overlong_penalty(capsys):
    responses = [
        _fused(["Same  Translation", " same translation ", "third", "fourth"], "final one"),
        _fused(["a", "b", "c", "d"], "final two"),
    ]
    config = _config(
        diversity_algorithm="exact_match",
        diversity_penalty_weight=1.0,
        overlong_buffer={
            "enable": True,
            "max_resp_len": 4,
            "len": 2,
            "penalty_factor": 0.5,
        },
    )
    processor = _processor(responses, config)
    data = _Data(responses, [_extra(), _extra()])

    scores = processor.compute_scores(data, lambda prompts: [_output("A: 2, B: 8")])

    assert scores == [pytest.approx(0.2 - 0.25 - 1.0), pytest.approx(0.8 - 0.25)]
    stats = capsys.readouterr().out
    assert "[FUSED_FLASH_GPE_STATS] total=2 valid=2" in stats
    assert "duplicate_hits=1" in stats
    assert "mt_score_mean=0.250000" in stats
    assert "penalty_mean=0.500000" in stats


def test_token_myers_uses_mean_normalized_pairwise_distance():
    responses = [
        _fused(["a b", "a c"], "final one"),
        _fused(["x", "y"], "final two"),
    ]
    candidate_tokens = {
        "a b": [1, 2],
        "a c": [1, 3],
        "x": [4],
        "y": [5],
    }
    processor = _processor(
        responses,
        _config(diversity_algorithm="token_myers", diversity_penalty_weight=1.0),
        candidate_tokens=candidate_tokens,
    )
    data = _Data(responses, [_extra("adaptive", 2), _extra("adaptive", 2)])

    scores = processor.compute_scores(data, lambda prompts: [_output("A: 2, B: 8")])

    assert _myers_insert_delete_distance([1, 2], [1, 3]) == 2
    assert scores == [pytest.approx(0.2 - 0.5), pytest.approx(0.8)]


@pytest.mark.parametrize(
    ("clip", "expected_penalty"),
    [(0.2, 0.15), (0.5, 0.0), (0.8, 0.0)],
)
def test_token_myers_clip_creates_diversity_dead_zone(clip, expected_penalty):
    responses = [
        _fused(["a b", "a c"], "final one"),
        _fused(["x", "y"], "final two"),
    ]
    candidate_tokens = {
        "a b": [1, 2],
        "a c": [1, 3],
        "x": [4],
        "y": [5],
    }
    processor = _processor(
        responses,
        _config(
            diversity_algorithm="token_myers",
            diversity_penalty_weight=0.5,
            diversity_penalty_clip=clip,
        ),
        candidate_tokens=candidate_tokens,
    )
    data = _Data(responses, [_extra("adaptive", 2), _extra("adaptive", 2)])

    scores = processor.compute_scores(data, lambda prompts: [_output("A: 2, B: 8")])

    assert scores == [pytest.approx(0.2 - expected_penalty), pytest.approx(0.8)]


@pytest.mark.parametrize("diversity_algorithm", ["exact_match", "token_myers"])
def test_diversity_penalty_max_caps_enabled_algorithms(diversity_algorithm):
    first_candidates = (
        ["same", " SAME "]
        if diversity_algorithm == "exact_match"
        else ["a b", "a c"]
    )
    responses = [
        _fused(first_candidates, "final one"),
        _fused(["x", "y"], "final two"),
    ]
    candidate_tokens = {
        "a b": [1, 2],
        "a c": [1, 3],
        "x": [4],
        "y": [5],
    }
    processor = _processor(
        responses,
        _config(
            diversity_algorithm=diversity_algorithm,
            diversity_penalty_weight=2.0,
            diversity_penalty_max=0.3,
        ),
        candidate_tokens=candidate_tokens,
    )
    data = _Data(responses, [_extra("adaptive", 2), _extra("adaptive", 2)])

    scores = processor.compute_scores(data, lambda prompts: [_output("A: 2, B: 8")])

    assert scores[0] == pytest.approx(0.2 - 0.3)
    assert scores[1] == pytest.approx(0.8)


def test_reward_model_metadata_matches_group_processor_shape():
    responses = [
        _fused(["a", "b", "c", "d"], "final one"),
        _fused(["e", "f", "g", "h"], "final two"),
    ]
    processor = _processor(responses, _config(return_reward_model_metadata=True))
    data = _Data(responses, [_extra(), _extra()])

    result = processor.compute_scores(data, lambda prompts: [_output("analysis\nA: 2, B: 8")])

    assert isinstance(result, RewardProcessorOutput)
    assert result.scores == [pytest.approx(0.2), pytest.approx(0.8)]
    assert all(result.non_tensor_batch[REWARD_MODEL_PROMPTS_KEY])
    assert result.non_tensor_batch[REWARD_MODEL_RESPONSES_KEY] == [
        "analysis\nA: 2, B: 8",
        "analysis\nA: 2, B: 8",
    ]


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"diversity_algorithm": "cosine"}, "diversity_algorithm"),
        ({"diversity_penalty_weight": -0.1}, "non-negative"),
        ({"diversity_penalty_clip": -0.1}, "diversity_penalty_clip"),
        ({"diversity_penalty_clip": 1.1}, "diversity_penalty_clip"),
        ({"diversity_penalty_max": -0.1}, "diversity_penalty_max"),
    ],
)
def test_invalid_diversity_configuration_raises(overrides, error):
    with pytest.raises(ValueError, match=error):
        _processor([], _config(**overrides))
