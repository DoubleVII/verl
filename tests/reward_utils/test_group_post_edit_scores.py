from types import SimpleNamespace

import pytest
import torch

from reward_utils.rm_lib import compute_group_post_edit_scores


class _Batch:
    def __init__(self, responses):
        self._items = {
            "responses": torch.tensor(responses, dtype=torch.long),
            "attention_mask": torch.ones((len(responses), len(responses[0])), dtype=torch.long),
        }
        self.batch_size = (len(responses),)

    def __getitem__(self, key):
        return self._items[key]


class _Data:
    def __init__(self, decoded_responses, extra_info, uids=None):
        self.batch = _Batch([[i + 1, 0, 0] for i in range(len(decoded_responses))])
        self.non_tensor_batch = {"extra_info": extra_info}
        if uids is not None:
            self.non_tensor_batch["uid"] = uids


class _Tokenizer:
    eos_token = "<eos>"

    def __init__(self, decoded=None):
        self.decoded = list(decoded or [])
        self.encoded_texts = []

    def decode(self, ids, skip_special_tokens=True):
        token_id = int(ids[0])
        return self.decoded[token_id - 1]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]

    def encode(self, text, add_special_tokens=False):
        self.encoded_texts.append(text)
        return list(range(max(1, len(text.split()))))


def _extra(mt_texts=None):
    return {
        "src_text": "source",
        "src_lang": "English",
        "trg_lang": "Chinese",
        "mt_texts": list(mt_texts or []),
    }


def _output(text):
    return SimpleNamespace(outputs=[SimpleNamespace(text=text)])


def test_default_mode_keeps_mt_group_advantage_without_reward_scale():
    input_tokenizer = _Tokenizer(["post edited"])
    tokenizer = _Tokenizer()
    data = _Data(["post edited"], [_extra(["baseline one", "baseline two"])])

    scores = compute_group_post_edit_scores(
        data,
        lambda prompts: [_output("A: 2, B: 4, C: 8")],
        tokenizer,
        input_tokenizer,
        extractor_type="none",
        max_prompt_length=1000,
        prompt_format="ranking_score",
        add_example=False,
        score_scale_factor=0.5,
        default_reward=-1.0,
        rm_max_candidates=4,
        overlong_buffer_cfg=None,
        enable_language_detection=False,
    )

    assert scores == {0: pytest.approx((8 - (2 + 4 + 8) / 3) * 0.5)}


def test_grpo_group_score_scores_once_per_uid_and_ignores_mt_texts():
    decoded = ["same edit", "same edit", "worse edit", "other edit", "best edit"]
    input_tokenizer = _Tokenizer(decoded)
    tokenizer = _Tokenizer()
    data = _Data(
        decoded,
        [
            _extra(["SHOULD_NOT_APPEAR"]),
            _extra(["SHOULD_NOT_APPEAR"]),
            _extra(["SHOULD_NOT_APPEAR"]),
            _extra(["SHOULD_NOT_APPEAR"]),
            _extra(["SHOULD_NOT_APPEAR"]),
        ],
        uids=["g1", "g1", "g1", "g2", "g2"],
    )
    calls = []

    def generate_fn(prompts):
        calls.append(prompts)
        return [_output("A: 9, B: 3"), _output("A: 5, B: 7")]

    scores = compute_group_post_edit_scores(
        data,
        generate_fn,
        tokenizer,
        input_tokenizer,
        extractor_type="none",
        max_prompt_length=1000,
        prompt_format="ranking_score",
        add_example=False,
        score_scale_factor=0.1,
        default_reward=-1.0,
        rm_max_candidates=2,
        overlong_buffer_cfg=None,
        enable_language_detection=False,
        score_mode="grpo_group_score",
    )

    assert len(calls) == 1
    assert len(calls[0]) == 2
    assert "SHOULD_NOT_APPEAR" not in "\n".join(tokenizer.encoded_texts)
    assert scores == {
        0: pytest.approx(0.9),
        1: pytest.approx(0.9),
        2: pytest.approx(0.3),
        3: pytest.approx(0.5),
        4: pytest.approx(0.7),
    }


def test_invalid_group_post_edit_score_mode_raises():
    input_tokenizer = _Tokenizer(["post edited"])
    tokenizer = _Tokenizer()
    data = _Data(["post edited"], [_extra(["baseline"])], uids=["g1"])

    with pytest.raises(ValueError, match="group post-edit score_mode"):
        compute_group_post_edit_scores(
            data,
            lambda prompts: [],
            tokenizer,
            input_tokenizer,
            extractor_type="none",
            max_prompt_length=1000,
            prompt_format="ranking_score",
            add_example=False,
            score_scale_factor=0.1,
            default_reward=-1.0,
            rm_max_candidates=2,
            overlong_buffer_cfg=None,
            enable_language_detection=False,
            score_mode="missing",
        )
