from types import SimpleNamespace

import pytest
import torch

from reward_utils.rm_lib import MultiTaskSelfRewardProcessor, compute_group_post_edit_scores, compute_group_translation_scores


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
    def __init__(self, decoded_responses, extra_info, uids=None, abilities=None, messages=None):
        self.batch = _Batch([[i + 1, 0, 0] for i in range(len(decoded_responses))])
        self.non_tensor_batch = {"extra_info": extra_info}
        if uids is not None:
            self.non_tensor_batch["uid"] = uids
        if abilities is not None:
            self.non_tensor_batch["ability"] = abilities
        if messages is not None:
            self.non_tensor_batch["messages"] = messages


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

    assert scores == {0: pytest.approx((8 - (2 + 4) / 2) * 0.5)}


def test_group_translation_scores_can_return_reward_model_metadata_without_changing_scores():
    input_tokenizer = _Tokenizer(["candidate one", "candidate two"])
    tokenizer = _Tokenizer()
    data = _Data(
        ["candidate one", "candidate two"],
        [_extra(), _extra()],
        uids=["u0", "u0"],
    )

    scores, gqm_prompts, gqm_outputs = compute_group_translation_scores(
        data,
        lambda prompts: [_output("analysis\nA: 2, B: 8")],
        tokenizer,
        input_tokenizer,
        extractor_type="none",
        max_prompt_length=1000,
        prompt_type="ranking_score",
        add_example=False,
        score_scale_factor=0.5,
        default_reward=-1.0,
        overlong_buffer_cfg=None,
        enable_language_detection=False,
        return_reward_model_metadata=True,
    )

    assert scores == {0: pytest.approx(1.0), 1: pytest.approx(4.0)}
    assert set(gqm_prompts) == {0, 1}
    assert "prompt_token_ids" in gqm_prompts[0]
    assert "prompt_token_ids" in gqm_prompts[1]
    assert gqm_outputs == {0: "analysis\nA: 2, B: 8", 1: "analysis\nA: 2, B: 8"}


def test_group_post_edit_dedupes_duplicate_post_edit_and_tracks_score_index():
    input_tokenizer = _Tokenizer(["baseline two"])
    tokenizer = _Tokenizer()
    data = _Data(["baseline two"], [_extra(["baseline one", "baseline two"])])
    calls = []

    def generate_fn(prompts):
        calls.append(prompts)
        return [_output("A: 2, B: 8")]

    scores = compute_group_post_edit_scores(
        data,
        generate_fn,
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

    encoded = "\n".join(tokenizer.encoded_texts)
    assert len(calls[0]) == 1
    assert encoded.count("baseline two") == 1
    assert scores == {0: pytest.approx((8 - (2 + 8) / 2) * 0.5)}


def test_group_post_edit_duplicate_post_edit_mean_excludes_post_edit_copy():
    input_tokenizer = _Tokenizer(["candidate three"])
    tokenizer = _Tokenizer()
    data = _Data(["candidate three"], [_extra(["candidate one", "candidate two", "candidate three"])])

    scores = compute_group_post_edit_scores(
        data,
        lambda prompts: [_output("A: 6, B: 7, C: 8")],
        tokenizer,
        input_tokenizer,
        extractor_type="none",
        max_prompt_length=1000,
        prompt_format="ranking_score",
        add_example=False,
        score_scale_factor=1.0,
        default_reward=-1.0,
        rm_max_candidates=4,
        overlong_buffer_cfg=None,
        enable_language_detection=False,
    )

    assert scores == {0: pytest.approx(8 - (6 + 7 + 8) / 3)}


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


def test_multitask_gqm_post_edit_uses_last_assistant_message():
    config = SimpleNamespace(
        prompt_length=1000,
        custom_processor={"extractor_type": "codeblock"},
        group_prompt_type="ranking_score",
        group_add_example=False,
        score_scale_factor=0.1,
        gpe_score_scale_factor=0.1,
        default_reward=-1.0,
        rm_max_candidates=4,
        group_post_edit_score_mode="mt_group_advantage",
    )
    tokenizer = _Tokenizer()
    input_tokenizer = _Tokenizer(["this full decoded response should be ignored"])
    data = _Data(
        ["unused"],
        [_extra(["baseline"])],
        abilities=["gqm_post_edit"],
        messages=[
            {
                "messages": [
                    {"role": "user", "content": "gqm prompt"},
                    {"role": "assistant", "content": "A: 1, B: 0"},
                    {"role": "user", "content": "post edit prompt"},
                    {"role": "assistant", "content": "analysis\n```zh\nfinal edit\n```"},
                ]
            }
        ],
    )

    calls = []

    def generate_fn(prompts):
        calls.append(prompts)
        return [_output("A: 2, B: 8")]

    processor = MultiTaskSelfRewardProcessor(config=config, tokenizer=tokenizer, input_tokenizer=input_tokenizer)
    scores = processor.compute_scores(data, generate_fn)

    assert [len(call) for call in calls] == [0, 0, 1]
    assert "final edit" in "\n".join(tokenizer.encoded_texts)
    assert "this full decoded response should be ignored" not in "\n".join(tokenizer.encoded_texts)
    assert scores == [pytest.approx((8 - 2) * 0.1)]


def test_multitask_gqm_post_edit_reuse_first_turn_scores_default_off():
    config = SimpleNamespace(
        prompt_length=1000,
        custom_processor={"extractor_type": "codeblock"},
        group_prompt_type="ranking_score",
        group_add_example=False,
        score_scale_factor=0.1,
        gpe_score_scale_factor=0.1,
        default_reward=-1.0,
        rm_max_candidates=4,
        group_post_edit_score_mode="mt_group_advantage",
    )
    tokenizer = _Tokenizer()
    input_tokenizer = _Tokenizer(["ignored full response"])
    data = _Data(
        ["unused"],
        [_extra(["candidate one", "candidate two"])],
        abilities=["gqm_post_edit"],
        messages=[
            {
                "messages": [
                    {"role": "assistant", "content": "A: 2, B: 8"},
                    {"role": "assistant", "content": "```zh\ncandidate two\n```"},
                ]
            }
        ],
    )
    calls = []

    def generate_fn(prompts):
        calls.append(prompts)
        return [_output("A: 1, B: 9")]

    processor = MultiTaskSelfRewardProcessor(config=config, tokenizer=tokenizer, input_tokenizer=input_tokenizer)
    scores = processor.compute_scores(data, generate_fn)

    assert [len(call) for call in calls] == [0, 0, 1]
    assert scores == [pytest.approx((9 - (1 + 9) / 2) * 0.1)]


def test_multitask_gqm_post_edit_reuses_first_turn_scores_for_duplicate_post_edit():
    config = SimpleNamespace(
        prompt_length=1000,
        custom_processor={
            "extractor_type": "codeblock",
            "reuse_gqm_post_edit_first_turn_scores": True,
        },
        group_prompt_type="ranking_score",
        group_add_example=False,
        score_scale_factor=0.1,
        gpe_score_scale_factor=0.1,
        default_reward=-1.0,
        rm_max_candidates=4,
        group_post_edit_score_mode="mt_group_advantage",
    )
    tokenizer = _Tokenizer()
    input_tokenizer = _Tokenizer(["ignored full response"])
    data = _Data(
        ["unused"],
        [_extra(["candidate one", "candidate two"])],
        abilities=["gqm_post_edit"],
        messages=[
            {
                "messages": [
                    {"role": "assistant", "content": "Analysis\nA: 2, B: 8"},
                    {"role": "assistant", "content": "```zh\n candidate two \n```"},
                ]
            }
        ],
    )
    calls = []

    def generate_fn(prompts):
        calls.append(prompts)
        return []

    processor = MultiTaskSelfRewardProcessor(config=config, tokenizer=tokenizer, input_tokenizer=input_tokenizer)
    scores = processor.compute_scores(data, generate_fn)

    assert calls == [[], [], []]
    assert tokenizer.encoded_texts == []
    assert scores == [pytest.approx((8 - (2 + 8) / 2) * 0.1)]


def test_multitask_gqm_post_edit_reuse_falls_back_when_first_turn_parse_fails():
    config = SimpleNamespace(
        prompt_length=1000,
        custom_processor={
            "extractor_type": "codeblock",
            "reuse_gqm_post_edit_first_turn_scores": True,
        },
        group_prompt_type="ranking_score",
        group_add_example=False,
        score_scale_factor=0.1,
        gpe_score_scale_factor=0.1,
        default_reward=-1.0,
        rm_max_candidates=4,
        group_post_edit_score_mode="mt_group_advantage",
    )
    tokenizer = _Tokenizer()
    input_tokenizer = _Tokenizer(["ignored full response"])
    data = _Data(
        ["unused"],
        [_extra(["candidate one", "candidate two"])],
        abilities=["gqm_post_edit"],
        messages=[
            {
                "messages": [
                    {"role": "assistant", "content": "not parseable"},
                    {"role": "assistant", "content": "```zh\ncandidate two\n```"},
                ]
            }
        ],
    )
    calls = []

    def generate_fn(prompts):
        calls.append(prompts)
        return [_output("A: 1, B: 9")]

    processor = MultiTaskSelfRewardProcessor(config=config, tokenizer=tokenizer, input_tokenizer=input_tokenizer)
    scores = processor.compute_scores(data, generate_fn)

    assert [len(call) for call in calls] == [0, 0, 1]
    assert scores == [pytest.approx((9 - (1 + 9) / 2) * 0.1)]


def test_multitask_gqm_post_edit_reuse_scores_only_duplicates_and_falls_back_for_others():
    config = SimpleNamespace(
        prompt_length=1000,
        custom_processor={
            "extractor_type": "codeblock",
            "reuse_gqm_post_edit_first_turn_scores": True,
        },
        group_prompt_type="ranking_score",
        group_add_example=False,
        score_scale_factor=0.1,
        gpe_score_scale_factor=0.1,
        default_reward=-1.0,
        rm_max_candidates=4,
        group_post_edit_score_mode="mt_group_advantage",
    )
    tokenizer = _Tokenizer()
    input_tokenizer = _Tokenizer(["ignored one", "ignored two"])
    data = _Data(
        ["unused one", "unused two"],
        [_extra(["candidate one", "candidate two"]), _extra(["baseline one", "baseline two"])],
        abilities=["gqm_post_edit", "gqm_post_edit"],
        messages=[
            {
                "messages": [
                    {"role": "assistant", "content": "A: 2, B: 8"},
                    {"role": "assistant", "content": "```zh\ncandidate two\n```"},
                ]
            },
            {
                "messages": [
                    {"role": "assistant", "content": "A: 7, B: 3"},
                    {"role": "assistant", "content": "```zh\nnew edit\n```"},
                ]
            },
        ],
    )
    calls = []

    def generate_fn(prompts):
        calls.append(prompts)
        return [_output("A: 1, B: 3, C: 9")]

    processor = MultiTaskSelfRewardProcessor(config=config, tokenizer=tokenizer, input_tokenizer=input_tokenizer)
    scores = processor.compute_scores(data, generate_fn)

    assert [len(call) for call in calls] == [0, 0, 1]
    assert "new edit" in "\n".join(tokenizer.encoded_texts)
    assert "candidate two" not in "\n".join(tokenizer.encoded_texts)
    assert scores == [
        pytest.approx((8 - (2 + 8) / 2) * 0.1),
        pytest.approx((9 - (1 + 3) / 2) * 0.1),
    ]


def test_multitask_gqm_post_edit_grpo_group_score_uses_uid_group():
    config = SimpleNamespace(
        prompt_length=1000,
        custom_processor={"extractor_type": "codeblock"},
        group_prompt_type="ranking_score",
        group_add_example=False,
        score_scale_factor=0.1,
        gpe_score_scale_factor=0.1,
        default_reward=-1.0,
        rm_max_candidates=4,
        group_post_edit_score_mode="grpo_group_score",
    )
    tokenizer = _Tokenizer()
    input_tokenizer = _Tokenizer(["ignored one", "ignored two"])
    data = _Data(
        ["unused one", "unused two"],
        [_extra(["SHOULD_NOT_APPEAR"]), _extra(["SHOULD_NOT_APPEAR"])],
        uids=["g1", "g1"],
        abilities=["gqm_post_edit", "gqm_post_edit"],
        messages=[
            {"messages": [{"role": "assistant", "content": "```zh\nedit one\n```"}]},
            {"messages": [{"role": "assistant", "content": "```zh\nedit two\n```"}]},
        ],
    )

    processor = MultiTaskSelfRewardProcessor(config=config, tokenizer=tokenizer, input_tokenizer=input_tokenizer)
    scores = processor.compute_scores(data, lambda prompts: [_output("A: 3, B: 7")])

    assert "SHOULD_NOT_APPEAR" not in "\n".join(tokenizer.encoded_texts)
    assert scores == [pytest.approx(0.3), pytest.approx(0.7)]


def _multitask_config(prompt_length=1000, score_mode="mt_group_advantage"):
    return SimpleNamespace(
        prompt_length=prompt_length,
        custom_processor={"extractor_type": "none"},
        group_prompt_type="ranking_score",
        group_add_example=False,
        score_scale_factor=0.1,
        mt_score_scale_factor=0.1,
        gpe_score_scale_factor=0.1,
        default_reward=-1.0,
        rm_max_candidates=4,
        group_post_edit_score_mode=score_mode,
        ranking_score_scale_factor=0.1,
    )


def _ranking_extra():
    return {
        "src_text": "source",
        "src_lang": "English",
        "trg_lang": "Chinese",
    }


def test_multitask_pure_ranking_still_calls_empty_generation_stages():
    tokenizer = _Tokenizer()
    input_tokenizer = _Tokenizer(["reasoning\nRanking:\nA > B\nScore:\nA: 2, B: 1"])
    data = _Data(
        ["unused"],
        [_ranking_extra()],
        uids=["rank-1"],
        abilities=["ranking"],
        messages=[{"messages": []}],
    )
    data.non_tensor_batch["reward_model"] = [{"ground_truth": '{"A": 2, "B": 1}'}]
    calls = []

    def generate_fn(prompts):
        calls.append(prompts)
        return []

    processor = MultiTaskSelfRewardProcessor(
        config=_multitask_config(),
        tokenizer=tokenizer,
        input_tokenizer=input_tokenizer,
    )
    scores = processor.compute_scores(data, generate_fn)

    assert calls == [[], [], []]
    assert scores == [pytest.approx(0.2)]


def test_multitask_mixed_batch_with_gpe_prompts_keeps_fixed_generation_order():
    tokenizer = _Tokenizer()
    input_tokenizer = _Tokenizer(["ignored ranking", "post edit"])
    data = _Data(
        ["unused ranking", "unused gpe"],
        [_ranking_extra(), _extra(["baseline"])],
        uids=["rank-1", "gpe-1"],
        abilities=["ranking", "gqm_post_edit"],
        messages=[
            {"messages": []},
            {"messages": [{"role": "assistant", "content": "post edit"}]},
        ],
    )
    calls = []

    def generate_fn(prompts):
        calls.append(prompts)
        if prompts:
            return [_output("A: 2, B: 8")]
        return []

    processor = MultiTaskSelfRewardProcessor(
        config=_multitask_config(),
        tokenizer=tokenizer,
        input_tokenizer=input_tokenizer,
    )
    scores = processor.compute_scores(data, generate_fn)

    assert [len(call) for call in calls] == [0, 0, 1]
    assert scores[0] == pytest.approx(-1.0)
    assert scores[1] == pytest.approx((8 - 2) * 0.1)


def test_multitask_mixed_batch_without_local_gpe_prompts_calls_empty_gpe_stage():
    tokenizer = _Tokenizer()
    input_tokenizer = _Tokenizer(["translation output"])
    data = _Data(
        ["unused"],
        [_extra()],
        uids=["translation-1"],
        abilities=["translation"],
        messages=[{"messages": []}],
    )
    calls = []

    def generate_fn(prompts):
        calls.append(prompts)
        return []

    processor = MultiTaskSelfRewardProcessor(
        config=_multitask_config(score_mode="grpo_group_score"),
        tokenizer=tokenizer,
        input_tokenizer=input_tokenizer,
    )
    scores = processor.compute_scores(data, generate_fn)

    assert calls == [[], [], []]
    assert scores == [-1.0]


def test_group_post_edit_filtered_empty_prompt_still_calls_generate_fn():
    tokenizer = _Tokenizer()
    input_tokenizer = _Tokenizer(["post edit"])
    data = _Data(
        ["unused"],
        [_extra(["baseline"])],
        uids=["gpe-1"],
    )
    calls = []

    scores = compute_group_post_edit_scores(
        data,
        lambda prompts: calls.append(prompts) or [],
        tokenizer,
        input_tokenizer,
        extractor_type="none",
        max_prompt_length=0,
        prompt_format="ranking_score",
        add_example=False,
        score_scale_factor=0.1,
        default_reward=-1.0,
        rm_max_candidates=4,
        overlong_buffer_cfg=None,
        enable_language_detection=False,
    )

    assert calls == [[]]
    assert scores == {}
