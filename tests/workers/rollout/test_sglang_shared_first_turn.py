import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.interactions.gqm_post_edit_interaction import GQM_POST_EDIT_PROMPT
from verl.workers.rollout.schemas import AsyncRolloutRequest, AsyncRolloutRequestStateEnum
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=False, return_dict=False):
        text = "".join(f"<{msg['role']}>{msg['content']}" for msg in messages)
        if add_generation_prompt:
            text += "<assistant>"
        if not tokenize:
            return text
        ids = torch.arange(10, 10 + max(1, len(messages) + int(add_generation_prompt)), dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(ids)
        if return_dict:
            return {"input_ids": ids, "attention_mask": attention_mask}
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(x)) for x in ids)


class _Interaction:
    async def start_interaction(self, instance_id=None, **kwargs):
        return instance_id

    async def generate_response(self, instance_id, messages, **kwargs):
        return False, GQM_POST_EDIT_PROMPT, 0.0, {}


def _fake_apply_chat_template(
    processing_class,
    messages,
    multi_modal_data,
    tools=None,
    add_generation_prompt=False,
    tokenize=False,
    return_dict=False,
):
    content_len = 0
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        content_len += max(1, len(str(content).split()))
    length = max(1, len(messages) + int(add_generation_prompt) + content_len)
    ids = torch.arange(10, 10 + length, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(ids)
    if return_dict:
        return {"input_ids": ids, "attention_mask": attention_mask}
    return ids


@pytest.fixture(autouse=True)
def _patch_chat_template(monkeypatch):
    monkeypatch.setattr(AsyncRolloutRequest, "_handle_apply_chat_template", staticmethod(_fake_apply_chat_template))


def _make_rollout():
    rollout = object.__new__(SGLangRollout)
    rollout._tp_rank = 0
    rollout.processing_class = _TinyTokenizer()
    rollout.pad_token_id = 0
    rollout.sampling_params = {"max_new_tokens": 8}
    rollout.config = SimpleNamespace(
        response_length=64,
        prompt_length=16,
        max_model_len=256,
        calculate_log_probs=False,
        skip_tokenizer_init=False,
        val_kwargs=SimpleNamespace(top_k=-1, top_p=1.0, temperature=0),
        multi_turn=SimpleNamespace(
            max_assistant_turns=2,
            max_user_turns=1,
            response_mask_mode="last_assistant",
            shared_first_turn_by_uid=True,
            use_inference_chat_template=False,
            tokenization_sanity_check_mode="disable",
        ),
    )
    rollout.interaction_map = {"gqm_post_edit": _Interaction()}
    rollout._tool_schemas = []
    rollout._tool_map = {}
    rollout._function_call_parser = None
    return rollout


def _make_prompt_batch():
    input_ids = torch.ones((6, 4), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(4, dtype=torch.long).repeat(6, 1)
    batch = TensorDict(
        {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids},
        batch_size=6,
    )
    raw_prompt = np.array([[{"role": "user", "content": f"prompt-{i // 3}"}] for i in range(6)], dtype=object)
    interaction_kwargs = np.array([{"name": "gqm_post_edit"} for _ in range(6)], dtype=object)
    uids = np.array(["u0", "u0", "u0", "u1", "u1", "u1"], dtype=object)
    return DataProto(
        batch=batch,
        non_tensor_batch={"raw_prompt": raw_prompt, "interaction_kwargs": interaction_kwargs, "uid": uids},
    )


def _make_mixed_prompt_batch():
    prompts = _make_prompt_batch()
    prompts.non_tensor_batch["interaction_kwargs"] = np.concatenate(
        [
            prompts.non_tensor_batch["interaction_kwargs"],
            np.array([{}, {}, {}], dtype=object),
        ]
    )
    prompts.non_tensor_batch["raw_prompt"] = np.concatenate(
        [
            prompts.non_tensor_batch["raw_prompt"],
            np.array([[{"role": "user", "content": "ranking"}] for _ in range(3)], dtype=object),
        ]
    )
    prompts.non_tensor_batch["uid"] = np.concatenate(
        [prompts.non_tensor_batch["uid"], np.array(["r0", "r0", "r0"], dtype=object)]
    )
    prompts.batch = TensorDict(
        {
            key: torch.cat([value, value[:3]], dim=0)
            for key, value in prompts.batch.items()
        },
        batch_size=9,
    )
    return prompts


def test_shared_first_turn_fans_out_second_turn_by_uid():
    rollout = _make_rollout()
    prompts = _make_prompt_batch()
    req_list = rollout._preprocess_prompt_to_async_rollout_requests(prompts)
    calls = []

    async def fake_engine(req, sampling_params, image_data=None):
        calls.append(len([msg for msg in req.messages if msg.role == "assistant"]))
        assistant_count = calls[-1]
        text = f"first-{req.batch_data_id}" if assistant_count == 0 else f"second-{req.batch_data_id}"
        return {"text": text, "meta_info": {"finish_reason": {"type": "stop"}}}

    rollout._handle_engine_call = fake_engine

    output = asyncio.run(rollout._async_rollout_shared_first_turn_by_uid(prompts, req_list, True, False))
    output = sorted(output, key=lambda req: (req.batch_data_id, req.rollout_offset))

    assert len(output) == 6
    assert calls.count(0) == 2
    assert calls.count(1) == 6
    assert [req.batch_data_id for req in output] == list(range(6))
    assert [req.messages[1].content for req in output[:3]] == ["first-0"] * 3
    assert [req.messages[1].content for req in output[3:]] == ["first-3"] * 3
    assert all(req.messages[2].role == "user" and req.messages[2].content == GQM_POST_EDIT_PROMPT for req in output)
    assert [req.messages[3].content for req in output] == [f"second-{i}" for i in range(6)]
    assert all(req.state == AsyncRolloutRequestStateEnum.COMPLETED for req in output)
    assert all(req.response_loss_mask.sum().item() > 0 for req in output)


def test_shared_first_turn_requires_uid():
    rollout = _make_rollout()
    prompts = _make_prompt_batch()
    prompts.non_tensor_batch.pop("uid")
    req_list = rollout._preprocess_prompt_to_async_rollout_requests(prompts)

    with pytest.raises(ValueError, match="requires uid"):
        rollout._shared_first_turn_enabled(prompts, req_list, is_validate=False)


def test_shared_first_turn_handles_mixed_gqm_post_edit_and_single_turn_tasks():
    rollout = _make_rollout()
    prompts = _make_mixed_prompt_batch()
    req_list = rollout._preprocess_prompt_to_async_rollout_requests(prompts)
    calls = []

    async def fake_engine(req, sampling_params, image_data=None):
        calls.append((req.batch_data_id, len([msg for msg in req.messages if msg.role == "assistant"])))
        assistant_count = calls[-1][1]
        text = f"first-{req.batch_data_id}" if assistant_count == 0 else f"second-{req.batch_data_id}"
        return {"text": text, "meta_info": {"finish_reason": {"type": "stop"}}}

    rollout._handle_engine_call = fake_engine

    assert rollout._shared_first_turn_enabled(prompts, req_list, is_validate=False) is True
    output = asyncio.run(rollout._async_rollout_shared_first_turn_by_uid(prompts, req_list, True, False))
    output = sorted(output, key=lambda req: (req.batch_data_id, req.rollout_offset))

    assert len(output) == 9
    assert sum(1 for _, assistant_count in calls if assistant_count == 0) == 5
    assert sum(1 for _, assistant_count in calls if assistant_count == 1) == 6
    assert [req.batch_data_id for req in output] == list(range(9))
    assert [len([msg for msg in req.messages if msg.role == "assistant"]) for req in output[:6]] == [2] * 6
    assert [len([msg for msg in req.messages if msg.role == "assistant"]) for req in output[6:]] == [1] * 3


def test_shared_first_turn_disabled_falls_back():
    rollout = _make_rollout()
    rollout.config.multi_turn.shared_first_turn_by_uid = False
    prompts = _make_prompt_batch()
    req_list = rollout._preprocess_prompt_to_async_rollout_requests(prompts)

    assert rollout._shared_first_turn_enabled(prompts, req_list, is_validate=False) is False
