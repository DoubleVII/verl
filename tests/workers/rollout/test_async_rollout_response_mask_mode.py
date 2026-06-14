import torch
import pytest

from verl.workers.rollout.schemas import AsyncRolloutRequest, AsyncRolloutRequestStateEnum


class _TinyTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=False):
        text = "".join(f"<{msg['role']}>{msg['content']}" for msg in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def decode(self, ids):
        return " ".join(str(int(x)) for x in ids)


def _tokens(length: int, start: int = 100) -> torch.Tensor:
    return torch.arange(start, start + length, dtype=torch.long).unsqueeze(0)


def _fake_apply_chat_template(
    processing_class,
    messages,
    multi_modal_data,
    tools=None,
    add_generation_prompt=False,
    tokenize=False,
    return_dict=False,
):
    length = 1 + len(messages) + int(add_generation_prompt)
    input_ids = _tokens(length, start=10)
    attention_mask = torch.ones_like(input_ids)
    if return_dict:
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    return input_ids


def _make_request(monkeypatch, *, max_model_len=128, max_response_len=128):
    monkeypatch.setattr(AsyncRolloutRequest, "_handle_apply_chat_template", staticmethod(_fake_apply_chat_template))
    return AsyncRolloutRequest(
        request_id="req",
        state=AsyncRolloutRequestStateEnum.PENDING,
        messages=[{"role": "user", "content": "prompt"}],
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        reward_scores={},
        max_prompt_len=64,
        max_response_len=max_response_len,
        max_model_len=max_model_len,
        use_inference_chat_template=False,
        tokenization_sanity_check_mode="disable",
        processing_class=_TinyTokenizer(),
    )


def _add_two_assistant_turns(req):
    req.add_assistant_message(_TinyTokenizer(), "first", content_ids=_tokens(3, start=100))
    req.add_user_message(_TinyTokenizer(), "second prompt")
    req.add_assistant_message(_TinyTokenizer(), "second", content_ids=_tokens(2, start=200))


def test_all_assistant_response_mask_keeps_both_assistant_turns(monkeypatch):
    req = _make_request(monkeypatch)
    _add_two_assistant_turns(req)

    req.finalize(_TinyTokenizer(), {}, response_mask_mode="all_assistant")

    assert req.response_loss_mask.tolist() == [[1, 1, 1, 0, 1, 1]]


def test_last_assistant_response_mask_keeps_only_final_assistant_turn(monkeypatch):
    req = _make_request(monkeypatch)
    _add_two_assistant_turns(req)

    req.finalize(_TinyTokenizer(), {}, response_mask_mode="last_assistant")

    assert req.response_loss_mask.tolist() == [[0, 0, 0, 0, 1, 1]]


def test_last_assistant_response_mask_handles_truncated_final_turn(monkeypatch):
    req = _make_request(monkeypatch, max_response_len=6, max_model_len=8)
    _add_two_assistant_turns(req)

    req.finalize(_TinyTokenizer(), {}, response_mask_mode="last_assistant")

    assert req.response_loss_mask.tolist() == [[0, 0, 0, 0, 1]]


def test_last_assistant_response_mask_keeps_single_assistant_turn(monkeypatch):
    req = _make_request(monkeypatch)
    req.add_assistant_message(_TinyTokenizer(), "only", content_ids=_tokens(3, start=100))

    req.finalize(_TinyTokenizer(), {}, response_mask_mode="last_assistant")

    assert req.response_loss_mask.tolist() == [[1, 1, 1]]


def test_invalid_response_mask_mode_raises(monkeypatch):
    req = _make_request(monkeypatch)
    req.add_assistant_message(_TinyTokenizer(), "only", content_ids=_tokens(1, start=100))

    with pytest.raises(ValueError, match="Unsupported response_mask_mode"):
        req.finalize(_TinyTokenizer(), {}, response_mask_mode="bad")
