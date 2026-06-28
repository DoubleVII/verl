from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.interactions.gqm_post_edit_interaction import GQM_POST_EDIT_PROMPT
from verl.trainer.ppo.opsd_utils import (
    DISTILLATION_LOSS_MASK_KEY,
    OPSD_TEACHER_PROMPT_KEY,
    ORIGINAL_RESPONSE_MASK_KEY,
    REWARD_MODEL_PROMPTS_KEY,
    REWARD_MODEL_RESPONSES_KEY,
    SELECTED_RESPONSE_MASK_KEY,
    attach_opsd_metadata,
    build_opsd_teacher_batch,
    build_reward_model_teacher_batch,
    extract_opsd_teacher_prompt,
    restore_selected_response_teacher_logprobs,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.texts = []

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        text = "\n".join(message["content"] for message in messages)
        if add_generation_prompt:
            text += "\nAssistant:"
        return text

    def __call__(self, prompts, return_tensors="pt", padding=True, add_special_tokens=False):
        self.texts.extend(list(prompts))
        encoded = [[ord(char) for char in prompt] for prompt in prompts]
        max_len = max(len(ids) for ids in encoded)
        input_ids = []
        attention_mask = []
        for ids in encoded:
            pad = [self.pad_token_id] * (max_len - len(ids))
            input_ids.append(ids + pad)
            attention_mask.append([1] * len(ids) + [0] * len(pad))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=False):
        if token_ids == [1, 2, 3] or token_ids == [4, 5, 6]:
            return "Decoded original GQM prompt"
        return "".join(chr(token_id) for token_id in token_ids if token_id != self.pad_token_id)


def _config(
    prompt_path="extra_info.teacher_prompt",
    teacher_max_prompt_length=None,
    response_source="full_response",
):
    config = OmegaConf.create(
        {
            "distillation": {
                "teacher": {
                    "source": "ref_policy",
                    "prompt_source": "data_teacher_prompt",
                    "teacher_prompt_path": prompt_path,
                    "response_source": response_source,
                    "prompt_constructor_path": "reward_utils/prompts.py",
                    "prompt_constructor_name": "gqm_post_edit_teacher_prompt_constructor",
                }
            },
            "actor_rollout_ref": {
                "rollout": {
                    "multi_turn": {
                        "response_mask_mode": "last_assistant",
                    }
                }
            },
            "data": {
                "max_prompt_length": 64,
                "truncation": "right",
                "apply_chat_template_kwargs": {},
            },
        }
    )
    if teacher_max_prompt_length is not None:
        config.distillation.teacher.max_prompt_length = teacher_max_prompt_length
    return config


def _batch(extra_info=None, teacher_prompt=None):
    tensors = TensorDict(
        {
            "input_ids": torch.tensor([[0, 0, 11, 12, 13, 21, 22], [0, 14, 15, 16, 17, 23, 24]]),
            "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1]]),
            "position_ids": torch.tensor([[0, 0, 0, 1, 2, 3, 4], [0, 0, 1, 2, 3, 4, 5]]),
            "prompts": torch.tensor([[0, 0, 11, 12, 13], [0, 14, 15, 16, 17]]),
            "responses": torch.tensor([[21, 22], [23, 24]]),
            "response_mask": torch.tensor([[1, 1], [1, 1]]),
        },
        batch_size=2,
    )
    non_tensors = {}
    if extra_info is not None:
        non_tensors["extra_info"] = np.array(extra_info, dtype=object)
    if teacher_prompt is not None:
        non_tensors["teacher_prompt"] = np.array(teacher_prompt, dtype=object)
    return DataProto(batch=tensors, non_tensor_batch=non_tensors)


def test_extracts_default_extra_info_teacher_prompt():
    sample = {"extra_info": {"teacher_prompt": "full teacher prompt"}}

    assert extract_opsd_teacher_prompt(sample, "extra_info.teacher_prompt") == "full teacher prompt"


def test_attach_opsd_metadata_rejects_missing_or_empty_prompt():
    batch = _batch(extra_info=[{"teacher_prompt": "ok"}, {"teacher_prompt": " "}])

    with pytest.raises(ValueError, match="OPSD teacher prompt is missing or empty"):
        attach_opsd_metadata(batch, _config())


def test_build_teacher_batch_preserves_responses_and_recomputes_masks():
    batch = _batch(
        extra_info=[
            {"teacher_prompt": "Full teacher prompt A"},
            {"teacher_prompt": "Full teacher prompt B"},
        ]
    )
    tokenizer = FakeTokenizer()

    attach_opsd_metadata(batch, _config())
    teacher_batch = build_opsd_teacher_batch(batch, tokenizer, _config())

    assert torch.equal(teacher_batch.batch["responses"], batch.batch["responses"])
    assert torch.equal(teacher_batch.batch["response_mask"], batch.batch["response_mask"])
    assert teacher_batch.batch["input_ids"].shape == (2, 66)
    assert teacher_batch.batch["attention_mask"].shape == teacher_batch.batch["input_ids"].shape
    assert teacher_batch.batch["position_ids"].shape == teacher_batch.batch["input_ids"].shape
    assert torch.all(teacher_batch.batch["attention_mask"][:, -2:] == 1)
    assert tokenizer.texts == ["Full teacher prompt A", "Full teacher prompt B"]


def test_build_teacher_batch_keeps_last_prompt_token_unpadded_for_varied_prompt_lengths():
    batch = _batch(
        extra_info=[
            {"teacher_prompt": "short"},
            {"teacher_prompt": "much longer teacher prompt"},
        ]
    )

    attach_opsd_metadata(batch, _config())
    teacher_batch = build_opsd_teacher_batch(batch, FakeTokenizer(), _config())

    response_length = batch.batch["responses"].shape[-1]
    last_prompt_attention = teacher_batch.batch["attention_mask"][:, -response_length - 1]
    assert torch.all(last_prompt_attention == 1)


def test_build_teacher_batch_renders_chat_list_prompts():
    batch = _batch(
        extra_info=[
            {"teacher_prompt": [{"role": "user", "content": "Problem A"}]},
            {"teacher_prompt": [{"role": "user", "content": "Problem B"}]},
        ]
    )
    tokenizer = FakeTokenizer()

    attach_opsd_metadata(batch, _config())
    build_opsd_teacher_batch(batch, tokenizer, _config())

    assert tokenizer.texts == ["Problem A\nAssistant:", "Problem B\nAssistant:"]


def test_build_teacher_batch_can_select_last_assistant_responses_only():
    batch = _batch(
        extra_info=[
            {"teacher_prompt": "Full teacher prompt A"},
            {"teacher_prompt": "Full teacher prompt B"},
        ]
    )
    batch.batch["responses"] = torch.tensor([[31, 32, 41, 42], [33, 43, 44, 0]])
    batch.batch["response_mask"] = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 0]])
    batch.batch["attention_mask"] = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 0]]
    )
    batch.meta_info["pad_token_id"] = 0
    config = _config(response_source="last_assistant_response")

    attach_opsd_metadata(batch, config)
    teacher_batch = build_opsd_teacher_batch(batch, FakeTokenizer(), config)

    assert torch.equal(teacher_batch.batch["responses"], torch.tensor([[41, 42], [43, 44]]))
    assert torch.equal(teacher_batch.batch["response_mask"], torch.ones(2, 2, dtype=torch.long))
    assert torch.equal(teacher_batch.batch[SELECTED_RESPONSE_MASK_KEY], torch.ones(2, 2, dtype=torch.long))
    assert torch.equal(teacher_batch.batch[ORIGINAL_RESPONSE_MASK_KEY], batch.batch["response_mask"])
    assert torch.equal(teacher_batch.batch["input_ids"][:, -2:], torch.tensor([[41, 42], [43, 44]]))


def test_build_teacher_batch_rejects_last_assistant_response_without_last_assistant_mask_mode():
    batch = _batch(extra_info=[{"teacher_prompt": "A"}, {"teacher_prompt": "B"}])
    config = _config(response_source="last_assistant_response")
    config.actor_rollout_ref.rollout.multi_turn.response_mask_mode = "all_assistant"

    attach_opsd_metadata(batch, config)
    with pytest.raises(ValueError, match="response_mask_mode=last_assistant"):
        build_opsd_teacher_batch(batch, FakeTokenizer(), config)


def test_build_teacher_batch_masks_empty_selected_last_assistant_response():
    batch = _batch(extra_info=[{"teacher_prompt": "A"}, {"teacher_prompt": "B"}])
    batch.batch["response_mask"] = torch.tensor([[1, 1], [0, 0]])
    config = _config(response_source="last_assistant_response")

    attach_opsd_metadata(batch, config)
    teacher_batch = build_opsd_teacher_batch(batch, FakeTokenizer(), config)

    assert torch.equal(teacher_batch.batch["responses"], torch.tensor([[21, 22], [0, 0]]))
    assert torch.equal(teacher_batch.batch["response_mask"], torch.tensor([[1, 1], [0, 0]]))
    assert torch.equal(teacher_batch.batch[SELECTED_RESPONSE_MASK_KEY], torch.tensor([[1, 1], [0, 0]]))
    assert torch.equal(teacher_batch.batch[ORIGINAL_RESPONSE_MASK_KEY], batch.batch["response_mask"])


def test_build_teacher_batch_handles_all_empty_selected_last_assistant_response():
    batch = _batch(extra_info=[{"teacher_prompt": "A"}, {"teacher_prompt": "B"}])
    batch.batch["response_mask"] = torch.zeros((2, 2), dtype=torch.long)
    config = _config(response_source="last_assistant_response")

    attach_opsd_metadata(batch, config)
    teacher_batch = build_opsd_teacher_batch(batch, FakeTokenizer(), config)

    assert torch.equal(teacher_batch.batch["responses"], torch.zeros((2, 1), dtype=torch.long))
    assert torch.equal(teacher_batch.batch["response_mask"], torch.zeros((2, 1), dtype=torch.long))
    assert torch.equal(teacher_batch.batch[SELECTED_RESPONSE_MASK_KEY], torch.zeros((2, 1), dtype=torch.long))


def test_restore_selected_response_teacher_logprobs_to_original_response_shape():
    batch = _batch(extra_info=[{"teacher_prompt": "A"}, {"teacher_prompt": "B"}])
    batch.batch["responses"] = torch.tensor([[31, 32, 41, 42], [33, 43, 44, 0]])
    batch.batch["response_mask"] = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 0]])
    batch.batch["attention_mask"] = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 0]]
    )
    config = _config(response_source="last_assistant_response")

    attach_opsd_metadata(batch, config)
    teacher_input_batch = build_opsd_teacher_batch(batch, FakeTokenizer(), config)
    teacher_output = DataProto.from_dict(
        tensors={
            "ref_log_prob": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "teacher_logprobs": torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),
            "teacher_ids": torch.tensor([[[11], [12]], [[13], [14]]]),
        }
    )

    restored = restore_selected_response_teacher_logprobs(teacher_output, teacher_input_batch)

    assert torch.equal(
        restored.batch["ref_log_prob"],
        torch.tensor([[0.0, 0.0, 1.0, 2.0], [0.0, 3.0, 4.0, 0.0]]),
    )
    assert torch.equal(
        restored.batch["teacher_logprobs"],
        torch.tensor([[[0.0], [0.0], [1.0], [2.0]], [[0.0], [3.0], [4.0], [0.0]]]),
    )
    assert torch.equal(
        restored.batch["teacher_ids"],
        torch.tensor([[[0], [0], [11], [12]], [[0], [13], [14], [0]]]),
    )


def test_build_reward_model_teacher_batch_uses_online_gqm_outputs():
    batch = _batch()
    batch.non_tensor_batch[REWARD_MODEL_PROMPTS_KEY] = np.array(
        [{"prompt_token_ids": [1, 2, 3]}, {"prompt_token_ids": [4, 5, 6]}], dtype=object
    )
    batch.non_tensor_batch[REWARD_MODEL_RESPONSES_KEY] = np.array(
        ["GQM analysis A\nScore: 80", "GQM analysis B\nScore: 70"], dtype=object
    )
    tokenizer = FakeTokenizer()

    teacher_batch = build_reward_model_teacher_batch(batch, tokenizer, _config(teacher_max_prompt_length=512))

    assert torch.equal(teacher_batch.batch["responses"], batch.batch["responses"])
    assert torch.equal(teacher_batch.batch["response_mask"], batch.batch["response_mask"])
    assert teacher_batch.batch["input_ids"].shape == (2, 514)
    assert torch.all(teacher_batch.batch["attention_mask"][:, -2:] == 1)
    assert "Decoded original GQM prompt" in tokenizer.texts[0]
    assert "GQM analysis A" in tokenizer.texts[0]
    assert GQM_POST_EDIT_PROMPT in tokenizer.texts[0]
    assert tokenizer.texts[0].endswith("Assistant:")
    assert torch.equal(teacher_batch.batch[DISTILLATION_LOSS_MASK_KEY], torch.ones(2, 1, dtype=torch.long))
    assert teacher_batch.meta_info["distillation_loss_mask_valid_ratio"] == 1.0


def test_build_reward_model_teacher_batch_masks_missing_reward_model_outputs():
    batch = _batch()
    batch.non_tensor_batch[REWARD_MODEL_PROMPTS_KEY] = np.array(
        [{"prompt_token_ids": [1, 2, 3]}, {"prompt_token_ids": [4, 5, 6]}], dtype=object
    )
    batch.non_tensor_batch[REWARD_MODEL_RESPONSES_KEY] = np.array(
        ["GQM analysis A\nScore: 80", ""], dtype=object
    )
    tokenizer = FakeTokenizer()

    teacher_batch = build_reward_model_teacher_batch(batch, tokenizer, _config(teacher_max_prompt_length=512))

    assert torch.equal(teacher_batch.batch["responses"], batch.batch["responses"])
    assert torch.equal(teacher_batch.batch[DISTILLATION_LOSS_MASK_KEY], torch.tensor([[1], [0]]))
    assert teacher_batch.meta_info["distillation_loss_mask_valid_ratio"] == 0.5


def test_build_reward_model_teacher_batch_truncates_reward_model_response_first():
    batch = _batch()
    batch.non_tensor_batch[REWARD_MODEL_PROMPTS_KEY] = np.array(
        [{"prompt_token_ids": [1, 2, 3]}, {"prompt_token_ids": [4, 5, 6]}], dtype=object
    )
    batch.non_tensor_batch[REWARD_MODEL_RESPONSES_KEY] = np.array(
        ["GQM analysis A " + "x" * 200, "GQM analysis B " + "y" * 200], dtype=object
    )
    tokenizer = FakeTokenizer()
    config = _config(teacher_max_prompt_length=320)

    teacher_batch = build_reward_model_teacher_batch(batch, tokenizer, config)

    assert teacher_batch.batch["prompts"].shape[-1] == 320
    assert torch.equal(teacher_batch.batch[DISTILLATION_LOSS_MASK_KEY], torch.ones(2, 1, dtype=torch.long))
    assert "Decoded original GQM prompt" in tokenizer.texts[-2]
    assert GQM_POST_EDIT_PROMPT in tokenizer.texts[-2]
    assert "x" * 200 not in tokenizer.texts[-2]


def test_get_gen_batch_keeps_opsd_metadata_on_training_batch():
    trainer = object.__new__(RayPPOTrainer)
    trainer.async_rollout_mode = False
    trainer.use_distillation = True
    trainer.distillation_data_teacher_prompt = True
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(multi_turn=SimpleNamespace(shared_first_turn_by_uid=False))
        )
    )
    batch = _batch(
        extra_info=[
            {"teacher_prompt": "Full teacher prompt A"},
            {"teacher_prompt": "Full teacher prompt B"},
        ]
    )

    attach_opsd_metadata(batch, _config())
    gen_batch = trainer._get_gen_batch(batch)

    assert OPSD_TEACHER_PROMPT_KEY in batch.non_tensor_batch
    assert OPSD_TEACHER_PROMPT_KEY not in gen_batch.non_tensor_batch
