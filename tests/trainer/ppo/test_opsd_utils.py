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
    REWARD_MODEL_PROMPTS_KEY,
    REWARD_MODEL_RESPONSES_KEY,
    OPSD_TEACHER_PROMPT_KEY,
    attach_opsd_metadata,
    build_reward_model_teacher_batch,
    build_opsd_teacher_batch,
    extract_opsd_teacher_prompt,
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
        encoded = [[ord(char) % 251 + 1 for char in prompt] for prompt in prompts]
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
        return "Decoded original GQM prompt"


def _config(prompt_path="extra_info.teacher_prompt"):
    return OmegaConf.create(
        {
            "distillation": {
                "teacher": {
                    "prompt_source": "data_teacher_prompt",
                    "teacher_prompt_path": prompt_path,
                    "prompt_constructor_path": "reward_utils/prompts.py",
                    "prompt_constructor_name": "gqm_post_edit_teacher_prompt_constructor",
                }
            },
            "data": {
                "max_prompt_length": 64,
                "truncation": "right",
                "apply_chat_template_kwargs": {},
            },
        }
    )


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


def test_build_reward_model_teacher_batch_uses_online_gqm_outputs():
    batch = _batch()
    batch.non_tensor_batch[REWARD_MODEL_PROMPTS_KEY] = np.array(
        [{"prompt_token_ids": [1, 2, 3]}, {"prompt_token_ids": [4, 5, 6]}], dtype=object
    )
    batch.non_tensor_batch[REWARD_MODEL_RESPONSES_KEY] = np.array(
        ["GQM analysis A\nScore: 80", "GQM analysis B\nScore: 70"], dtype=object
    )
    tokenizer = FakeTokenizer()

    teacher_batch = build_reward_model_teacher_batch(batch, tokenizer, _config())

    assert torch.equal(teacher_batch.batch["responses"], batch.batch["responses"])
    assert torch.equal(teacher_batch.batch["response_mask"], batch.batch["response_mask"])
    assert teacher_batch.batch["input_ids"].shape == (2, 66)
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

    teacher_batch = build_reward_model_teacher_batch(batch, tokenizer, _config())

    assert torch.equal(teacher_batch.batch["responses"], batch.batch["responses"])
    assert torch.equal(teacher_batch.batch[DISTILLATION_LOSS_MASK_KEY], torch.tensor([[1], [0]]))
    assert teacher_batch.meta_info["distillation_loss_mask_valid_ratio"] == 0.5


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
