from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from verl import DataProto
from verl.utils.import_utils import load_extern_type
from verl.utils.torch_functional import postprocess_data


OPSD_TEACHER_PROMPT_KEY = "opsd_teacher_prompt"
REWARD_MODEL_PROMPTS_KEY = "reward_model_prompts"
REWARD_MODEL_RESPONSES_KEY = "reward_model_responses"
DISTILLATION_LOSS_MASK_KEY = "distillation_loss_mask"


def distillation_uses_data_teacher_prompt(config: Any) -> bool:
    return _select(config, "distillation.teacher.prompt_source", "actor_prompt") == "data_teacher_prompt"


def distillation_uses_reward_model_prompt(config: Any) -> bool:
    return _select(config, "distillation.teacher.prompt_source", "actor_prompt") == "reward_model"


def attach_opsd_metadata(batch: DataProto, config: Any) -> None:
    """Copy the configured OPSD teacher prompt into a stable non-tensor key."""
    if not distillation_uses_data_teacher_prompt(config):
        return

    prompt_path = str(_select(config, "distillation.teacher.teacher_prompt_path", "extra_info.teacher_prompt"))
    teacher_prompts = []
    batch_size = len(batch.batch["input_ids"])
    for index in range(batch_size):
        sample = _sample_dict_from_dataproto(batch, index)
        teacher_prompt = extract_opsd_teacher_prompt(sample, prompt_path)
        if not _has_prompt_value(teacher_prompt):
            raise ValueError(
                f"OPSD teacher prompt is missing or empty for sample index {index}. "
                f"Tried path: {prompt_path}"
            )
        teacher_prompts.append(teacher_prompt)

    batch.non_tensor_batch[OPSD_TEACHER_PROMPT_KEY] = np.array(teacher_prompts, dtype=object)


def extract_opsd_teacher_prompt(sample: dict[str, Any], prompt_path: str) -> Any | None:
    return _get_path(sample, prompt_path)


def build_opsd_teacher_batch(batch: DataProto, tokenizer: Any, config: Any) -> DataProto:
    """Return a copy of batch whose prompt context is the data-provided teacher prompt."""
    if OPSD_TEACHER_PROMPT_KEY not in batch.non_tensor_batch:
        raise ValueError("OPSD metadata is missing from rollout batch")

    return _build_teacher_batch_from_prompts(batch, tokenizer, config, batch.non_tensor_batch[OPSD_TEACHER_PROMPT_KEY])


def build_reward_model_teacher_batch(batch: DataProto, tokenizer: Any, config: Any) -> DataProto:
    """Return a teacher batch prompted by reward-model metadata."""
    if REWARD_MODEL_PROMPTS_KEY not in batch.non_tensor_batch:
        raise ValueError(
            "Reward model prompts are missing from rollout batch. "
            "Expected reward model metadata key: reward_model_prompts"
        )
    if REWARD_MODEL_RESPONSES_KEY not in batch.non_tensor_batch:
        raise ValueError(
            "Reward model responses are missing from rollout batch. "
            "Expected reward model metadata key: reward_model_responses"
        )

    constructor = _load_reward_model_prompt_constructor(config)
    constructor_kwargs = dict(_select(config, "distillation.teacher.prompt_constructor_kwargs", {}) or {})
    teacher_prompts = []
    distillation_loss_mask = []
    for index, (reward_model_prompt, reward_model_response) in enumerate(
        zip(
            batch.non_tensor_batch[REWARD_MODEL_PROMPTS_KEY],
            batch.non_tensor_batch[REWARD_MODEL_RESPONSES_KEY],
            strict=True,
        )
    ):
        try:
            teacher_prompt = constructor(
                prompt=reward_model_prompt,
                response=reward_model_response,
                tokenizer=tokenizer,
                sample=_sample_dict_from_dataproto(batch, index),
                config=config,
                **constructor_kwargs,
            )
            distillation_loss_mask.append(1)
        except ValueError:
            teacher_prompt = _actor_prompt_from_batch(batch, tokenizer, index)
            distillation_loss_mask.append(0)
        teacher_prompts.append(teacher_prompt)

    teacher_batch = _build_teacher_batch_from_prompts(batch, tokenizer, config, teacher_prompts)
    teacher_batch.batch[DISTILLATION_LOSS_MASK_KEY] = torch.tensor(
        distillation_loss_mask,
        dtype=teacher_batch.batch["response_mask"].dtype,
        device=teacher_batch.batch["response_mask"].device,
    ).unsqueeze(-1)
    teacher_batch.meta_info["distillation_loss_mask_valid_ratio"] = (
        float(sum(distillation_loss_mask)) / len(distillation_loss_mask) if distillation_loss_mask else 0.0
    )
    return teacher_batch


def _actor_prompt_from_batch(batch: DataProto, tokenizer: Any, index: int) -> str:
    prompt_ids = batch.batch["prompts"][index]
    prompt_attention_mask = batch.batch["attention_mask"][index, : prompt_ids.shape[-1]].to(bool)
    return tokenizer.decode(
        prompt_ids[prompt_attention_mask].detach().cpu().tolist(),
        skip_special_tokens=False,
    )


def _sample_dict_from_dataproto(batch: DataProto, index: int) -> dict[str, Any]:
    sample: dict[str, Any] = {}
    for key, value in batch.non_tensor_batch.items():
        sample[key] = value[index]
    return sample


def _render_teacher_prompt(teacher_prompt: Any, tokenizer: Any, apply_chat_template_kwargs: dict[str, Any]) -> str:
    if isinstance(teacher_prompt, np.ndarray):
        teacher_prompt = teacher_prompt.tolist()
    if isinstance(teacher_prompt, (list, tuple)):
        return tokenizer.apply_chat_template(
            list(teacher_prompt),
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs,
        )
    return str(teacher_prompt)


def _load_reward_model_prompt_constructor(config: Any):
    path = _select(config, "distillation.teacher.prompt_constructor_path", None)
    name = _select(config, "distillation.teacher.prompt_constructor_name", None)
    if not path or not name:
        raise ValueError(
            "distillation.teacher.prompt_source=reward_model requires "
            "distillation.teacher.prompt_constructor_path and prompt_constructor_name."
        )
    return load_extern_type(str(path), str(name))


def _build_teacher_batch_from_prompts(
    batch: DataProto,
    tokenizer: Any,
    config: Any,
    teacher_prompts: list[Any],
) -> DataProto:
    apply_chat_template_kwargs = _select(config, "data.apply_chat_template_kwargs", {}) or {}
    prompts = [
        _render_teacher_prompt(teacher_prompt, tokenizer, apply_chat_template_kwargs)
        for teacher_prompt in teacher_prompts
    ]

    prompt_ids, prompt_attention_mask = _tokenize_and_left_pad_prompts(
        tokenizer=tokenizer,
        prompts=prompts,
        max_length=int(_select(config, "data.max_prompt_length")),
        truncation=str(_select(config, "data.truncation", "error")),
    )

    device = batch.batch["responses"].device
    prompt_ids = prompt_ids.to(device)
    prompt_attention_mask = prompt_attention_mask.to(device)
    responses = batch.batch["responses"]
    response_length = responses.size(-1)
    response_attention_mask = batch.batch["attention_mask"][:, -response_length:].to(device)

    input_ids = torch.cat([prompt_ids, responses], dim=-1)
    attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)
    position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)

    teacher_batch = deepcopy(batch)
    teacher_batch.batch["prompts"] = prompt_ids
    teacher_batch.batch["input_ids"] = input_ids
    teacher_batch.batch["attention_mask"] = attention_mask
    teacher_batch.batch["position_ids"] = position_ids
    return teacher_batch


def _tokenize_and_left_pad_prompts(
    tokenizer: Any,
    prompts: list[str],
    max_length: int,
    truncation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = []
    attention_mask = []
    for prompt in prompts:
        encoded = tokenizer([prompt], return_tensors="pt", padding=False, add_special_tokens=False)
        prompt_ids, prompt_attention_mask = postprocess_data(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id,
            left_pad=True,
            truncation=truncation,
        )
        input_ids.append(prompt_ids.squeeze(0))
        attention_mask.append(prompt_attention_mask.squeeze(0))

    return torch.stack(input_ids, dim=0), torch.stack(attention_mask, dim=0)


def _has_prompt_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, np.ndarray):
        return value.size > 0
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    return True


def _get_path(data: Any, path: str) -> Any:
    cur = data
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _select(config: Any, path: str, default: Any = None) -> Any:
    if isinstance(config, (DictConfig, ListConfig)):
        value = OmegaConf.select(config, path)
        return default if value is None else value
    value = _get_path(config, path)
    return default if value is None else value
