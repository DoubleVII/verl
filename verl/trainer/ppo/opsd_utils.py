from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from verl import DataProto
from verl.utils.torch_functional import postprocess_data


OPSD_TEACHER_PROMPT_KEY = "opsd_teacher_prompt"


def distillation_uses_data_teacher_prompt(config: Any) -> bool:
    return _select(config, "distillation.teacher.prompt_source", "actor_prompt") == "data_teacher_prompt"


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

    apply_chat_template_kwargs = _select(config, "data.apply_chat_template_kwargs", {}) or {}
    prompts = [
        _render_teacher_prompt(teacher_prompt, tokenizer, apply_chat_template_kwargs)
        for teacher_prompt in batch.non_tensor_batch[OPSD_TEACHER_PROMPT_KEY]
    ]

    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    prompt_ids, prompt_attention_mask = postprocess_data(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        max_length=int(_select(config, "data.max_prompt_length")),
        pad_token_id=tokenizer.pad_token_id,
        left_pad=True,
        truncation=str(_select(config, "data.truncation", "error")),
    )

    device = batch.batch["responses"].device
    prompt_ids = prompt_ids.to(device)
    prompt_attention_mask = prompt_attention_mask.to(device)
    responses = batch.batch["responses"]
    response_length = responses.size(-1)
    response_attention_mask = batch.batch.get("response_mask", batch.batch["attention_mask"][:, -response_length:]).to(
        device=device,
        dtype=prompt_attention_mask.dtype,
    )

    input_ids = torch.cat([prompt_ids, responses], dim=-1)
    attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)
    position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)

    teacher_batch = deepcopy(batch)
    teacher_batch.batch["prompts"] = prompt_ids
    teacher_batch.batch["input_ids"] = input_ids
    teacher_batch.batch["attention_mask"] = attention_mask
    teacher_batch.batch["position_ids"] = position_ids
    return teacher_batch


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
