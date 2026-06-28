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
SELECTED_RESPONSE_MASK_KEY = "selected_response_mask"
ORIGINAL_RESPONSE_MASK_KEY = "original_response_mask"


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

    selected_responses = None
    selected_response_mask = None
    if _teacher_response_source(config) == "last_assistant_response":
        _validate_last_assistant_response_source(config)
        selected_responses, selected_response_mask = _select_masked_responses(batch)

    return _build_teacher_batch_from_prompts(
        batch,
        tokenizer,
        config,
        batch.non_tensor_batch[OPSD_TEACHER_PROMPT_KEY],
        responses=selected_responses,
        response_mask=selected_response_mask,
        selected_response_mask=selected_response_mask,
    )


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
            teacher_prompt = _truncate_reward_model_teacher_prompt_if_needed(teacher_prompt, tokenizer, config)
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


def _truncate_reward_model_teacher_prompt_if_needed(teacher_prompt: Any, tokenizer: Any, config: Any) -> Any:
    max_length = _teacher_max_prompt_length(config)
    if not isinstance(teacher_prompt, list) or len(teacher_prompt) < 3:
        return teacher_prompt
    if _prompt_token_length(teacher_prompt, tokenizer, config) <= max_length:
        return teacher_prompt

    assistant_index = _find_last_assistant_message_index(teacher_prompt)
    if assistant_index is None:
        return teacher_prompt

    truncated_prompt = deepcopy(teacher_prompt)
    assistant_message = dict(truncated_prompt[assistant_index])
    response_text = str(assistant_message.get("content", ""))
    response_ids = tokenizer([response_text], return_tensors="pt", padding=False, add_special_tokens=False)[
        "input_ids"
    ].squeeze(0)

    low = 0
    high = response_ids.numel()
    best_content = ""
    truncation = str(_select(config, "distillation.teacher.response_truncation", "middle"))
    while low <= high:
        keep = (low + high) // 2
        candidate = dict(assistant_message)
        candidate["content"] = _decode_truncated_ids(response_ids, keep, truncation, tokenizer)
        truncated_prompt[assistant_index] = candidate
        if _prompt_token_length(truncated_prompt, tokenizer, config) <= max_length:
            best_content = candidate["content"]
            low = keep + 1
        else:
            high = keep - 1

    assistant_message["content"] = best_content
    truncated_prompt[assistant_index] = assistant_message
    if _prompt_token_length(truncated_prompt, tokenizer, config) > max_length:
        raise ValueError("Reward model teacher prompt exceeds max_prompt_length after response truncation.")
    return truncated_prompt


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


def _prompt_token_length(teacher_prompt: Any, tokenizer: Any, config: Any) -> int:
    apply_chat_template_kwargs = _select(config, "data.apply_chat_template_kwargs", {}) or {}
    prompt_text = _render_teacher_prompt(teacher_prompt, tokenizer, apply_chat_template_kwargs)
    encoded = tokenizer([prompt_text], return_tensors="pt", padding=False, add_special_tokens=False)
    return int(encoded["attention_mask"].sum().item())


def _find_last_assistant_message_index(messages: list[Any]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, dict) and message.get("role") == "assistant":
            return index
    return None


def _decode_truncated_ids(token_ids: torch.Tensor, keep: int, truncation: str, tokenizer: Any) -> str:
    if keep <= 0:
        return ""
    keep = min(keep, token_ids.numel())
    if truncation == "left":
        kept_ids = token_ids[-keep:]
    elif truncation == "right":
        kept_ids = token_ids[:keep]
    elif truncation == "middle":
        left = keep // 2
        right = keep - left
        kept_ids = torch.cat([token_ids[:left], token_ids[-right:]], dim=0) if right else token_ids[:left]
    else:
        raise ValueError("distillation.teacher.response_truncation must be 'left', 'right', or 'middle'.")
    return tokenizer.decode(kept_ids.detach().cpu().tolist(), skip_special_tokens=False)


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
    responses: torch.Tensor | None = None,
    response_mask: torch.Tensor | None = None,
    selected_response_mask: torch.Tensor | None = None,
) -> DataProto:
    apply_chat_template_kwargs = _select(config, "data.apply_chat_template_kwargs", {}) or {}
    prompts = [
        _render_teacher_prompt(teacher_prompt, tokenizer, apply_chat_template_kwargs)
        for teacher_prompt in teacher_prompts
    ]

    prompt_ids, prompt_attention_mask = _tokenize_and_left_pad_prompts(
        tokenizer=tokenizer,
        prompts=prompts,
        max_length=_teacher_max_prompt_length(config),
        truncation=str(_select(config, "data.truncation", "error")),
    )

    device = batch.batch["responses"].device
    prompt_ids = prompt_ids.to(device)
    prompt_attention_mask = prompt_attention_mask.to(device)
    responses = batch.batch["responses"] if responses is None else responses
    response_length = responses.size(-1)
    if response_mask is None:
        response_attention_mask = batch.batch["attention_mask"][:, -response_length:].to(device)
    else:
        response_attention_mask = response_mask.to(device)

    input_ids = torch.cat([prompt_ids, responses], dim=-1)
    attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)
    position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)

    teacher_batch = deepcopy(batch)
    teacher_batch.batch["prompts"] = prompt_ids
    teacher_batch.batch["input_ids"] = input_ids
    teacher_batch.batch["attention_mask"] = attention_mask
    teacher_batch.batch["position_ids"] = position_ids
    teacher_batch.batch["responses"] = responses
    teacher_batch.batch["response_mask"] = response_attention_mask
    if selected_response_mask is not None:
        teacher_batch.batch[SELECTED_RESPONSE_MASK_KEY] = selected_response_mask.to(device)
        teacher_batch.batch[ORIGINAL_RESPONSE_MASK_KEY] = batch.batch["response_mask"].to(device)
    return teacher_batch


def restore_selected_response_teacher_logprobs(
    teacher_logprobs: DataProto,
    teacher_input_batch: DataProto,
) -> DataProto:
    """Restore teacher outputs from selected response length to the original rollout response length."""
    if SELECTED_RESPONSE_MASK_KEY not in teacher_input_batch.batch:
        return teacher_logprobs

    selected_response_mask = teacher_input_batch.batch[SELECTED_RESPONSE_MASK_KEY]
    original_response_mask = teacher_input_batch.batch.get(ORIGINAL_RESPONSE_MASK_KEY)
    if original_response_mask is None:
        raise ValueError("Selected-response teacher batch is missing original_response_mask.")

    selected_response_mask = selected_response_mask.to(torch.bool)
    original_response_mask = original_response_mask.to(torch.bool)
    if selected_response_mask.sum().item() != original_response_mask.sum().item():
        raise ValueError("Selected-response teacher output cannot be restored because token counts differ.")

    restored_tensors = {}
    for key, value in teacher_logprobs.batch.items():
        if key not in {"ref_log_prob", "teacher_logprobs", "teacher_ids"}:
            restored_tensors[key] = value
            continue
        restored_tensors[key] = _restore_selected_response_tensor(
            value,
            selected_response_mask.to(value.device),
            original_response_mask.to(value.device),
        )

    return DataProto.from_dict(tensors=restored_tensors, meta_info=teacher_logprobs.meta_info)


def _restore_selected_response_tensor(
    tensor: torch.Tensor,
    selected_response_mask: torch.Tensor,
    original_response_mask: torch.Tensor,
) -> torch.Tensor:
    restored_shape = (original_response_mask.shape[0], original_response_mask.shape[1], *tensor.shape[2:])
    restored = torch.zeros(restored_shape, dtype=tensor.dtype, device=tensor.device)
    restored[original_response_mask] = tensor[selected_response_mask]
    return restored


def _select_masked_responses(batch: DataProto) -> tuple[torch.Tensor, torch.Tensor]:
    if "response_mask" not in batch.batch:
        raise ValueError("distillation.teacher.response_source=last_assistant_response requires response_mask.")

    responses = batch.batch["responses"]
    response_mask = batch.batch["response_mask"].to(torch.bool)
    selected_lengths = response_mask.sum(dim=-1)
    if torch.any(selected_lengths == 0):
        raise ValueError("distillation.teacher.response_source=last_assistant_response selected an empty response.")

    max_selected_length = int(selected_lengths.max().item())
    selected_responses = responses.new_full(
        (responses.shape[0], max_selected_length),
        _select_pad_token_id(batch),
    )
    selected_response_mask = response_mask.new_zeros(
        (responses.shape[0], max_selected_length),
        dtype=batch.batch["response_mask"].dtype,
    )
    for index, length in enumerate(selected_lengths.tolist()):
        selected_tokens = responses[index][response_mask[index]]
        selected_responses[index, :length] = selected_tokens
        selected_response_mask[index, :length] = 1
    return selected_responses, selected_response_mask


def _select_pad_token_id(batch: DataProto) -> int:
    if "pad_token_id" in batch.meta_info:
        return int(batch.meta_info["pad_token_id"])
    return 0


def _teacher_response_source(config: Any) -> str:
    return str(_select(config, "distillation.teacher.response_source", "full_response"))


def _validate_last_assistant_response_source(config: Any) -> None:
    if _select(config, "distillation.teacher.source", "ref_policy") != "ref_policy":
        raise ValueError(
            "distillation.teacher.response_source=last_assistant_response requires "
            "distillation.teacher.source=ref_policy."
        )
    if _select(config, "distillation.teacher.prompt_source", "actor_prompt") != "data_teacher_prompt":
        raise ValueError(
            "distillation.teacher.response_source=last_assistant_response requires "
            "distillation.teacher.prompt_source=data_teacher_prompt."
        )
    if _select(config, "actor_rollout_ref.rollout.multi_turn.response_mask_mode", None) != "last_assistant":
        raise ValueError(
            "distillation.teacher.response_source=last_assistant_response requires "
            "actor_rollout_ref.rollout.multi_turn.response_mask_mode=last_assistant."
        )


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


def _teacher_max_prompt_length(config: Any) -> int:
    return int(
        _select(
            config,
            "distillation.teacher.max_prompt_length",
            _select(config, "data.max_prompt_length"),
        )
    )


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
