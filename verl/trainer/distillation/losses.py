# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
import torch.nn.functional as F

from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty


def is_distillation_enabled(config) -> bool:
    return config is not None and bool(config.get("enabled", False))


def validate_distillation_config(config) -> None:
    distillation_config = config.get("distillation")
    if not is_distillation_enabled(distillation_config):
        return

    teacher_source = distillation_config.teacher.source
    loss_mode = distillation_config.distillation_loss.loss_mode
    actor_strategy = config.actor_rollout_ref.actor.strategy
    if teacher_source == "reward_model":
        raise NotImplementedError(
            "distillation.teacher.source=reward_model is reserved for future GenRM-as-teacher support "
            "and is not implemented yet."
        )
    if loss_mode == "forward_kl_topk":
        if actor_strategy not in {"fsdp", "fsdp2"}:
            raise NotImplementedError("distillation loss_mode=forward_kl_topk is implemented for FSDP only.")
        if config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1:
            raise NotImplementedError(
                "distillation loss_mode=forward_kl_topk does not yet support actor Ulysses sequence parallelism."
            )
    if teacher_source == "ref_policy" and (
        config.actor_rollout_ref.actor.use_kl_loss or config.algorithm.use_kl_in_reward
    ):
        raise ValueError(
            "Invalid OPD configuration: distillation.teacher.source=ref_policy reuses actor_rollout_ref.ref as the "
            "teacher model, but actor_rollout_ref.actor.use_kl_loss or algorithm.use_kl_in_reward is enabled. "
            "That would mix teacher distillation semantics with PPO reference-KL semantics. Disable reference KL "
            "or use distillation.teacher.source=current_policy."
        )
    if teacher_source == "ref_policy" and (
        config.actor_rollout_ref.model.get("lora_rank", 0) > 0
        or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
    ):
        raise ValueError(
            "Invalid OPD configuration: distillation.teacher.source=ref_policy is not supported with LoRA in this "
            "first implementation. The existing LoRA reference path treats the actor without adapters as the "
            "reference policy, which is not a standalone OPD teacher. Use distillation.teacher.source=current_policy "
            "or disable LoRA for ref-policy teacher OPD."
        )


def compute_sampled_distillation_loss(
    config,
    distillation_config,
    log_prob: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    old_log_prob: Optional[torch.Tensor] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
):
    """Compute sampled-token OPD loss and metrics for padded tensors."""
    loss_config = distillation_config.distillation_loss
    response_mask = response_mask.to(bool)
    teacher_logprobs = teacher_logprobs.to(log_prob.device)
    if loss_config.log_prob_min_clamp is not None:
        teacher_logprobs = teacher_logprobs.clamp_min(loss_config.log_prob_min_clamp)
        log_prob_for_kl = log_prob.clamp_min(loss_config.log_prob_min_clamp)
    else:
        log_prob_for_kl = log_prob

    distillation_losses = kl_penalty(
        logprob=log_prob_for_kl,
        ref_logprob=teacher_logprobs,
        kl_penalty=loss_config.loss_mode,
    )
    valid_losses = distillation_losses[response_mask]
    metrics = {
        "distillation/abs_loss": valid_losses.abs().mean().detach().item(),
        "distillation/loss_min": valid_losses.min().detach().item(),
        "distillation/loss_max": valid_losses.max().detach().item(),
    }

    if loss_config.loss_max_clamp is not None:
        distillation_losses = distillation_losses.clamp(
            min=-loss_config.loss_max_clamp,
            max=loss_config.loss_max_clamp,
        )

    if loss_config.use_policy_gradient:
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        if old_log_prob is None:
            old_log_prob = log_prob.detach()
        distillation_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=-distillation_losses.detach(),
            response_mask=response_mask,
            loss_agg_mode=config.loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        metrics.update(
            {
                "distillation/pg_loss": distillation_loss.detach().item(),
                "distillation/pg_clipfrac": pg_clipfrac.detach().item(),
                "distillation/ppo_kl": ppo_kl.detach().item(),
                "distillation/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
            }
        )
    else:
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask,
            loss_agg_mode=config.loss_agg_mode,
        )

    metrics["distillation/loss"] = distillation_loss.detach().item()
    return distillation_loss, metrics


def is_forward_kl_topk_enabled(distillation_config) -> bool:
    return is_distillation_enabled(distillation_config) and distillation_config.distillation_loss.loss_mode == "forward_kl_topk"


def compute_topk_logprobs_from_logits(
    logits: torch.Tensor,
    topk: int,
    use_chunked_topk: bool = False,
    chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k log probabilities and ids for logits shaped [..., vocab]."""
    topk_logits, topk_ids = torch.topk(logits, k=topk, dim=-1)
    if use_chunked_topk:
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_topk_logits = topk_logits.reshape(-1, topk)
        flat_topk_logprobs = torch.empty_like(flat_topk_logits)
        for start in range(0, flat_logits.shape[0], chunk_size):
            end = min(start + chunk_size, flat_logits.shape[0])
            log_z = torch.logsumexp(flat_logits[start:end].float(), dim=-1, keepdim=True)
            flat_topk_logprobs[start:end] = (flat_topk_logits[start:end].float() - log_z).to(logits.dtype)
        topk_logprobs = flat_topk_logprobs.reshape(topk_logits.shape)
    else:
        topk_logprobs = F.log_softmax(logits, dim=-1).gather(dim=-1, index=topk_ids)
    return topk_logprobs, topk_ids


def gather_logprobs_at_ids(
    logits: torch.Tensor,
    ids: torch.Tensor,
    use_chunked_topk: bool = False,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Compute log_softmax(logits).gather(ids) without requiring caller to materialize logprobs."""
    if use_chunked_topk:
        vocab_size = logits.shape[-1]
        topk = ids.shape[-1]
        flat_logits = logits.reshape(-1, vocab_size)
        flat_ids = ids.reshape(-1, topk)
        flat_out = torch.empty(flat_ids.shape, dtype=logits.dtype, device=logits.device)
        for start in range(0, flat_logits.shape[0], chunk_size):
            end = min(start + chunk_size, flat_logits.shape[0])
            chunk_logits = flat_logits[start:end].float()
            log_z = torch.logsumexp(chunk_logits, dim=-1, keepdim=True)
            gathered = torch.gather(chunk_logits, dim=-1, index=flat_ids[start:end])
            flat_out[start:end] = (gathered - log_z).to(logits.dtype)
        return flat_out.reshape(ids.shape)
    return F.log_softmax(logits, dim=-1).gather(dim=-1, index=ids)


def compute_topk_kl_losses(
    teacher_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
    topk_kl_mode: str,
) -> torch.Tensor:
    """Compute per-token KL on the teacher top-k support."""
    teacher_logprobs = teacher_logprobs.float()
    student_logprobs = student_logprobs.float()
    if topk_kl_mode == "forward":
        return (teacher_logprobs.exp() * (teacher_logprobs - student_logprobs)).sum(dim=-1)
    if topk_kl_mode == "reverse":
        return (student_logprobs.exp() * (student_logprobs - teacher_logprobs)).sum(dim=-1)
    raise ValueError(f"Unsupported distillation.distillation_loss.topk_kl_mode={topk_kl_mode!r}.")


def compute_forward_kl_topk_distillation_loss(
    config,
    distillation_config,
    student_logits: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    teacher_ids: torch.Tensor,
    response_mask: torch.Tensor,
):
    """Compute FSDP top-k forward KL OPD loss for padded response tensors."""
    loss_config = distillation_config.distillation_loss
    response_mask = response_mask.to(bool)
    teacher_logprobs = teacher_logprobs.to(student_logits.device)
    teacher_ids = teacher_ids.to(student_logits.device)

    student_topk_logprobs = gather_logprobs_at_ids(
        logits=student_logits,
        ids=teacher_ids,
        use_chunked_topk=loss_config.use_chunked_topk,
        chunk_size=loss_config.chunked_topk_chunk_size,
    )
    teacher_for_loss = teacher_logprobs
    student_for_loss = student_topk_logprobs
    if loss_config.log_prob_min_clamp is not None:
        teacher_for_loss = teacher_for_loss.clamp_min(loss_config.log_prob_min_clamp)
        student_for_loss = student_for_loss.clamp_min(loss_config.log_prob_min_clamp)

    distillation_losses = compute_topk_kl_losses(
        teacher_logprobs=teacher_for_loss,
        student_logprobs=student_for_loss,
        topk_kl_mode=loss_config.topk_kl_mode,
    )
    distillation_losses = distillation_losses.clamp_min(0.0)
    valid_losses = distillation_losses[response_mask]

    if loss_config.loss_max_clamp is not None:
        distillation_losses = distillation_losses.clamp(max=loss_config.loss_max_clamp)

    distillation_loss = agg_loss(
        loss_mat=distillation_losses,
        loss_mask=response_mask,
        loss_agg_mode=config.loss_agg_mode,
    )

    student_mass = student_topk_logprobs.float().exp().sum(dim=-1)
    teacher_mass = teacher_logprobs.float().exp().sum(dim=-1)
    valid_student_mass = student_mass[response_mask]
    valid_teacher_mass = teacher_mass[response_mask]

    metrics = {
        "distillation/loss": distillation_loss.detach().item(),
        "distillation/loss_min": valid_losses.min().detach().item(),
        "distillation/loss_max": valid_losses.max().detach().item(),
        "distillation/student_mass": valid_student_mass.mean().detach().item(),
        "distillation/teacher_mass": valid_teacher_mass.mean().detach().item(),
    }
    return distillation_loss, metrics


def compute_forward_kl_topk_distillation_loss_flat(
    config,
    distillation_config,
    student_logits: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    teacher_ids: torch.Tensor,
    response_mask: torch.Tensor,
):
    """Compute top-k forward KL from flat valid response logits without building [B, T, V]."""
    loss_config = distillation_config.distillation_loss
    response_mask = response_mask.to(bool)
    valid_teacher_logprobs = teacher_logprobs.to(student_logits.device)[response_mask]
    valid_teacher_ids = teacher_ids.to(student_logits.device)[response_mask]
    if student_logits.shape[0] != valid_teacher_ids.shape[0]:
        raise ValueError(
            "Flat forward_kl_topk inputs are misaligned: "
            f"student_logits has {student_logits.shape[0]} rows but response_mask selects "
            f"{valid_teacher_ids.shape[0]} teacher rows."
        )

    valid_student_logprobs = gather_logprobs_at_ids(
        logits=student_logits,
        ids=valid_teacher_ids,
        use_chunked_topk=True,
        chunk_size=loss_config.chunked_topk_chunk_size,
    )
    teacher_for_loss = valid_teacher_logprobs
    student_for_loss = valid_student_logprobs
    if loss_config.log_prob_min_clamp is not None:
        teacher_for_loss = teacher_for_loss.clamp_min(loss_config.log_prob_min_clamp)
        student_for_loss = student_for_loss.clamp_min(loss_config.log_prob_min_clamp)

    valid_losses = compute_topk_kl_losses(
        teacher_logprobs=teacher_for_loss,
        student_logprobs=student_for_loss,
        topk_kl_mode=loss_config.topk_kl_mode,
    )
    valid_losses = valid_losses.clamp_min(0.0)
    if loss_config.loss_max_clamp is not None:
        valid_losses = valid_losses.clamp(max=loss_config.loss_max_clamp)

    distillation_losses = torch.zeros(response_mask.shape, dtype=valid_losses.dtype, device=valid_losses.device)
    distillation_losses[response_mask] = valid_losses
    distillation_loss = agg_loss(
        loss_mat=distillation_losses,
        loss_mask=response_mask,
        loss_agg_mode=config.loss_agg_mode,
    )

    valid_student_mass = valid_student_logprobs.float().exp().sum(dim=-1)
    valid_teacher_mass = valid_teacher_logprobs.float().exp().sum(dim=-1)
    metrics = {
        "distillation/loss": distillation_loss.detach().item(),
        "distillation/loss_min": valid_losses.min().detach().item(),
        "distillation/loss_max": valid_losses.max().detach().item(),
        "distillation/student_mass": valid_student_mass.mean().detach().item(),
        "distillation/teacher_mass": valid_teacher_mass.mean().detach().item(),
    }
    return distillation_loss, metrics


def combine_policy_and_distillation_loss(policy_loss, distillation_loss, distillation_config):
    loss_config = distillation_config.distillation_loss
    if not loss_config.use_task_rewards:
        return distillation_loss
    return policy_loss + distillation_loss * loss_config.distillation_loss_coef
