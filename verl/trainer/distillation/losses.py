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

from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty


def is_distillation_enabled(config) -> bool:
    return config is not None and bool(config.get("enabled", False))


def validate_distillation_config(config) -> None:
    distillation_config = config.get("distillation")
    if not is_distillation_enabled(distillation_config):
        return

    teacher_source = distillation_config.teacher.source
    loss_mode = distillation_config.distillation_loss.loss_mode
    if teacher_source == "reward_model":
        raise NotImplementedError(
            "distillation.teacher.source=reward_model is reserved for future GenRM-as-teacher support "
            "and is not implemented yet."
        )
    if loss_mode == "forward_kl_topk":
        raise NotImplementedError(
            "distillation.distillation_loss.loss_mode=forward_kl_topk is not supported in this first OPD "
            "implementation. Use sampled-token KL modes such as k3, low_var_kl, k1, kl, abs, mse, or k2."
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


def combine_policy_and_distillation_loss(policy_loss, distillation_loss, distillation_config):
    loss_config = distillation_config.distillation_loss
    if not loss_config.use_task_rewards:
        return distillation_loss
    return policy_loss + distillation_loss * loss_config.distillation_loss_coef
