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

from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig

__all__ = ["DistillationLossConfig", "DistillationTeacherConfig", "DistillationConfig"]


@dataclass
class DistillationTeacherConfig(BaseConfig):
    """Configuration for the OPD teacher logits provider."""

    source: str = "ref_policy"
    ref_model_path: Optional[str] = None
    prompt_source: str = "actor_prompt"
    teacher_prompt_path: str = "extra_info.teacher_prompt"
    prompt_constructor_path: Optional[str] = None
    prompt_constructor_name: Optional[str] = None
    prompt_constructor_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.source not in {"ref_policy", "current_policy", "reward_model"}:
            raise ValueError(
                f"Unsupported distillation.teacher.source={self.source!r}. "
                "Supported values are 'ref_policy', 'current_policy', and 'reward_model'."
            )
        if self.prompt_source not in {"actor_prompt", "data_teacher_prompt", "reward_model"}:
            raise ValueError(
                f"Unsupported distillation.teacher.prompt_source={self.prompt_source!r}. "
                "Supported values are 'actor_prompt', 'data_teacher_prompt', and 'reward_model'."
            )


@dataclass
class DistillationLossConfig(BaseConfig):
    """Configuration for on-policy distillation losses."""

    loss_mode: str = "k3"
    topk: Optional[int] = 32
    use_task_rewards: bool = True
    distillation_loss_coef: float = 1.0
    loss_max_clamp: Optional[float] = None
    log_prob_min_clamp: Optional[float] = None
    use_policy_gradient: bool = False
    policy_loss_mode: str = "vanilla"
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    topk_kl_mode: str = "forward"
    norm_to_one_for_kl: bool = False
    use_chunked_topk: bool = True
    chunked_topk_chunk_size: int = 4096
    global_batch_info: dict = field(default_factory=dict)

    def __post_init__(self):
        supported = {"kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3", "forward_kl_topk"}
        if self.loss_mode not in supported:
            raise ValueError(f"Unsupported distillation loss_mode={self.loss_mode!r}. Supported modes: {sorted(supported)}")
        if self.loss_mode == "forward_kl_topk" and self.topk is None:
            raise ValueError("distillation.distillation_loss.topk must be set when loss_mode=forward_kl_topk.")
        if self.topk_kl_mode not in {"forward", "reverse"}:
            raise ValueError("distillation.distillation_loss.topk_kl_mode must be 'forward' or 'reverse'.")
        if self.policy_loss_mode != "vanilla":
            raise NotImplementedError(
                "Only distillation.distillation_loss.policy_loss_mode=vanilla is supported when "
                "use_policy_gradient=True."
            )
        if self.loss_mode == "forward_kl_topk" and self.use_policy_gradient:
            raise NotImplementedError(
                "distillation.distillation_loss.loss_mode=forward_kl_topk currently supports direct supervised "
                "distillation only. Set use_policy_gradient=False."
            )
        if not self.use_policy_gradient and self.loss_mode == "k1":
            raise ValueError(
                "Directly backpropagating distillation loss_mode=k1 is incorrect because its gradient does not "
                "depend on teacher log probabilities. Set use_policy_gradient=True or use k3/low_var_kl."
            )


@dataclass
class DistillationConfig(BaseConfig):
    """Configuration for on-policy distillation."""

    enabled: bool = False
    teacher: DistillationTeacherConfig = field(default_factory=DistillationTeacherConfig)
    distillation_loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)
