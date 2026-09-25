"""Composable multi-task reward routing.

This module intentionally contains only the small, independently selectable
task handlers used by :class:`MultiTaskRewardProcessor`.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

from verl.utils.import_utils import load_extern_type

try:
    from .fused_parsers import (
        _is_valid_fused_candidate_count,
        _parse_fused_flash_gqm_response,
        _token_myers_diversity,
        _has_normalized_duplicate,
    )
    from .helpers import _compute_overlong_penalty, _decode_response
except ImportError:
    from reward_utils.fused_parsers import (
        _is_valid_fused_candidate_count,
        _parse_fused_flash_gqm_response,
        _token_myers_diversity,
        _has_normalized_duplicate,
    )
    from reward_utils.helpers import _compute_overlong_penalty, _decode_response


class RewardTaskHandler(Protocol):
    """Interface implemented by one ability-specific reward task."""

    def score(self, data, indices: List[int], generate_fn) -> Dict[int, float]:
        ...


@dataclass
class RewardTaskContext:
    config: Any
    tokenizer: Any
    input_tokenizer: Any


def _cfg(config, name: str, default):
    return getattr(config, name, default)


def _custom(config) -> dict:
    return getattr(config, "custom_processor", {}) or {}


def _load_ranking_reward_fn(config, reward_fn=None):
    custom = _custom(config)
    reward_config = custom.get("ranking_reward_fn")
    kwargs = {}
    if reward_fn is not None:
        if reward_config:
            kwargs = dict(getattr(reward_config, "reward_kwargs", None) or
                          (reward_config.get("reward_kwargs", {}) if isinstance(reward_config, dict) else {}))
        return reward_fn, kwargs
    if reward_config:
        path = reward_config.get("path") if isinstance(reward_config, dict) else getattr(reward_config, "path", None)
        name = reward_config.get("name") if isinstance(reward_config, dict) else getattr(reward_config, "name", None)
        if not path or not name:
            raise ValueError("custom_processor.ranking_reward_fn requires both 'path' and 'name'")
        kwargs = dict(reward_config.get("reward_kwargs", {}) if isinstance(reward_config, dict) else getattr(reward_config, "reward_kwargs", {}) or {})
        return load_extern_type(str(path), str(name)), kwargs
    try:
        from .ranking_score_reward import ranking_score_reward_fn
    except ImportError:
        from reward_utils.ranking_score_reward import ranking_score_reward_fn
    return ranking_score_reward_fn, kwargs


class RankingTaskHandler:
    def __init__(self, context: RewardTaskContext, reward_fn=None):
        self.context = context
        config = context.config
        self.default_reward = _cfg(config, "default_reward", 0.0)
        scale = _cfg(config, "score_scale_factor", 1.0)
        self.score_scale_factor = _cfg(config, "ranking_score_scale_factor", scale)
        self.reward_fn, self.reward_kwargs = _load_ranking_reward_fn(config, reward_fn)

    def score(self, data, indices: List[int], generate_fn=None) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        if not indices:
            return scores
        for idx in indices:
            response_ids = data.batch["responses"][idx]
            response_length = response_ids.shape[-1]
            valid_length = data.batch["attention_mask"][idx][-response_length:].sum()
            solution = self.context.input_tokenizer.decode(
                response_ids[:valid_length], skip_special_tokens=True
            ).replace(self.context.input_tokenizer.eos_token, "")

            reward_model_data = data.non_tensor_batch.get("reward_model")
            ground_truth = None
            if isinstance(reward_model_data, dict):
                ground_truth = reward_model_data.get("ground_truth")
            elif reward_model_data is not None:
                try:
                    value = reward_model_data[idx]
                    ground_truth = value.get("ground_truth") if isinstance(value, dict) else value
                except (IndexError, KeyError, TypeError):
                    pass
            if ground_truth is None:
                scores[idx] = self.default_reward
                continue

            data_source = data.non_tensor_batch.get("data_source", [""] * (idx + 1))
            if isinstance(data_source, (list, tuple)):
                data_source = data_source[idx] if idx < len(data_source) else ""
            extra = data.non_tensor_batch.get("extra_info")
            try:
                extra = extra[idx] if extra is not None else None
            except (IndexError, KeyError, TypeError):
                extra = None
            call_kwargs = {
                "data_source": data_source,
                "solution_str": solution,
                "ground_truth": ground_truth,
                "extra_info": extra,
                "score_scale_factor": self.score_scale_factor,
                **self.reward_kwargs,
            }
            result = self.reward_fn(**call_kwargs)
            if isinstance(result, dict):
                scores[idx] = result.get("score", self.default_reward)
            elif isinstance(result, (int, float)):
                scores[idx] = float(result)
            else:
                raise TypeError(
                    "ranking reward function must return a score number or a dict containing 'score', "
                    f"got {type(result).__name__}"
                )
        return scores


class FusedFlashGQMTaskHandler:
    def __init__(self, context: RewardTaskContext):
        self.context = context
        config = context.config
        custom = _custom(config)
        self.max_prompt_length = _cfg(config, "prompt_length", 1 << 20)
        self.prompt_type = _cfg(config, "group_prompt_type", "ranking_score")
        self.add_example = _cfg(config, "group_add_example", False)
        self.score_scale_factor = _cfg(config, "score_scale_factor", 0.1)
        self.default_reward = _cfg(config, "default_reward", 0.0)
        self.enable_language_detection = custom.get("enable_language_detection", False)
        self.overlong_buffer_cfg = custom.get("overlong_buffer", None)
        self.rm_max_candidates = _cfg(config, "rm_max_candidates", 4)
        self.diversity_algorithm = custom.get("diversity_algorithm", "none")
        self.diversity_penalty_weight = float(custom.get("diversity_penalty_weight", 1.0))
        self.diversity_penalty_clip = float(custom.get("diversity_penalty_clip", 0.0))
        max_penalty = custom.get("diversity_penalty_max", None)
        self.diversity_penalty_max = None if max_penalty is None else float(max_penalty)
        if self.diversity_algorithm not in {"none", "exact_match", "token_myers"}:
            raise ValueError(f"unsupported diversity_algorithm: {self.diversity_algorithm!r}")

    def _diversity_penalty(self, candidates):
        duplicate = _has_normalized_duplicate(candidates)
        if self.diversity_algorithm == "none":
            penalty = 0.0
        elif self.diversity_algorithm == "exact_match":
            penalty = self.diversity_penalty_weight if duplicate else 0.0
        else:
            diversity = _token_myers_diversity(candidates, self.context.input_tokenizer)
            penalty = max(1.0 - diversity - self.diversity_penalty_clip, 0.0) * self.diversity_penalty_weight
        if self.diversity_penalty_max is not None:
            penalty = min(penalty, self.diversity_penalty_max)
        return penalty

    def score(self, data, indices: List[int], generate_fn) -> Dict[int, float]:
        try:
            from .rm_lib import compute_group_translation_scores, _get_response_valid_len
        except ImportError:
            from reward_utils.rm_lib import compute_group_translation_scores, _get_response_valid_len

        raw_responses = _decode_response(data, self.context.input_tokenizer, "none")
        extra_info = data.non_tensor_batch.get("extra_info")
        if extra_info is None:
            raise ValueError("extra_info not found in batch")
        final_translations = [None] * len(raw_responses)
        penalties: Dict[int, float] = {}
        valid_indices: List[int] = []
        for idx in indices:
            parsed = _parse_fused_flash_gqm_response(raw_responses[idx])
            if parsed is None:
                continue
            candidates, candidate_scores = parsed
            if not _is_valid_fused_candidate_count(extra_info[idx], len(candidates)):
                continue
            selected = max(range(len(candidate_scores)), key=candidate_scores.__getitem__)
            final_translations[idx] = candidates[selected]
            penalty = self._diversity_penalty(candidates)
            penalty += _compute_overlong_penalty(_get_response_valid_len(data, idx), self.overlong_buffer_cfg)
            penalties[idx] = penalty
            valid_indices.append(idx)

        result = compute_group_translation_scores(
            data, generate_fn, self.context.tokenizer, self.context.input_tokenizer,
            extractor_type="none", max_prompt_length=self.max_prompt_length,
            prompt_type=self.prompt_type, add_example=self.add_example,
            score_scale_factor=self.score_scale_factor, default_reward=self.default_reward,
            overlong_buffer_cfg=None, enable_language_detection=self.enable_language_detection,
            indices=valid_indices, response_texts=final_translations,
        )
        return {idx: result.get(idx, self.default_reward) - penalties[idx] for idx in valid_indices}


class MultiTaskRewardProcessor:
    """Routes explicitly enabled abilities to isolated reward task handlers."""

    _HANDLER_TYPES = {"ranking": RankingTaskHandler, "fused_flash_gqm": FusedFlashGQMTaskHandler}

    def __init__(self, *args, **kwargs):
        self.config = kwargs.get("config")
        self.tokenizer = kwargs.get("tokenizer")
        self.input_tokenizer = kwargs.get("input_tokenizer", self.tokenizer)
        if self.tokenizer is None or self.input_tokenizer is None:
            raise ValueError("tokenizer and input_tokenizer must be provided")
        custom = _custom(self.config)
        configured = custom.get("multitask_tasks", ["ranking", "fused_flash_gqm"])
        if isinstance(configured, str):
            configured = [configured]
        self.task_names = [str(name).strip().lower() for name in configured]
        unknown = set(self.task_names) - set(self._HANDLER_TYPES)
        if unknown:
            raise ValueError(f"unsupported multitask task(s): {sorted(unknown)}")
        context = RewardTaskContext(self.config, self.tokenizer, self.input_tokenizer)
        self.default_reward = _cfg(self.config, "default_reward", 0.0)
        self.handlers = {}
        for name in self.task_names:
            if name == "ranking":
                handler = RankingTaskHandler(context, kwargs.get("ranking_reward_fn"))
            else:
                handler = FusedFlashGQMTaskHandler(context)
            self.handlers[name] = handler

    def compute_scores(self, data, generate_fn):
        abilities = data.non_tensor_batch.get("ability")
        if abilities is None:
            raise ValueError("ability not found in data.non_tensor_batch")
        grouped = {name: [] for name in self.task_names}
        for idx, ability in enumerate(abilities):
            name = str(ability).strip().lower()
            if name not in self.handlers:
                raise ValueError(f"unsupported or disabled ability: {ability!r}")
            grouped[name].append(idx)
        total_size = data.batch.batch_size[0]
        scores = [self.default_reward] * total_size
        for name in self.task_names:
            for idx, score in self.handlers[name].score(data, grouped[name], generate_fn).items():
                scores[idx] = score
        return scores
