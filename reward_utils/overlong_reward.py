from typing import Any, Optional

try:
    from .helpers import _compute_overlong_penalty
except ImportError:
    from reward_utils.helpers import _compute_overlong_penalty


def compute_overlong_penalty_reward_batch(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    overlong_buffer_cfg: Optional[dict] = None,
    **kwargs,
) -> list[dict[str, Any]]:
    """Return only the negative overlong penalty for each sample in the batch."""
    rewards = []
    for extra_info in extra_infos:
        if "valid_response_length" not in extra_info:
            raise KeyError("extra_info must contain valid_response_length for overlong penalty reward.")

        length = int(extra_info["valid_response_length"])
        penalty = _compute_overlong_penalty(length, overlong_buffer_cfg)
        rewards.append(
            {
                "score": -penalty,
                "overlong_penalty": penalty,
                "valid_response_length": length,
            }
        )
    return rewards
