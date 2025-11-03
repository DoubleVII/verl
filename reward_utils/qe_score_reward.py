
from typing import Optional


def _extract_score(output_text: str) -> Optional[int]:
    output_text = output_text.strip()
    try:
        last_line_index = output_text.rfind("\n")
        last_line = output_text[last_line_index:].strip()
        if "." in last_line:
            last_line = last_line.split(".")[0]
        score = int(last_line)
        return score
    except Exception:
        return None

def score_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    _error_reward_map = {
        0: 1,
        1: 0.8,
        2: 0.4,
    }

    score = _extract_score(solution_str)
    ground_truth = int(ground_truth)
    if score is None:
        return 0
    score_error = abs(score - ground_truth)
    
    if score_error in _error_reward_map:
        reward = _error_reward_map[score_error]
    else:
        reward = 0
    
    # correction
    if ground_truth == 0 or ground_truth == 10:
        reward *= 1.6
        reward = min(reward, 1)
    elif ground_truth == 1 or ground_truth == 9:
        reward *= 1.2
        reward = min(reward, 1)
    
    return reward