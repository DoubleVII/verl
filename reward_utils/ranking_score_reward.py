from typing import Optional, Union
from itertools import combinations
import json

def get_cand_num(rank_text: str) -> int:
    rank_text = rank_text.strip()
    # 3*(x-1)+x=len(rank_text)
    # 4*x-3=len(rank_text)
    # x=(len(rank_text)+3)/4
    assert (len(rank_text) + 3) % 4 == 0
    return (len(rank_text) + 3) // 4


def _split_last_line(text: str) -> Optional[tuple[str, str]]:
    text = text.strip()
    last_line_index = text.rfind("\n")
    if last_line_index == -1:
        return None
    last_line = text[last_line_index:].strip()
    residual_text = text[:last_line_index].strip()
    return residual_text, last_line


def parse_order(order_str):
    tiers = []
    for group in order_str.split(">"):
        tier = set(x.strip() for x in group.split("="))
        tiers.append(tier)
    return tiers


def pair_relation(tiers, x, y):
    for i, tier in enumerate(tiers):
        if x in tier:
            ix = i
        if y in tier:
            iy = i
    if ix < iy:
        return 1
    elif ix > iy:
        return -1
    else:
        return 0


def compare_orderings(test_str, ref_str):
    try:
        test_tiers = parse_order(test_str)
        ref_tiers = parse_order(ref_str)
        items = sorted(set().union(*test_tiers))

        total = 0
        score = 0
        for x, y in combinations(items, 2):
            r_ref = pair_relation(ref_tiers, x, y)
            r_test = pair_relation(test_tiers, x, y)
            total += 1
            if r_ref == r_test:
                score += 1
            elif 0 in (r_ref, r_test):
                score += 0  # we treat ties as incorrect
        return score / total
    except Exception:
        return 0


def _extract_ranking(output_text: str) -> Optional[tuple[str, str]]:
    return _split_last_line(output_text)


def validate_ranking(test_str: str, ref_str: str) -> bool:
    try:
        if "<" in test_str:
            return False
        ref_tiers = parse_order(ref_str)
        ref_count = sum(len(tiers) for tiers in ref_tiers)

        if len(test_str) != (ref_count - 1) * 3 + ref_count:
            return False

        test_tiers = parse_order(test_str)
        test_count = sum(len(tiers) for tiers in test_tiers)
        if test_count != ref_count:
            return False
        for tiers in ref_tiers:
            for candidate_identifier in tiers:
                if test_str.count(candidate_identifier) != 1:
                    return False
        return True
    except Exception:
        return False


def ranking_reward_fn_no_cot(data_source, solution_str, ground_truth, extra_info=None):
    """
    ranking reward function for no cot (direct ranking prediction).
    """
    pred_ranking_str = solution_str.strip()
    if not validate_ranking(pred_ranking_str, ground_truth):
        return {"score": 0, "valid_answer": 0}

    if solution_str.count(pred_ranking_str) != 1:  # prohibit repeat output
        return {"score": 0, "valid_answer": 0}

    return {
        "score": compare_orderings(pred_ranking_str, ground_truth),
        "valid_answer": 1,
    }


def ranking_reward_fn_zero(data_source, solution_str, ground_truth, extra_info=None):
    """
    ranking reward function for zero cot (base model RL).
    """
    extract_out = _extract_ranking(solution_str)
    if extract_out is None:
        return {"score": 0, "valid_answer": 0}
    cot_text, pred_ranking_str = extract_out
    cand_num = get_cand_num(ground_truth)

    if (
        len(cot_text) < 50 * cand_num
    ):  # we require at least 50 chars of cot per candidate
        return {"score": 0, "valid_answer": 0}

    if not validate_ranking(pred_ranking_str, ground_truth):
        return {"score": 0, "valid_answer": 0}

    if solution_str.count(pred_ranking_str) != 1:  # prohibit repeat output
        return {"score": 0, "valid_answer": 0}

    return {
        "score": compare_orderings(pred_ranking_str, ground_truth),
        "valid_answer": 1,
    }


def ranking_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    """
    ranking reward function for cold-start (SFT-model RL).
    """
    extract_out = _extract_ranking(solution_str)
    if extract_out is None:
        return {"score": 0, "valid_answer": 0}
    cot_text, pred_ranking_str = extract_out

    if len(cot_text) == 0:
        return {"score": 0, "valid_answer": 0}
    
    return ranking_reward_fn_no_cot(data_source, solution_str, ground_truth, extra_info)


def _score_to_rank(d) -> str:
    # Sort by score descending, then by key ascending
    sorted_items = sorted(d.items(), key=lambda x: (-x[1], x[0]))

    # Group by score
    result_parts = []
    current_score = None
    current_group = []

    for k, v in sorted_items:
        if v != current_score:
            if current_group:
                result_parts.append(" = ".join(current_group))
            current_score = v
            current_group = [k]
        else:
            current_group.append(k)

    # Add the last group
    if current_group:
        result_parts.append(" = ".join(current_group))

    # Join groups with ' > '
    return " > ".join(result_parts)


def compare_ranking_scores(
    test_score_dict: dict, ref_score_dict: dict, error_to_reward: dict = None
) -> float:
    if not error_to_reward:
        error_to_reward = {0: 1, 1: 0.6, 2: 0.2}
    try:
        cand_items = list(ref_score_dict.keys())

        total = 0
        score = 0
        for x, y in combinations(cand_items, 2):
            margin_ref = ref_score_dict[x] - ref_score_dict[y]
            margin_test = test_score_dict[x] - test_score_dict[y]
            total += 1

            margin_error = abs(margin_ref - margin_test)
            if margin_error in error_to_reward:
                score += error_to_reward[margin_error]
        return score / total
    except Exception:
        return 0


def _extract_ranking_score(output_text: str) -> Optional[dict]:
    try:
        residual_text, score_text = _split_last_line(output_text)
        residual_text, score_header = _split_last_line(residual_text)
        residual_text, ranking_text = _split_last_line(residual_text)
        residual_text, ranking_header = _split_last_line(residual_text)
        return {
            "cot_text": residual_text,
            "score_text": score_text,
            "ranking_text": ranking_text,
            "score_header": score_header,
            "ranking_header": ranking_header,
        }
    except: # catch unpack error
        return None


def _parse_score_text(score_text: str) -> Optional[dict]:
    """
    B: 6, A: 5, C: 2
    """
    try:
        score_text = score_text.strip()
        score_text = score_text.split(",")
        score_dict = {}
        for item in score_text:
            item = item.strip()
            candidate_identifier, score = item.split(":")
            candidate_identifier = candidate_identifier.strip()
            score = int(score.strip())
            score_dict[candidate_identifier] = score
        return score_dict
    except:
        return None


def ranking_score_reward_fn_no_cot(
    data_source, solution_str, ground_truth: Union[str, dict], extra_info=None
):
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    reward_out = {"score": 0, "valid_answer": 0, "ranking_reward": 0, "score_reward": 0}

    solution_str = solution_str.strip()
    if solution_str.count("\n") != 1:
        return reward_out
    ranking_text, score_text = solution_str.split("\n")

    pred_score_dict = _parse_score_text(score_text)
    if pred_score_dict is None:
        return reward_out

    pred_score_to_rank = _score_to_rank(pred_score_dict)
    consistency_check = ranking_reward_fn_no_cot(
        data_source, ranking_text, pred_score_to_rank, extra_info
    )
    if consistency_check["score"] != 1:
        return reward_out

    # validation
    if len(pred_score_dict) != len(ground_truth):
        return reward_out

    for candidate_identifier in pred_score_dict.keys():
        if candidate_identifier not in ground_truth:
            return reward_out

    ref_score_to_rank = _score_to_rank(ground_truth)
    ranking_reward = compare_orderings(pred_score_to_rank, ref_score_to_rank)
    score_reward = compare_ranking_scores(pred_score_dict, ground_truth)
    reward = ranking_reward + score_reward  # [0, 2]
    reward_out["score"] = reward
    reward_out["valid_answer"] = 1
    reward_out["ranking_reward"] = ranking_reward
    reward_out["score_reward"] = score_reward
    return reward_out


def ranking_score_reward_fn(
    data_source, solution_str, ground_truth: Union[str, dict], extra_info=None
):
    reward_out = {"score": 0, "valid_answer": 0, "ranking_reward": 0, "score_reward": 0}
    extract_out = _extract_ranking_score(solution_str)
    if extract_out is None:
        return reward_out
    
    cot_text, score_text, ranking_text, score_header, ranking_header = (
        extract_out["cot_text"],
        extract_out["score_text"],
        extract_out["ranking_text"],
        extract_out["score_header"],
        extract_out["ranking_header"],
    )

    if len(cot_text) == 0:
        return reward_out
    
    no_cot_solution_str = f"{ranking_text}\n{score_text}"
    return ranking_score_reward_fn_no_cot(
        data_source, no_cot_solution_str, ground_truth, extra_info
    )
