
from typing import Optional
from itertools import combinations


def parse_order(order_str):
    tiers = []
    for group in order_str.split('>'):
        tier = set(x.strip() for x in group.split('='))
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
            score += 0 # we treat ties as incorrect
    return score / total



def _extract_ranking(output_text: str) -> Optional[int]:
    output_text = output_text.strip()
    try:
        last_line_index = output_text.rfind("\n")
        last_line = output_text[last_line_index:].strip()
        return last_line
    except Exception:
        return None

def validate_ranking(test_str: str, ref_str: str) -> bool:
    try:
        if "<" in test_str:
            return False
        ref_tiers = parse_order(ref_str)
        ref_count = sum(len(tiers) for tiers in ref_tiers)

        if len(test_str) != (ref_count-1)*3 + ref_count:
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


def score_reward_fn(data_source, solution_str, ground_truth, extra_info=None):

    pred_ranking_str = _extract_ranking(solution_str)
    if not validate_ranking(pred_ranking_str, ground_truth):
        return 0
    
    return compare_orderings(pred_ranking_str, ground_truth)