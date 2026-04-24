from typing import Optional, Dict, List, Any
from reward_utils.config import candidate_identifiers

def _line_extractor(response: str) -> Optional[str]:
    response = response.strip()
    if not response:
        return None
    last = response.split("\n")[-1].strip()
    if not last:
        return None
    return last

def _block_extractor(response: str) -> Optional[str]:
    response = response.strip()
    if response.count("```") != 2:
        return None
    if not response:
        return None
    if not response.endswith("```"):
        return None
    response = response[:-3]
    block_start = response.rfind("```")
    if block_start == -1:
        return None
    extract_out = response[block_start + 3 :].strip()
    if not extract_out:
        return None
    return extract_out


def _one_line_extractor(response: str) -> Optional[str]:
    response = response.strip()
    if not response:
        return None
    if "\n" in response:
        return None
    return response



def _group_validate_ranking(text: str, expected_num: int) -> bool:
    try:
        if "<" in text:
            return False
        tiers = []
        for group in text.split('>'):
            tier = set(x.strip() for x in group.split('='))
            tiers.append(tier)
        count = sum(len(t) for t in tiers)
        if count != expected_num:
            return False
        for cid in candidate_identifiers[:expected_num]:
            if text.count(cid) != 1:
                return False
        return True
    except Exception:
        return False

def _group_ranking_to_scores(ranking_text: str) -> Dict[str, int]:
    tiers = ranking_text.split('>')
    score_map = {}
    max_score = len(tiers) - 1
    for i, tier in enumerate(tiers):
        for cid in tier.split('='):
            score_map[cid.strip()] = max_score - i
    return score_map

def _group_parse_score_text(score_text: str) -> Optional[Dict[str, int]]:
    try:
        result = {}
        for it in score_text.split(','):
            k, v = it.split(':')
            result[k.strip()] = int(v.strip())
        return result
    except Exception:
        return None

def group_extract_scores(output_text: str, prompt_type: str, expected_num: int) -> Optional[List[int]]:
    output_text = output_text.strip()
    try:
        if "\n" not in output_text:
            last_line = output_text
        else:
            idx = output_text.rfind("\n")
            last_line = output_text[idx:].strip()
        if prompt_type == "score":
            scores = [int(s.strip().split(":")[-1]) for s in last_line.split(",")]
            if len(scores) != expected_num:
                return None
            return scores
        if prompt_type == "ranking":
            if not _group_validate_ranking(last_line, expected_num):
                return None
            score_map = _group_ranking_to_scores(last_line)
            scores = [score_map[cid] for cid in candidate_identifiers[:expected_num]]
            return scores
        if prompt_type == "ranking_score":
            score_dict = _group_parse_score_text(last_line)
            if score_dict is None:
                return None
            scores = [score_dict[cid] for cid in candidate_identifiers[:expected_num]]
            return scores
        raise ValueError(f"prompt_type must be one of ['score', 'ranking', 'ranking_score']")
    except Exception:
        return None