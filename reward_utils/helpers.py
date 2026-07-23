from typing import Optional, Dict, List
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
    extract_out = response[block_start + 3 :]
    newline_pos = extract_out.find("\n")
    if newline_pos != -1:
        extract_out = extract_out[newline_pos + 1 :]
    extract_out = extract_out.strip()
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


def _decode_response(data, src_tokenizer, extractor_type: str = "line") -> List[Optional[str]]:
    """Decode batch response token IDs into strings, applying the given extractor strategy."""
    response_list: List[Optional[str]] = []

    for i in range(data.batch.batch_size[0]):
        response_ids = data.batch["responses"][i]
        response_length = response_ids.shape[-1]
        valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        response = src_tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        response = response.replace(src_tokenizer.eos_token, "")
        if extractor_type == "line":
            extracted = _line_extractor(response)
        elif extractor_type == "codeblock":
            extracted = _block_extractor(response)
        elif extractor_type == "oneline":
            extracted = _one_line_extractor(response)
        elif extractor_type == "none":
            extracted = response.strip()
        else:
            raise ValueError(f"extractor_type: {extractor_type}")

        response_list.append(extracted)

    return response_list


def _get_lang_pair(extra_info: dict) -> tuple:
    """Extract (src_lang, tgt_lang) from extra_info dict. Supports both separate keys and 'lang_pair' format like 'en-zh'."""
    if "src_lang" in extra_info and "trg_lang" in extra_info:
        src_lang = extra_info["src_lang"]
        tgt_lang = extra_info["trg_lang"]
    elif "lang_pair" in extra_info:
        src_lang, tgt_lang = extra_info["lang_pair"].split("-")
    else:
        raise ValueError(f"extra_info: {extra_info}")
    return src_lang, tgt_lang


def _group_validate_ranking(text: str, expected_num: int) -> bool:
    """Validate a ranking string like 'B > A = D > C' contains exactly expected_num unique candidates."""
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


def _group_ranking_tiers(ranking_text: str) -> list[set[str]]:
    return [{cid.strip() for cid in tier.split("=")} for tier in ranking_text.split(">")]


def _group_score_tiers(score_dict: Dict[str, int]) -> list[set[str]]:
    return [
        {cid for cid, score in score_dict.items() if score == tier_score}
        for tier_score in sorted(set(score_dict.values()), reverse=True)
    ]


def _single_nonempty_line(text: str) -> Optional[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if len(lines) == 1 else None


def _extract_gqmpe_scores(
    output_text: str,
    prompt_type: str,
    expected_num: int,
) -> Optional[List[int]]:
    post_edit_header = "# Post-edit Analysis"
    final_translation_header = "# Final post-edited translation"
    final_ranking_header = "### Final Ranking:"
    scores_header = "### Scores:"

    post_edit_index = output_text.rfind(post_edit_header)
    final_translation_index = output_text.rfind(final_translation_header)
    if post_edit_index == -1 or final_translation_index <= post_edit_index:
        return None

    post_edit_analysis = output_text[post_edit_index + len(post_edit_header) : final_translation_index].strip()
    final_translation = _block_extractor(output_text)
    if not post_edit_analysis or final_translation is None:
        return None

    gqm_text = output_text[:post_edit_index].strip()
    expected_identifiers = candidate_identifiers[:expected_num]
    ranking_text = None

    if prompt_type == "score":
        lines = gqm_text.splitlines()
        score_line_index = next(
            (i for i in range(len(lines) - 1, -1, -1) if lines[i].strip()),
            None,
        )
        if score_line_index is None:
            return None
        score_text = lines[score_line_index].strip()
        gqm_analysis = "\n".join(lines[:score_line_index]).strip()
    elif prompt_type == "ranking":
        ranking_header_index = gqm_text.rfind(final_ranking_header)
        if ranking_header_index == -1:
            return None
        gqm_analysis = gqm_text[:ranking_header_index].strip()
        ranking_text = _single_nonempty_line(
            gqm_text[ranking_header_index + len(final_ranking_header) :]
        )
        if ranking_text is None or not _group_validate_ranking(ranking_text, expected_num):
            return None
        score_map = _group_ranking_to_scores(ranking_text)
        return [score_map[cid] for cid in expected_identifiers] if gqm_analysis else None
    elif prompt_type == "ranking_score":
        ranking_header_index = gqm_text.rfind(final_ranking_header)
        score_header_index = gqm_text.rfind(scores_header)
        if ranking_header_index == -1 or score_header_index <= ranking_header_index:
            return None
        gqm_analysis = gqm_text[:ranking_header_index].strip()
        ranking_text = _single_nonempty_line(
            gqm_text[ranking_header_index + len(final_ranking_header) : score_header_index]
        )
        score_text = _single_nonempty_line(gqm_text[score_header_index + len(scores_header) :])
        if (
            ranking_text is None
            or score_text is None
            or not _group_validate_ranking(ranking_text, expected_num)
        ):
            return None
    else:
        raise ValueError("prompt_type must be one of ['score', 'ranking', 'ranking_score']")

    score_dict = _group_parse_score_text(score_text)
    if (
        not gqm_analysis
        or score_dict is None
        or len(score_text.split(",")) != expected_num
        or set(score_dict) != set(expected_identifiers)
    ):
        return None
    if ranking_text is not None and _group_ranking_tiers(ranking_text) != _group_score_tiers(score_dict):
        return None
    return [score_dict[cid] for cid in expected_identifiers]


def group_extract_scores(
    output_text: str,
    prompt_type: str,
    expected_num: int,
    task_type: str = "gqm",
) -> Optional[List[int]]:
    """Parse GQM or GQMPE scores in the configured score/ranking format."""
    output_text = output_text.strip()
    if task_type not in {"gqm", "gqmpe"}:
        raise ValueError("task_type must be one of ['gqm', 'gqmpe']")
    try:
        if task_type == "gqmpe":
            return _extract_gqmpe_scores(output_text, prompt_type, expected_num)
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


def _compute_overlong_penalty(length: int, overlong_buffer_cfg: Optional[Dict]) -> float:
    """Linear penalty in [0, penalty_factor] for responses exceeding max_resp_len - buffer_len."""
    if not overlong_buffer_cfg:
        return 0.0
    if not overlong_buffer_cfg.get("enable", False):
        return 0.0
    max_resp_len = overlong_buffer_cfg.get("max_resp_len", None)
    buffer_len = overlong_buffer_cfg.get("len", 0)
    penalty_factor = overlong_buffer_cfg.get("penalty_factor", 0.0)
    if max_resp_len is None or buffer_len <= 0 or penalty_factor <= 0.0:
        return 0.0
    threshold = max_resp_len - buffer_len
    if length <= threshold:
        return 0.0
    delta = length - threshold
    if delta >= buffer_len:
        return penalty_factor
    return penalty_factor * (delta / buffer_len)
