"""Parsing and diversity helpers for fused Flash reward protocols.

This module contains pure response parsing and candidate diversity logic.
The public compatibility imports remain in :mod:`reward_utils.rm_lib`.
"""

import json
import re
from typing import Optional, List, Dict, Tuple, Any

try:
    from .helpers import _block_extractor
except ImportError:
    from reward_utils.helpers import _block_extractor


_FUSED_THINKING_OPEN = "<thinking>"
_FUSED_THINKING_CLOSE = "</thinking>"
_FUSED_RESPONSE_OPEN = "<response>"
_FUSED_RESPONSE_CLOSE = "</response>"
_FUSED_FLASH_GPE_PATTERN = re.compile(
    rf"\s*{re.escape(_FUSED_THINKING_OPEN)}\s*(.*?)\s*"
    rf"{re.escape(_FUSED_THINKING_CLOSE)}\s*"
    rf"{re.escape(_FUSED_RESPONSE_OPEN)}\s*(.*?)\s*"
    rf"{re.escape(_FUSED_RESPONSE_CLOSE)}\s*"
    rf"{re.escape(_FUSED_THINKING_OPEN)}\s*(.*?)\s*"
    rf"{re.escape(_FUSED_THINKING_CLOSE)}\s*"
    rf"{re.escape(_FUSED_RESPONSE_OPEN)}\s*(.*?)\s*"
    rf"{re.escape(_FUSED_RESPONSE_CLOSE)}\s*",
    re.DOTALL,
)
_FUSED_SIMPLE_SEPARATOR = "---"
_FUSED_SIMPLE_CONNECTOR = (
    "Now, review the candidates and produce the best final translation."
)
_FUSED_SIMPLE_ANALYSIS_HEADING = "# Step-by-step Analysis"


def single_extract_score(output_text: str) -> Optional[float]:
    output_text = output_text.strip()
    try:
        last_line_index = output_text.rfind("\n")
        last_line = output_text[last_line_index:].strip()
        score = int(last_line)
        return float(score)
    except Exception:
        return None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3].strip()
    start = text.find("{")
    if start == -1:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _parse_fused_flash_gpe_response(text: Optional[str]) -> Optional[Tuple[List[str], str]]:
    if not isinstance(text, str):
        return None
    tags = (
        _FUSED_THINKING_OPEN,
        _FUSED_THINKING_CLOSE,
        _FUSED_RESPONSE_OPEN,
        _FUSED_RESPONSE_CLOSE,
    )
    if any(text.count(tag) != 2 for tag in tags):
        return None
    match = _FUSED_FLASH_GPE_PATTERN.fullmatch(text)
    if match is None:
        return None
    candidate_thinking, candidate_response, post_edit_thinking, post_edit_response = (
        value.strip() for value in match.groups()
    )
    if not all((candidate_thinking, candidate_response, post_edit_thinking, post_edit_response)):
        return None

    candidate_payload = _extract_json_object(candidate_response)
    raw_candidates = candidate_payload.get("translations") if candidate_payload is not None else None
    if not isinstance(raw_candidates, list):
        return None
    if any(not isinstance(candidate, str) or not candidate.strip() for candidate in raw_candidates):
        return None
    candidates = [candidate.strip() for candidate in raw_candidates]
    final_translation = _block_extractor(post_edit_response)
    if final_translation is None:
        return None
    return candidates, final_translation


def _parse_fused_flash_gpe_markdown_response(
    text: Optional[str],
) -> Optional[Tuple[List[str], str]]:
    """Parse the markdown candidate section and final translation section."""
    if not isinstance(text, str):
        return None
    tags = (
        _FUSED_THINKING_OPEN,
        _FUSED_THINKING_CLOSE,
        _FUSED_RESPONSE_OPEN,
        _FUSED_RESPONSE_CLOSE,
    )
    if any(text.count(tag) != 2 for tag in tags):
        return None
    match = _FUSED_FLASH_GPE_PATTERN.fullmatch(text)
    if match is None:
        return None
    candidate_thinking, candidate_response, post_edit_thinking, post_edit_response = (
        value.strip() for value in match.groups()
    )
    if not all((candidate_thinking, candidate_response, post_edit_thinking, post_edit_response)):
        return None

    candidates = _extract_fused_markdown_candidates(candidate_response)
    if candidates is None:
        return None

    marker = "# Final Translation"
    marker_index = post_edit_response.find(marker)
    if marker_index != -1:
        final_translation = post_edit_response[marker_index + len(marker) :].strip()
        if final_translation.startswith("# ") or "```" in final_translation:
            return None
    else:
        final_translation = _block_extractor(post_edit_response)
    if not final_translation:
        return None
    return candidates, final_translation


def _extract_fused_markdown_candidates(response: str) -> Optional[List[str]]:
    matches = list(re.finditer(r"(?m)^# Candidate ([1-9][0-9]*)[ \t]*$", response))
    if not matches:
        return None
    candidates: List[str] = []
    for index, candidate_match in enumerate(matches):
        if int(candidate_match.group(1)) != index + 1:
            return None
        end = matches[index + 1].start() if index + 1 < len(matches) else len(response)
        value = response[candidate_match.end() : end].strip()
        if not value or "```" in value:
            return None
        candidates.append(value)
    normalized = {_normalize_fused_candidate(candidate) for candidate in candidates}
    if len(normalized) != len(candidates):
        return None
    return candidates


def _split_fused_simple_analysis_section(
    text: str,
    body_heading_pattern: str,
) -> Optional[Tuple[str, str]]:
    text = text.strip()
    analysis_match = re.match(
        rf"^{re.escape(_FUSED_SIMPLE_ANALYSIS_HEADING)}[ \t]*\n",
        text,
    )
    if analysis_match is None:
        return None
    body_match = re.search(body_heading_pattern, text, re.MULTILINE)
    if body_match is None:
        return None
    analysis = text[analysis_match.end() : body_match.start()].strip()
    response = text[body_match.start() :].strip()
    if not analysis or not response:
        return None
    return analysis, response


def _parse_fused_flash_gpe_simple_markdown_response(
    text: Optional[str],
) -> Optional[Tuple[List[str], str]]:
    """Parse the visible-analysis simple Markdown Fused FlashGPE protocol."""
    if not isinstance(text, str):
        return None
    text = text.replace("\r\n", "\n")
    connector_matches = list(re.finditer(
        rf"(?m)^[ \t]*{re.escape(_FUSED_SIMPLE_SEPARATOR)}[ \t]*\n{{2,}}"
        rf"[ \t]*{re.escape(_FUSED_SIMPLE_CONNECTOR)}[ \t]*$",
        text,
    ))
    if len(connector_matches) != 1:
        return None

    connector_match = connector_matches[0]
    candidate_stage = text[: connector_match.start()]
    post_edit_stage = text[connector_match.end() :]
    if not candidate_stage.endswith("\n\n") or not post_edit_stage.startswith("\n\n"):
        return None
    candidate_sections = _split_fused_simple_analysis_section(
        candidate_stage,
        r"^# Candidate 1[ \t]*$",
    )
    post_edit_sections = _split_fused_simple_analysis_section(
        post_edit_stage,
        r"^# Final Translation[ \t]*$",
    )
    if candidate_sections is None or post_edit_sections is None:
        return None

    _, candidate_response = candidate_sections
    _, post_edit_response = post_edit_sections
    candidates = _extract_fused_markdown_candidates(candidate_response)
    if candidates is None:
        return None
    if post_edit_response.count("# Final Translation") != 1:
        return None
    final_section = post_edit_response[len("# Final Translation") :].strip()
    if not final_section.startswith("```"):
        return None
    final_translation = _block_extractor(final_section)
    if not final_translation or final_translation.startswith("# ") or "```" in final_translation:
        return None
    return candidates, final_translation


def _parse_fused_flash_gqm_response(
    text: Optional[str],
) -> Optional[Tuple[List[str], List[int]]]:
    """Parse the simple fused FlashGQM candidate and ranking sections."""
    if not isinstance(text, str):
        return None
    text = text.replace("\r\n", "\n")
    connector_matches = list(re.finditer(
        rf"(?m)^[ \t]*{re.escape(_FUSED_SIMPLE_SEPARATOR)}[ \t]*\n{{2,}}"
        rf"[ \t]*Now, rank and score the candidates\.[ \t]*$",
        text,
    ))
    if len(connector_matches) != 1:
        return None
    connector = connector_matches[0]
    candidate_stage = text[:connector.start()]
    gqm_stage = text[connector.end():]
    if not candidate_stage.endswith("\n\n") or not gqm_stage.startswith("\n\n"):
        return None
    candidate_sections = _split_fused_simple_analysis_section(
        candidate_stage, r"^# Candidate 1[ \t]*$"
    )
    gqm_sections = _split_fused_simple_analysis_section(
        gqm_stage, r"^# Final Ranking[ \t]*$"
    )
    if candidate_sections is None or gqm_sections is None:
        return None
    candidates = _extract_fused_markdown_candidates(candidate_sections[1])
    if candidates is None:
        return None

    gqm_response = gqm_sections[1]
    ranking_marker = "# Final Ranking"
    scores_marker = "# Scores"
    if gqm_response.count(ranking_marker) != 1 or gqm_response.count(scores_marker) != 1:
        return None
    ranking_index = gqm_response.index(ranking_marker)
    scores_index = gqm_response.index(scores_marker)
    if ranking_index >= scores_index:
        return None
    ranking = gqm_response[ranking_index + len(ranking_marker):scores_index].strip()
    score_text = gqm_response[scores_index + len(scores_marker):].strip()
    count = len(candidates)
    identifiers = [chr(ord("A") + index) for index in range(count)]
    if "\n" in ranking or "<" in ranking:
        return None
    tiers = ranking.split(">")
    if not tiers or any(not tier.strip() for tier in tiers):
        return None
    flattened = []
    for tier in tiers:
        values = [value.strip() for value in tier.split("=")]
        if any(value not in identifiers for value in values):
            return None
        flattened.extend(values)
    if len(flattened) != len(set(flattened)) or set(flattened) != set(identifiers):
        return None
    score_map: Dict[str, int] = {}
    for item in score_text.split(","):
        match = re.fullmatch(r"\s*([A-Z])\s*:\s*(10|[0-9])\s*", item)
        if match is None or match.group(1) in score_map:
            return None
        score_map[match.group(1)] = int(match.group(2))
    if set(score_map) != set(identifiers):
        return None
    previous_score = None
    for tier in tiers:
        tier_scores = {score_map[value.strip()] for value in tier.split("=")}
        if len(tier_scores) != 1:
            return None
        current_score = next(iter(tier_scores))
        if previous_score is not None and current_score >= previous_score:
            return None
        previous_score = current_score
    return candidates, [score_map[identifier] for identifier in identifiers]


def _is_valid_fused_candidate_count(extra_info: Any, candidate_count: int) -> bool:
    if not isinstance(extra_info, dict):
        return False
    prompt_type = extra_info.get("prompt_type")
    target_candidate_count = extra_info.get("target_candidate_count")
    max_candidates = extra_info.get("max_candidates")
    try:
        max_candidates = int(max_candidates)
    except (TypeError, ValueError):
        max_candidates = None

    if target_candidate_count is not None:
        try:
            target_candidate_count = int(target_candidate_count)
        except (TypeError, ValueError):
            return False
        if target_candidate_count < 2:
            return False
        if max_candidates is not None and target_candidate_count > max_candidates:
            return False
        return candidate_count == target_candidate_count

    if max_candidates is None:
        return False

    if prompt_type == "markdown":
        return max_candidates >= 2 and candidate_count == max_candidates
    if prompt_type == "fixed_4":
        return max_candidates == 4 and candidate_count == 4
    if prompt_type == "fixed_16":
        return max_candidates == 16 and candidate_count == 16
    if prompt_type == "adaptive":
        return max_candidates >= 2 and 2 <= candidate_count <= max_candidates
    return False


def _normalize_fused_candidate(text: str) -> str:
    return " ".join(text.split()).casefold()


def _has_normalized_duplicate(candidates: List[str]) -> bool:
    normalized = [_normalize_fused_candidate(candidate) for candidate in candidates]
    return len(set(normalized)) != len(normalized)


def _myers_insert_delete_distance(left: List[int], right: List[int]) -> int:
    """Return shortest insert/delete edit distance using Myers' O((N+M)D) algorithm."""
    left_len = len(left)
    right_len = len(right)
    if left_len == 0:
        return right_len
    if right_len == 0:
        return left_len

    frontier = {1: 0}
    for distance in range(left_len + right_len + 1):
        next_frontier: Dict[int, int] = {}
        for diagonal in range(-distance, distance + 1, 2):
            if diagonal == -distance or (
                diagonal != distance
                and frontier.get(diagonal - 1, -1) < frontier.get(diagonal + 1, -1)
            ):
                x = frontier.get(diagonal + 1, 0)
            else:
                x = frontier.get(diagonal - 1, 0) + 1
            y = x - diagonal
            while x < left_len and y < right_len and left[x] == right[y]:
                x += 1
                y += 1
            next_frontier[diagonal] = x
            if x >= left_len and y >= right_len:
                return distance
        frontier = next_frontier
    return left_len + right_len


def _token_myers_diversity(candidates: List[str], tokenizer) -> float:
    tokenized = [list(tokenizer.encode(candidate, add_special_tokens=False)) for candidate in candidates]
    pairwise_distances: List[float] = []
    for left_idx in range(len(tokenized)):
        for right_idx in range(left_idx + 1, len(tokenized)):
            left = tokenized[left_idx]
            right = tokenized[right_idx]
            denominator = len(left) + len(right)
            if denominator == 0:
                pairwise_distances.append(0.0)
                continue
            distance = _myers_insert_delete_distance(left, right)
            pairwise_distances.append(distance / denominator)
    return sum(pairwise_distances) / len(pairwise_distances) if pairwise_distances else 0.0


