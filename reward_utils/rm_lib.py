from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Iterable, Any

try:
    from .helpers import (
        _line_extractor,
        _block_extractor,
        _one_line_extractor,
        _decode_response,
        _get_lang_pair,
        group_extract_scores,
        _compute_overlong_penalty,
    )
except ImportError:
    from reward_utils.helpers import (
        _line_extractor,
        _block_extractor,
        _one_line_extractor,
        _decode_response,
        _get_lang_pair,
        group_extract_scores,
        _compute_overlong_penalty,
    )

try:
    from .config import LANG_MAP
except ImportError:
    from reward_utils.config import LANG_MAP

try:
    from .prompts import (
        single_get_prompt,
        get_GQM_prompt,
        _seedx_build_prompt,
        _vanilla_rm_build_prompt,
    )
except ImportError:
    from reward_utils.prompts import (
        single_get_prompt,
        get_GQM_prompt,
        _seedx_build_prompt,
        _vanilla_rm_build_prompt,
    )

try:
    from .language_detector import is_language_match
except ImportError:
    from reward_utils.language_detector import is_language_match


@dataclass
class RewardProcessorOutput:
    scores: Any
    non_tensor_batch: Dict[str, Any] = field(default_factory=dict)


REWARD_MODEL_PROMPTS_KEY = "reward_model_prompts"
REWARD_MODEL_RESPONSES_KEY = "reward_model_responses"


def single_extract_score(output_text: str) -> Optional[float]:
    output_text = output_text.strip()
    try:
        last_line_index = output_text.rfind("\n")
        last_line = output_text[last_line_index:].strip()
        score = int(last_line)
        return float(score)
    except Exception:
        return None


def _apply_response_extractor(response: str, extractor_type: str) -> Optional[str]:
    if extractor_type == "line":
        return _line_extractor(response)
    if extractor_type == "codeblock":
        return _block_extractor(response)
    if extractor_type == "oneline":
        return _one_line_extractor(response)
    if extractor_type == "none":
        response = response.strip()
        return response if response else None
    raise ValueError(f"extractor_type: {extractor_type}")


def _get_obj_value(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item.get("content", ""))))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(content, dict):
        return str(content.get("text", content.get("content", "")))
    return str(content)


def _decode_last_assistant_response(data, input_tokenizer, extractor_type: str) -> List[Optional[str]]:
    """Extract the final assistant message from SGLang multi-turn rollout output."""
    fallback_responses = _decode_response(data, input_tokenizer, extractor_type)
    message_rows = data.non_tensor_batch.get("messages", None)
    if message_rows is None:
        return fallback_responses

    response_list: List[Optional[str]] = []
    for idx in range(data.batch.batch_size[0]):
        row = message_rows[idx]
        messages = _get_obj_value(row, "messages", None)
        if messages is None:
            response_list.append(fallback_responses[idx])
            continue

        last_assistant = None
        for message in messages:
            if str(_get_obj_value(message, "role", "")).lower() == "assistant":
                last_assistant = message

        if last_assistant is None:
            response_list.append(fallback_responses[idx])
            continue

        content = _content_to_text(_get_obj_value(last_assistant, "content", ""))
        response_list.append(_apply_response_extractor(content, extractor_type))

    return response_list


def _extract_assistant_turn_text(data, turn_idx: int) -> List[Optional[str]]:
    """Extract raw assistant message content by assistant-turn index from stored chat messages."""
    message_rows = data.non_tensor_batch.get("messages", None)
    total_size = data.batch.batch_size[0]
    if message_rows is None:
        return [None] * total_size

    response_list: List[Optional[str]] = []
    for idx in range(total_size):
        row = message_rows[idx]
        messages = _get_obj_value(row, "messages", None)
        if messages is None:
            response_list.append(None)
            continue

        assistant_texts = []
        for message in messages:
            if str(_get_obj_value(message, "role", "")).lower() == "assistant":
                assistant_texts.append(_content_to_text(_get_obj_value(message, "content", "")))

        if turn_idx < len(assistant_texts):
            response_list.append(assistant_texts[turn_idx])
        else:
            response_list.append(None)

    return response_list


def _normalize_duplicate_text(text: str) -> str:
    return text.strip()


def _get_response_valid_len(data, idx: int) -> int:
    response_ids = data.batch["responses"][idx]
    resp_len = response_ids.shape[-1]
    valid_len = data.batch["attention_mask"][idx][-resp_len:].sum()
    try:
        return int(valid_len)
    except Exception:
        return resp_len


def _dedupe_texts_with_index(texts: List[str], target_idx: int) -> Tuple[List[str], int, List[int]]:
    seen: Dict[str, int] = {}
    unique_texts: List[str] = []
    score_indices: List[int] = []
    target_text = _normalize_duplicate_text(texts[target_idx])
    target_unique_idx = 0

    for text in texts:
        key = _normalize_duplicate_text(text)
        if key not in seen:
            seen[key] = len(unique_texts)
            unique_texts.append(text)
        score_indices.append(seen[key])
        if key == target_text:
            target_unique_idx = seen[key]

    return unique_texts, target_unique_idx, score_indices


def _try_score_gqm_post_edit_from_first_turn(
    data,
    *,
    indices: List[int],
    post_edit_responses: List[Optional[str]],
    prompt_format: str,
    score_scale_factor: float,
    rm_max_candidates: int,
    overlong_buffer_cfg,
    enable_language_detection: bool,
) -> Tuple[Dict[int, float], List[int]]:
    first_turn_responses = _extract_assistant_turn_text(data, 0)
    extra_info_list = data.non_tensor_batch["extra_info"]
    scores_dict: Dict[int, float] = {}
    remaining_indices: List[int] = []

    for idx in indices:
        pe_mt_text = post_edit_responses[idx]
        if pe_mt_text is None:
            remaining_indices.append(idx)
            continue

        extra = extra_info_list[idx]
        if enable_language_detection:
            _, tgt_lang = _get_lang_pair(extra)
            if not is_language_match(pe_mt_text, tgt_lang):
                remaining_indices.append(idx)
                continue

        raw_mt_texts = extra_info_list[idx].get("mt_texts", [])
        mt_texts = [] if raw_mt_texts is None else list(raw_mt_texts)
        if 1 + len(mt_texts) > rm_max_candidates:
            mt_texts = mt_texts[: rm_max_candidates - 1]

        pe_key = _normalize_duplicate_text(pe_mt_text)
        match_idx = None
        for candidate_idx, mt_text in enumerate(mt_texts):
            if _normalize_duplicate_text(str(mt_text)) == pe_key:
                match_idx = candidate_idx
                break

        if match_idx is None:
            remaining_indices.append(idx)
            continue

        first_turn = first_turn_responses[idx]
        if first_turn is None:
            remaining_indices.append(idx)
            continue

        candidate_scores = group_extract_scores(first_turn, prompt_format, len(mt_texts))
        if candidate_scores is None:
            remaining_indices.append(idx)
            continue

        pe_mt_score = candidate_scores[match_idx]
        mean_all = sum(candidate_scores) / len(candidate_scores)
        relative_reward = (pe_mt_score - mean_all) * score_scale_factor
        penalty = _compute_overlong_penalty(_get_response_valid_len(data, idx), overlong_buffer_cfg)
        scores_dict[idx] = relative_reward - penalty

    return scores_dict, remaining_indices


def compute_group_translation_scores(
    data,
    generate_fn,
    tokenizer,
    input_tokenizer,
    *,
    extractor_type: str,
    max_prompt_length: int,
    prompt_type: str,
    add_example: bool,
    score_scale_factor: float,
    default_reward: float,
    overlong_buffer_cfg,
    enable_language_detection: bool,
    indices: Optional[List[int]] = None,
    response_texts: Optional[List[Optional[str]]] = None,
    return_reward_model_metadata: bool = False,
    return_gqm_outputs: Optional[bool] = None,
) -> Any:
    """Shared group-based translation scoring pipeline.

    Decodes responses, groups by uid, deduplicates translations, builds prompts,
    calls generate_fn, extracts scores, and applies penalties.

    Args:
        indices: If None, process all items in the batch. Otherwise, process only
                 the specified indices (used by MultiTaskSelfRewardProcessor for
                 translation-only subset).

    Returns:
        Dict mapping original batch index to score.
    """
    if return_gqm_outputs is not None:
        return_reward_model_metadata = return_gqm_outputs

    responses = response_texts if response_texts is not None else _decode_response(data, input_tokenizer, extractor_type)
    extra = data.non_tensor_batch["extra_info"]

    if indices is None:
        indices = list(range(len(responses)))
    if not indices:
        # Keep distributed RM generation collectives aligned across ranks even for empty local work; see fix de799290.
        generate_fn([])
        return ({}, {}, {}) if return_reward_model_metadata else {}

    uids = data.non_tensor_batch.get("uid", None)
    if uids is None:
        raise ValueError("uid not found in batch")
    uids = list(uids)

    # Group indices by uid
    groups: Dict[str, List[int]] = {}
    for idx in indices:
        key = str(uids[idx])
        if key not in groups:
            groups[key] = []
        groups[key].append(idx)
    groups = list(groups.items())

    prompt_list: List[Dict[str, List[int]]] = []
    kept_groups: List[Dict] = []
    zero_groups: List[List[int]] = []


    for uid_key, group_indices in groups:
        src_text = extra[group_indices[0]]["src_text"]
        notes = extra[group_indices[0]].get("notes", None)
        ref_text = extra[group_indices[0]].get("ref_text", None)
        ref_lang = extra[group_indices[0]].get("ref_lang", None)
        src_lang, tgt_lang = _get_lang_pair(extra[group_indices[0]])
        if len(src_lang) == 2:
            src_lang = LANG_MAP[src_lang]
        if len(tgt_lang) == 2:
            tgt_lang = LANG_MAP[tgt_lang]

        seen: Dict[str, int] = {}
        unique_texts: List[str] = []
        dup_map: List[List[int]] = []
        invalid_indices: List[int] = []

        for idx in group_indices:
            t = responses[idx]
            if t is None:
                invalid_indices.append(idx)
                continue
            if enable_language_detection:
                tgt_lang_code = (
                    extra[group_indices[0]]["trg_lang"]
                    if "trg_lang" in extra[group_indices[0]]
                    else extra[group_indices[0]]["lang_pair"].split("-")[1]
                )
                if not is_language_match(t, tgt_lang_code):
                    invalid_indices.append(idx)
                    continue
            if t in seen:
                dup_map[seen[t]].append(idx)
            else:
                seen[t] = len(unique_texts)
                unique_texts.append(t)
                dup_map.append([idx])

        valid_indices = [i for i in group_indices if i not in invalid_indices]

        if len(unique_texts) <= 1:
            if valid_indices:
                zero_groups.append(valid_indices)
            for inv in invalid_indices:
                zero_groups.append([inv])
            continue

        prompt = get_GQM_prompt(
            src_lang, tgt_lang, src_text, unique_texts,
            prompt_type, add_example=add_example, notes=notes,
            ref_text=ref_text, ref_lang=ref_lang,
        )
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        raw_ids = tokenizer.encode(input_text, add_special_tokens=False)

        if len(raw_ids) > max_prompt_length:
            if valid_indices:
                zero_groups.append(valid_indices)
            for inv in invalid_indices:
                zero_groups.append([inv])
            continue

        candidate_lens: List[int] = []
        for targets in dup_map:
            first_idx = targets[0]
            response_ids = data.batch["responses"][first_idx]
            resp_len = response_ids.shape[-1]
            valid_len = data.batch["attention_mask"][first_idx][-resp_len:].sum()
            try:
                candidate_lens.append(int(valid_len))
            except Exception:
                candidate_lens.append(resp_len)

        kept_groups.append({
            "uid": [uid_key],
            "dup_map": dup_map,
            "candidate_lens": candidate_lens,
        })
        prompt_list.append({"prompt_token_ids": raw_ids})

    scores_dict: Dict[int, float] = {}
    gqm_prompts_dict: Dict[int, Any] = {}
    gqm_outputs_dict: Dict[int, str] = {}
    reward_failed_count = 0

    # Keep distributed RM generation collectives aligned across ranks even when this task has no local prompts; see fix de799290.
    outputs = generate_fn(prompt_list)
    if prompt_list:
        for j, output in enumerate(outputs):
            text = output.outputs[0].text
            group_info = kept_groups[j]
            dup_map = group_info["dup_map"]
            candidate_lens = group_info.get("candidate_lens", [0] * len(dup_map))
            scores = group_extract_scores(text, prompt_type, len(dup_map))
            if scores is None:
                scores = [default_reward] * len(dup_map)
                reward_failed_count += len(dup_map)
            normalized = [s * score_scale_factor for s in scores]
            for k, targets in enumerate(dup_map):
                penalty = _compute_overlong_penalty(candidate_lens[k], overlong_buffer_cfg)
                sc = normalized[k] - penalty
                for idx in targets:
                    scores_dict[idx] = sc
                    if return_reward_model_metadata:
                        gqm_prompts_dict[idx] = prompt_list[j]
                        gqm_outputs_dict[idx] = text

    for zero_indices in zero_groups:
        for idx in zero_indices:
            scores_dict[idx] = default_reward

    print(f"[DEBUG] Reward failed count: {reward_failed_count} / {len(scores_dict)}")
    if return_reward_model_metadata:
        return scores_dict, gqm_prompts_dict, gqm_outputs_dict
    return scores_dict


def prepare_group_post_edit_inputs(
    data,
    tokenizer,
    input_tokenizer,
    *,
    extractor_type: str,
    max_prompt_length: int,
    prompt_format: str,
    add_example: bool,
    rm_max_candidates: int,
    enable_language_detection: bool,
    indices: Optional[List[int]] = None,
    response_texts: Optional[List[Optional[str]]] = None,
) -> Tuple[List[Dict[str, List[int]]], List[Dict], int]:
    responses = response_texts if response_texts is not None else _decode_response(data, input_tokenizer, extractor_type)
    extra_info_list = data.non_tensor_batch["extra_info"]
    total_size = len(responses)

    if indices is None:
        indices = list(range(total_size))

    prompt_list: List[Dict[str, List[int]]] = []
    kept_info: List[Dict] = []
    for idx in indices:
        pe_mt_text = responses[idx]
        if pe_mt_text is None:
            continue

        extra = extra_info_list[idx]
        src_lang, tgt_lang = _get_lang_pair(extra)

        if enable_language_detection:
            if not is_language_match(pe_mt_text, tgt_lang):
                continue

        src_text = extra.get("src_text")
        raw_mt_texts = extra.get("mt_texts", [])
        mt_texts = [] if raw_mt_texts is None else list(raw_mt_texts)
        notes = extra.get("notes", None)
        ref_text = extra.get("ref_text", None)
        ref_lang = extra.get("ref_lang", None)

        if 1 + len(mt_texts) > rm_max_candidates:
            mt_texts = mt_texts[: rm_max_candidates - 1]

        all_mt_texts = mt_texts + [pe_mt_text]
        all_mt_texts, pe_score_idx, score_indices = _dedupe_texts_with_index(all_mt_texts, len(all_mt_texts) - 1)
        num_candidates = len(all_mt_texts)
        if num_candidates < 2:
            continue

        prompt = get_GQM_prompt(
            source_lang=src_lang,
            target_lang=tgt_lang,
            source_text=src_text,
            mt_texts=all_mt_texts,
            prompt_format=prompt_format,
            add_example=add_example,
            notes=notes,
            ref_text=ref_text,
            ref_lang=ref_lang,
        )
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        raw_ids = tokenizer.encode(input_text, add_special_tokens=False)

        if len(raw_ids) > max_prompt_length:
            continue

        candidate_len = _get_response_valid_len(data, idx)

        kept_info.append({
            "orig_idx": idx,
            "num_candidates": num_candidates,
            "pe_score_idx": pe_score_idx,
            "score_indices": score_indices,
            "response_len": candidate_len,
        })
        prompt_list.append({"prompt_token_ids": raw_ids})

    return prompt_list, kept_info, total_size


def process_group_post_edit_outputs(
    outputs,
    kept_info,
    *,
    prompt_format: str,
    score_scale_factor: float,
    default_reward: float,
    overlong_buffer_cfg,
) -> Dict[int, float]:
    scores_dict: Dict[int, float] = {}
    for j, output in enumerate(outputs):
        text = output.outputs[0].text
        info = kept_info[j]
        num_candidates = info["num_candidates"]
        orig_idx = info["orig_idx"]
        pe_score_idx = info.get("pe_score_idx", num_candidates - 1)
        scores = group_extract_scores(text, prompt_format, num_candidates)
        if scores is None:
            scores_dict[orig_idx] = default_reward
            continue
        pe_mt_score = scores[pe_score_idx]
        score_indices = info.get("score_indices", list(range(num_candidates)))
        mean_scores = [scores[score_idx] for score_idx in score_indices[:-1]]
        mean_all = sum(mean_scores) / len(mean_scores)
        relative_reward = (pe_mt_score - mean_all) * score_scale_factor
        penalty = _compute_overlong_penalty(info["response_len"], overlong_buffer_cfg)
        scores_dict[orig_idx] = relative_reward - penalty
    return scores_dict


def compute_group_post_edit_scores(
    data,
    generate_fn,
    tokenizer,
    input_tokenizer,
    *,
    extractor_type: str,
    max_prompt_length: int,
    prompt_format: str,
    add_example: bool,
    score_scale_factor: float,
    default_reward: float,
    rm_max_candidates: int,
    overlong_buffer_cfg,
    enable_language_detection: bool,
    indices: Optional[List[int]] = None,
    score_mode: str = "mt_group_advantage",
    response_texts: Optional[List[Optional[str]]] = None,
) -> Dict[int, float]:
    if score_mode == "grpo_group_score":
        return compute_group_translation_scores(
            data,
            generate_fn,
            tokenizer,
            input_tokenizer,
            extractor_type=extractor_type,
            max_prompt_length=max_prompt_length,
            prompt_type=prompt_format,
            add_example=add_example,
            score_scale_factor=score_scale_factor,
            default_reward=default_reward,
            overlong_buffer_cfg=overlong_buffer_cfg,
            enable_language_detection=enable_language_detection,
            indices=indices,
            response_texts=response_texts,
        )
    if score_mode != "mt_group_advantage":
        raise ValueError(
            "group post-edit score_mode must be one of "
            "['mt_group_advantage', 'grpo_group_score'], got "
            f"{score_mode!r}"
        )

    prompt_list, kept_info, _ = prepare_group_post_edit_inputs(
        data,
        tokenizer,
        input_tokenizer,
        extractor_type=extractor_type,
        max_prompt_length=max_prompt_length,
        prompt_format=prompt_format,
        add_example=add_example,
        rm_max_candidates=rm_max_candidates,
        enable_language_detection=enable_language_detection,
        indices=indices,
        response_texts=response_texts,
    )
    # Keep distributed RM generation collectives aligned across ranks even when this task has no local prompts; see fix de799290.
    outputs = generate_fn(prompt_list)
    if not prompt_list:
        return {}
    return process_group_post_edit_outputs(
        outputs,
        kept_info,
        prompt_format=prompt_format,
        score_scale_factor=score_scale_factor,
        default_reward=default_reward,
        overlong_buffer_cfg=overlong_buffer_cfg,
    )


class RewardModelProcessor:
    """Single-translation generative RM: prompts the RM to score each translation individually."""
    def __init__(self, *args, **kwargs):
        self.config = kwargs.get("config")
        self.tokenizer = kwargs.get("tokenizer", None)
        self.input_tokenizer = kwargs.get("input_tokenizer", self.tokenizer)
        self.max_prompt_length = self.config.prompt_length
        self.extractor_type = self.config.custom_processor.get("extractor_type", "line")
        print(f"Using extractor_type: {self.extractor_type}")
        self.score_scale_factor = getattr(self.config, "score_scale_factor", 0.1)
        self.default_reward = getattr(self.config, "default_reward", 0.0)
        self.overlong_buffer_cfg = self.config.custom_processor.get("overlong_buffer", None)
        self.enable_language_detection = self.config.custom_processor.get("enable_language_detection", False)
        if self.enable_language_detection:
            print(f"Language detection enabled")
        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided")
        if self.input_tokenizer is None:
            raise ValueError("input_tokenizer must be provided")

    def process_input(self, data):
        response_list = _decode_response(data, self.input_tokenizer, self.extractor_type)
        extra_info_list = data.non_tensor_batch["extra_info"]
        src_text_list = [item["src_text"] for item in extra_info_list]
        notes_list = [item.get("notes", None) for item in extra_info_list]
        lang_pair_list = [_get_lang_pair(item) for item in extra_info_list]
        src_langs, tgt_langs = zip(*lang_pair_list)

        assert len(src_text_list) == len(response_list) == len(src_langs) == len(tgt_langs)
        prompt_list = []
        kept_indices = []
        filtered_indices = []
        for idx, (src_text, mt_text, src_lang, tgt_lang, notes) in enumerate(zip(src_text_list, response_list, src_langs, tgt_langs, notes_list)):
            if mt_text is None:
                filtered_indices.append(idx)
                continue
            if self.enable_language_detection:
                if not is_language_match(mt_text, tgt_lang):
                    filtered_indices.append(idx)
                    continue
            prompt = single_get_prompt(src_text, mt_text, src_lang, tgt_lang, notes=notes)
            messages = [
                {"role": "user", "content": prompt},
            ]
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            raw_prompt_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
            # filter by max_prompt_length; too-long entries will be scored as 0 later
            if len(raw_prompt_ids) > self.max_prompt_length:
                filtered_indices.append(idx)
                continue
            kept_indices.append(idx)
            prompt_list.append({"prompt_token_ids": raw_prompt_ids})
        total_size = len(src_text_list)
        return prompt_list, kept_indices, total_size

    def process_output(self, outputs, data, kept_indices, total_size) -> List[float]:
        final_scores: list[float] = [self.default_reward] * total_size
        for j, output in enumerate(outputs):
            text = output.outputs[0].text
            score = single_extract_score(text)
            if score is None:
                score = self.default_reward
            else:
                score = score * self.score_scale_factor
            if j < len(kept_indices):
                idx = kept_indices[j]
                response_ids = data.batch["responses"][idx]
                resp_len = response_ids.shape[-1]
                valid_len = data.batch["attention_mask"][idx][-resp_len:].sum()
                try:
                    valid_len_int = int(valid_len)
                except Exception:
                    valid_len_int = resp_len
                penalty = _compute_overlong_penalty(valid_len_int, self.overlong_buffer_cfg)
                score = score - penalty
                final_scores[kept_indices[j]] = score

        # filtered indices remain default_reward
        return final_scores

    def compute_scores(self, data, generate_fn):
        prompt_list, kept_indices, total_size = self.process_input(data)
        outputs = generate_fn(prompt_list)
        return self.process_output(outputs, data, kept_indices, total_size)


class GroupRewardModelProcessor:
    """Group generative RM: ranks/scores multiple translations of the same source via a single LLM call."""
    def __init__(self, *args, **kwargs):
        self.config = kwargs.get("config")
        self.tokenizer = kwargs.get("tokenizer", None)
        self.input_tokenizer = kwargs.get("input_tokenizer", self.tokenizer)
        self.max_prompt_length = self.config.prompt_length
        self.extractor_type = self.config.custom_processor.get("extractor_type", "line")
        print(f"Using extractor_type: {self.extractor_type}")
        self.prompt_type = getattr(self.config, "group_prompt_type", "ranking_score")
        self.add_example = getattr(self.config, "group_add_example", False)
        self.score_scale_factor = getattr(self.config, "score_scale_factor", 0.1)
        self.default_reward = getattr(self.config, "default_reward", 0.0)
        self.overlong_buffer_cfg = self.config.custom_processor.get("overlong_buffer", None)
        self.enable_language_detection = self.config.custom_processor.get("enable_language_detection", False)
        self.return_reward_model_metadata = self.config.custom_processor.get(
            "return_reward_model_metadata",
            self.config.custom_processor.get("return_gqm_outputs", False),
        )
        if self.enable_language_detection:
            print(f"Language detection enabled")
        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided")
        if self.input_tokenizer is None:
            raise ValueError("input_tokenizer must be provided")

    def compute_scores(self, data, generate_fn):
        result = compute_group_translation_scores(
            data, generate_fn, self.tokenizer, self.input_tokenizer,
            extractor_type=self.extractor_type,
            max_prompt_length=self.max_prompt_length,
            prompt_type=self.prompt_type,
            add_example=self.add_example,
            score_scale_factor=self.score_scale_factor,
            default_reward=self.default_reward,
            overlong_buffer_cfg=self.overlong_buffer_cfg,
            enable_language_detection=self.enable_language_detection,
            return_reward_model_metadata=self.return_reward_model_metadata,
        )
        if self.return_reward_model_metadata:
            scores_dict, gqm_prompts_dict, gqm_outputs_dict = result
        else:
            scores_dict = result
            gqm_prompts_dict = {}
            gqm_outputs_dict = {}
        total_size = data.batch.batch_size[0]
        scores = [scores_dict.get(i, self.default_reward) for i in range(total_size)]
        if not self.return_reward_model_metadata:
            return scores
        gqm_prompts = [gqm_prompts_dict.get(i, None) for i in range(total_size)]
        gqm_outputs = [gqm_outputs_dict.get(i, "") for i in range(total_size)]
        return RewardProcessorOutput(
            scores=scores,
            non_tensor_batch={
                REWARD_MODEL_PROMPTS_KEY: gqm_prompts,
                REWARD_MODEL_RESPONSES_KEY: gqm_outputs,
            },
        )


class SeedXRewardModelProcessor:
    """SeedX-style RM: scores translations using a SeedX-format prompt and a separate score generation step."""
    def __init__(self, *args, **kwargs):
        self.config = kwargs.get("config")
        self.tokenizer = kwargs.get("tokenizer", None)
        self.input_tokenizer = kwargs.get("input_tokenizer", self.tokenizer)
        self.max_prompt_length = getattr(self.config, "prompt_length", 1 << 20)
        self.extractor_type = self.config.custom_processor.get("extractor_type", "line")
        self.batch_size = getattr(self.config, "seedx_rm_batch_size", 32)
        self.score_scale_factor = getattr(self.config, "score_scale_factor", 1.0)
        self.score_lower_bound = getattr(self.config, "score_lower_bound", -10000.0)
        self.score_upper_bound = getattr(self.config, "score_upper_bound", 10000.0)
        self.overlong_buffer_cfg = self.config.custom_processor.get("overlong_buffer", None)
        self.enable_language_detection = self.config.custom_processor.get("enable_language_detection", False)
        if self.enable_language_detection:
            print(f"Language detection enabled")
        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided")
        if self.input_tokenizer is None:
            raise ValueError("input_tokenizer must be provided")

    def process_input(self, data):
        response_list = _decode_response(data, self.input_tokenizer, self.extractor_type)
        src_text_list = [item["src_text"] for item in data.non_tensor_batch["extra_info"]]
        lang_pair_list = [_get_lang_pair(item) for item in data.non_tensor_batch["extra_info"]]
        src_langs, tgt_langs = zip(*lang_pair_list)
        assert len(src_text_list) == len(response_list) == len(src_langs) == len(tgt_langs)
        prompts: List[str] = []
        chosens: List[str] = []
        kept_indices: List[int] = []
        filtered_indices: List[int] = []
        for idx, (src_text, mt_text, src_lang, tgt_lang) in enumerate(
            zip(src_text_list, response_list, src_langs, tgt_langs)
        ):
            if mt_text is None:
                filtered_indices.append(idx)
                continue
            if self.enable_language_detection:
                if not is_language_match(mt_text, tgt_lang):
                    filtered_indices.append(idx)
                    continue
            prompt = _seedx_build_prompt(src_text, mt_text, src_lang, tgt_lang)
            ids_len = (
                len(self.tokenizer.encode(prompt))
                + len(self.tokenizer.encode(mt_text))
                + 1
            )
            if ids_len > self.max_prompt_length:
                filtered_indices.append(idx)
                continue
            kept_indices.append(idx)
            prompts.append(prompt)
            chosens.append(mt_text)
        total_size = len(src_text_list)
        return prompts, chosens, kept_indices, total_size

    def score_postprocess(self, scores: List[float]) -> List[float]:
        return [
            max(min(score * self.score_scale_factor, self.score_upper_bound), self.score_lower_bound)
            for score in scores
        ]

    def compute_scores(self, data, generate_fn):
        prompts, chosens, kept_indices, total_size = self.process_input(data)
        if len(kept_indices) > 0:
            kept_scores = generate_fn(prompts, chosens)
            default_score = min(kept_scores)
            scores: List[float] = [default_score] * total_size
            for j, idx in enumerate(kept_indices):
                response_ids = data.batch["responses"][idx]
                resp_len = response_ids.shape[-1]
                valid_len = data.batch["attention_mask"][idx][-resp_len:].sum()
                try:
                    valid_len_int = int(valid_len)
                except Exception:
                    valid_len_int = resp_len
                penalty = _compute_overlong_penalty(valid_len_int, self.overlong_buffer_cfg)
                scores[idx] = kept_scores[j] - penalty
        else:
            scores: List[float] = [self.score_lower_bound] * total_size
        return self.score_postprocess(scores)


class VHeadRewardModelProcessor:
    """Value-head RM: scores translations by feeding prompt+response through the model's value head."""
    def __init__(self, *args, **kwargs):
        self.config = kwargs.get("config")
        self.tokenizer = kwargs.get("tokenizer", None)
        self.input_tokenizer = kwargs.get("input_tokenizer", self.tokenizer)
        self.max_prompt_length = getattr(self.config, "prompt_length", 1 << 20)
        self.extractor_type = self.config.custom_processor.get("extractor_type", "line")
        self.chat_template = getattr(self.config, "chat_template", True)
        self.batch_size = getattr(self.config, "seedx_rm_batch_size", 32)
        self.score_scale_factor = getattr(self.config, "score_scale_factor", 1.0)
        self.score_lower_bound = getattr(self.config, "score_lower_bound", -10000.0)
        self.score_upper_bound = getattr(self.config, "score_upper_bound", 10000.0)
        self.overlong_buffer_cfg = self.config.custom_processor.get("overlong_buffer", None)
        self.enable_language_detection = self.config.custom_processor.get("enable_language_detection", False)
        if self.enable_language_detection:
            print(f"Language detection enabled")
        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided")
        if self.input_tokenizer is None:
            raise ValueError("input_tokenizer must be provided")

    def process_input(self, data):
        response_list = _decode_response(data, self.input_tokenizer, self.extractor_type)
        src_text_list = [item["src_text"] for item in data.non_tensor_batch["extra_info"]]
        lang_pair_list = [_get_lang_pair(item) for item in data.non_tensor_batch["extra_info"]]
        src_langs, tgt_langs = zip(*lang_pair_list)
        assert len(src_text_list) == len(response_list) == len(src_langs) == len(tgt_langs)
        input_texts = []
        kept_indices = []
        filtered_indices = []
        for idx, (src_text, mt_text, src_lang, tgt_lang) in enumerate(zip(src_text_list, response_list, src_langs, tgt_langs)):
            if mt_text is None:
                filtered_indices.append(idx)
                continue
            if self.enable_language_detection:
                if not is_language_match(mt_text, tgt_lang):
                    filtered_indices.append(idx)
                    continue
            full_text = _vanilla_rm_build_prompt(self.tokenizer, src_lang, tgt_lang, src_text, mt_text, chat_template=self.chat_template)
            ids_len = len(self.tokenizer.encode(full_text))
            if ids_len > self.max_prompt_length:
                filtered_indices.append(idx)
                continue
            kept_indices.append(idx)
            input_texts.append(full_text)
        total_size = len(src_text_list)
        return input_texts, kept_indices, total_size

    def score_postprocess(self, scores: List[float]) -> List[float]:
        return [
            max(min(score * self.score_scale_factor, self.score_upper_bound), self.score_lower_bound)
            for score in scores
        ]

    def compute_scores(self, data, generate_fn):
        input_texts, kept_indices, total_size = self.process_input(data)
        if len(kept_indices) > 0:
            kept_scores = generate_fn(input_texts)
            default_score = min(kept_scores)
            scores: List[float] = [default_score] * total_size
            for j, idx in enumerate(kept_indices):
                response_ids = data.batch["responses"][idx]
                resp_len = response_ids.shape[-1]
                valid_len = data.batch["attention_mask"][idx][-resp_len:].sum()
                try:
                    valid_len_int = int(valid_len)
                except Exception:
                    valid_len_int = resp_len
                penalty = _compute_overlong_penalty(valid_len_int, self.overlong_buffer_cfg)
                scores[idx] = kept_scores[j] - penalty
        else:
            scores: List[float] = [self.score_lower_bound] * total_size
        return self.score_postprocess(scores)


class GroupPostEditRewardProcessor:
    """Post-edit RM: scores a candidate MT against baseline MTs using GQM prompts, returns relative reward."""
    def __init__(self, *args, **kwargs):
        self.config = kwargs.get("config")
        self.tokenizer = kwargs.get("tokenizer", None)
        self.input_tokenizer = kwargs.get("input_tokenizer", self.tokenizer)
        self.max_prompt_length = self.config.prompt_length
        self.extractor_type = self.config.custom_processor.get("extractor_type", "codeblock")
        print(f"Using extractor_type: {self.extractor_type}")
        self.prompt_format = getattr(self.config, "group_prompt_type", "ranking_score")
        self.add_example = getattr(self.config, "group_add_example", False)
        self.default_reward = getattr(self.config, "default_reward", -1.0)
        self.rm_max_candidates = getattr(self.config, "rm_max_candidates", 4)
        self.score_scale_factor = getattr(self.config, "score_scale_factor", 0.1)
        self.group_post_edit_score_mode = getattr(self.config, "group_post_edit_score_mode", "mt_group_advantage")
        self.overlong_buffer_cfg = self.config.custom_processor.get("overlong_buffer", None)
        self.enable_language_detection = self.config.custom_processor.get("enable_language_detection", False)
        print(f"Using group_post_edit_score_mode: {self.group_post_edit_score_mode}")
        if self.enable_language_detection:
            print(f"Language detection enabled")
        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided")
        if self.input_tokenizer is None:
            raise ValueError("input_tokenizer must be provided")

    def process_input(self, data):
        return prepare_group_post_edit_inputs(
            data,
            self.tokenizer,
            self.input_tokenizer,
            extractor_type=self.extractor_type,
            max_prompt_length=self.max_prompt_length,
            prompt_format=self.prompt_format,
            add_example=self.add_example,
            rm_max_candidates=self.rm_max_candidates,
            enable_language_detection=self.enable_language_detection,
        )

    def process_output(self, outputs, kept_info, total_size) -> List[float]:
        final_scores: List[float] = [self.default_reward] * total_size
        scores_dict = process_group_post_edit_outputs(
            outputs,
            kept_info,
            prompt_format=self.prompt_format,
            score_scale_factor=self.score_scale_factor,
            default_reward=self.default_reward,
            overlong_buffer_cfg=self.overlong_buffer_cfg,
        )
        for idx, score in scores_dict.items():
            final_scores[idx] = score
        return final_scores

    def compute_scores(self, data, generate_fn):
        scores_dict = compute_group_post_edit_scores(
            data,
            generate_fn,
            self.tokenizer,
            self.input_tokenizer,
            extractor_type=self.extractor_type,
            max_prompt_length=self.max_prompt_length,
            prompt_format=self.prompt_format,
            add_example=self.add_example,
            score_scale_factor=self.score_scale_factor,
            default_reward=self.default_reward,
            rm_max_candidates=self.rm_max_candidates,
            overlong_buffer_cfg=self.overlong_buffer_cfg,
            enable_language_detection=self.enable_language_detection,
            score_mode=self.group_post_edit_score_mode,
        )
        total_size = data.batch.batch_size[0]
        return [scores_dict.get(i, self.default_reward) for i in range(total_size)]


def score_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    return 0
    # print(f"[debug] extra_info: {extra_info}")
    # raise ValueError("extra_info must be provided")



def batch_bleurt_reward_fn(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos=None,
    extractor_type: str = "line",
    score_scale_factor: float = 1.0,
):
    """Score translations against references using BLEURT-20."""
    assert extra_infos is not None
    assert (
        isinstance(data_sources, Iterable)
        and isinstance(solution_strs, Iterable)
        and isinstance(ground_truths, Iterable)
        and isinstance(extra_infos, Iterable)
    )
    assert (
        len(data_sources)
        == len(solution_strs)
        == len(ground_truths)
        == len(extra_infos)
    )
    n = len(solution_strs)
    kept_mt: List[str] = []
    kept_ref: List[str] = []
    kept_idx: List[int] = []
    default_score = 0.0
    for i in range(n):
        info = extra_infos[i] if extra_infos is not None else {}
        ref = None
        if isinstance(info, dict):
            ref = info.get("tgt_text") or info.get("trg_text")
        if ref is None:
            ref = ground_truths[i]
        mt_raw = solution_strs[i]
        if isinstance(mt_raw, str):
            if extractor_type == "line":
                mt = _line_extractor(mt_raw)
            elif extractor_type == "codeblock":
                mt = _block_extractor(mt_raw)
            elif extractor_type == "oneline":
                mt = _one_line_extractor(mt_raw)
            else:
                mt = mt_raw
        else:
            mt = None
        if isinstance(mt, str) and isinstance(ref, str) and mt.strip() and ref.strip():
            kept_mt.append(mt)
            kept_ref.append(ref)
            kept_idx.append(i)
    if len(kept_idx) > 0:
        try:
            try:
                from .bleurt_service import func_call as bleurt_func_call
            except Exception:
                from reward_utils.bleurt_service import func_call as bleurt_func_call
            result = bleurt_func_call("BLEURT-20", kept_mt, kept_ref)
            scores_list = result.get("scores", [])
        except Exception:
            scores_list = [default_score] * len(kept_idx)
    else:
        scores_list = []
    final_scores: List[float] = [default_score] * n
    for j, idx in enumerate(kept_idx):
        final_scores[idx] = float(scores_list[j]) * float(score_scale_factor)
    return final_scores


class MultiTaskSelfRewardProcessor:
    """Routes samples to translation, ranking, or group post-edit scoring based on the 'ability' field."""

    def __init__(self, *args, **kwargs):
        self.config = kwargs.get("config")
        self.tokenizer = kwargs.get("tokenizer", None)
        self.input_tokenizer = kwargs.get("input_tokenizer", self.tokenizer)

        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided")
        if self.input_tokenizer is None:
            raise ValueError("input_tokenizer must be provided")

        self.max_prompt_length = self.config.prompt_length
        self.extractor_type = self.config.custom_processor.get("extractor_type", "line")
        print(f"Using extractor_type: {self.extractor_type}")

        self.prompt_type = getattr(self.config, "group_prompt_type", "ranking_score")
        self.add_example = getattr(self.config, "group_add_example", False)
        score_scale_factor  = getattr(self.config, "score_scale_factor", 1.0)
        self.mt_score_scale_factor = getattr(self.config, "mt_score_scale_factor", score_scale_factor)
        self.gpe_score_scale_factor = getattr(self.config, "gpe_score_scale_factor", score_scale_factor)
        self.default_reward = getattr(self.config, "default_reward", 0.0)
        self.rm_max_candidates = getattr(self.config, "rm_max_candidates", 4)
        self.group_post_edit_score_mode = getattr(self.config, "group_post_edit_score_mode", "mt_group_advantage")
        self.mt_overlong_buffer_cfg = self.config.custom_processor.get("overlong_buffer", None)
        self.gpe_overlong_buffer_cfg = self.config.custom_processor.get(
            "gpe_overlong_buffer", self.mt_overlong_buffer_cfg
        )
        self.enable_language_detection = self.config.custom_processor.get("enable_language_detection", False)
        self.reuse_gqm_post_edit_first_turn_scores = self.config.custom_processor.get(
            "reuse_gqm_post_edit_first_turn_scores", False
        )
        self.enable_gqm_post_edit_fallback_bonus = self.config.custom_processor.get(
            "enable_gqm_post_edit_fallback_bonus", False
        )
        self.gqm_post_edit_fallback_bonus_reward = self.config.custom_processor.get(
            "gqm_post_edit_fallback_bonus_reward", 0.0
        )
        if self.enable_gqm_post_edit_fallback_bonus:
            if not self.reuse_gqm_post_edit_first_turn_scores:
                raise ValueError(
                    "enable_gqm_post_edit_fallback_bonus requires "
                    "reuse_gqm_post_edit_first_turn_scores=True"
                )
            if self.group_post_edit_score_mode != "mt_group_advantage":
                raise ValueError(
                    "enable_gqm_post_edit_fallback_bonus requires "
                    "group_post_edit_score_mode='mt_group_advantage'"
                )
        if self.enable_language_detection:
            print(f"Language detection enabled")

        self.ranking_score_scale_factor = getattr(self.config, "ranking_score_scale_factor", score_scale_factor)

        print(f"MultiTaskSelfRewardProcessor initialized with prompt_type={self.prompt_type}, "
              f"mt_score_scale_factor={self.mt_score_scale_factor}, "
              f"ranking_score_scale_factor={self.ranking_score_scale_factor}, "
              f"gpe_score_scale_factor={self.gpe_score_scale_factor}, "
              f"group_post_edit_score_mode={self.group_post_edit_score_mode}, "
              f"rm_max_candidates={self.rm_max_candidates}, "
              f"reuse_gqm_post_edit_first_turn_scores={self.reuse_gqm_post_edit_first_turn_scores}, "
              f"enable_gqm_post_edit_fallback_bonus={self.enable_gqm_post_edit_fallback_bonus}, "
              f"gqm_post_edit_fallback_bonus_reward={self.gqm_post_edit_fallback_bonus_reward}")

    def _split_by_ability(self, data) -> Tuple[List[int], List[int], List[int], List[int]]:
        abilities = data.non_tensor_batch.get("ability", None)
        if abilities is None:
            raise ValueError("ability not found in data.non_tensor_batch")

        translation_indices = []
        ranking_indices = []
        group_post_edit_indices = []
        gqm_post_edit_indices = []

        for idx, ability in enumerate(abilities):
            ability_str = str(ability).strip().lower()
            if ability_str == "translation":
                translation_indices.append(idx)
            elif ability_str == "ranking":
                ranking_indices.append(idx)
            elif ability_str == "group_post_edit":
                group_post_edit_indices.append(idx)
            elif ability_str == "gqm_post_edit":
                gqm_post_edit_indices.append(idx)
            else:
                print(f"Warning: Unknown ability type '{ability}' at index {idx}, treating as translation")
                translation_indices.append(idx)

        return translation_indices, ranking_indices, group_post_edit_indices, gqm_post_edit_indices

    def _process_translation_task(self, data, translation_indices: List[int], generate_fn) -> Dict[int, float]:
        return compute_group_translation_scores(
            data, generate_fn, self.tokenizer, self.input_tokenizer,
            extractor_type=self.extractor_type,
            max_prompt_length=self.max_prompt_length,
            prompt_type=self.prompt_type,
            add_example=self.add_example,
            score_scale_factor=self.mt_score_scale_factor,
            default_reward=self.default_reward,
            overlong_buffer_cfg=self.mt_overlong_buffer_cfg,
            enable_language_detection=self.enable_language_detection,
            indices=translation_indices,
        )

    def _process_group_post_edit_task(self, data, group_post_edit_indices: List[int], generate_fn) -> Dict[int, float]:
        return compute_group_post_edit_scores(
            data, generate_fn, self.tokenizer, self.input_tokenizer,
            extractor_type=self.extractor_type,
            max_prompt_length=self.max_prompt_length,
            prompt_format=self.prompt_type,
            add_example=self.add_example,
            score_scale_factor=self.gpe_score_scale_factor,
            default_reward=self.default_reward,
            rm_max_candidates=self.rm_max_candidates,
            overlong_buffer_cfg=self.gpe_overlong_buffer_cfg,
            enable_language_detection=self.enable_language_detection,
            indices=group_post_edit_indices,
            score_mode=self.group_post_edit_score_mode,
        )

    def _process_gqm_post_edit_task(self, data, gqm_post_edit_indices: List[int], generate_fn) -> Dict[int, float]:
        post_edit_responses = _decode_last_assistant_response(data, self.input_tokenizer, self.extractor_type)
        scores_dict: Dict[int, float] = {}
        remaining_indices = gqm_post_edit_indices
        if self.reuse_gqm_post_edit_first_turn_scores and self.group_post_edit_score_mode == "mt_group_advantage":
            scores_dict, remaining_indices = _try_score_gqm_post_edit_from_first_turn(
                data,
                indices=gqm_post_edit_indices,
                post_edit_responses=post_edit_responses,
                prompt_format=self.prompt_type,
                score_scale_factor=self.gpe_score_scale_factor,
                rm_max_candidates=self.rm_max_candidates,
                overlong_buffer_cfg=self.gpe_overlong_buffer_cfg,
                enable_language_detection=self.enable_language_detection,
            )
            skipped_count = len(scores_dict)
            total_count = len(gqm_post_edit_indices)
            skipped_ratio = skipped_count / total_count if total_count else 0.0
            print(
                "[GQM_GPE_FAST_PATH] "
                f"skipped={skipped_count}/{total_count} "
                f"ratio={skipped_ratio:.2%}"
            )

        fallback_scores = compute_group_post_edit_scores(
            data, generate_fn, self.tokenizer, self.input_tokenizer,
            extractor_type=self.extractor_type,
            max_prompt_length=self.max_prompt_length,
            prompt_format=self.prompt_type,
            add_example=self.add_example,
            score_scale_factor=self.gpe_score_scale_factor,
            default_reward=self.default_reward,
            rm_max_candidates=self.rm_max_candidates,
            overlong_buffer_cfg=self.gpe_overlong_buffer_cfg,
            enable_language_detection=self.enable_language_detection,
            indices=remaining_indices,
            score_mode=self.group_post_edit_score_mode,
            response_texts=post_edit_responses,
        )
        if self.enable_gqm_post_edit_fallback_bonus:
            fallback_scores = {
                idx: score + self.gqm_post_edit_fallback_bonus_reward if score > 0 else score
                for idx, score in fallback_scores.items()
            }
        scores_dict.update(fallback_scores)
        return scores_dict

    def _process_ranking_task(self, data, ranking_indices: List[int]) -> Dict[int, float]:
        if not ranking_indices:
            return {}

        try:
            from reward_utils.ranking_score_reward import ranking_score_reward_fn
        except ImportError:
            from .ranking_score_reward import ranking_score_reward_fn

        scores_dict: Dict[int, float] = {}

        for idx in ranking_indices:
            response_ids = data.batch["responses"][idx]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][idx][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            solution_str = self.input_tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            solution_str = solution_str.replace(self.input_tokenizer.eos_token, "")

            ground_truth = None
            if "reward_model" in data.non_tensor_batch:
                reward_model_data = data.non_tensor_batch["reward_model"]
                if isinstance(reward_model_data, dict) and "ground_truth" in reward_model_data:
                    ground_truth = reward_model_data["ground_truth"]
                elif hasattr(reward_model_data, "__getitem__"):
                    try:
                        ground_truth = reward_model_data[idx].get("ground_truth") if isinstance(reward_model_data[idx], dict) else reward_model_data[idx]
                    except (IndexError, KeyError, TypeError):
                        pass

            if ground_truth is None:
                print("[Warning] empty ground truth!")
                scores_dict[idx] = self.default_reward
                continue

            data_source = data.non_tensor_batch.get("data_source", [""] * len(ranking_indices))
            if isinstance(data_source, (list, tuple)):
                data_source = data_source[idx] if idx < len(data_source) else ""

            extra_info = data.non_tensor_batch.get("extra_info", None)
            if extra_info is not None and hasattr(extra_info, "__getitem__"):
                try:
                    extra_info = extra_info[idx]
                except (IndexError, KeyError, TypeError):
                    extra_info = None

            reward_result = ranking_score_reward_fn(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                score_scale_factor=self.ranking_score_scale_factor,
            )

            scores_dict[idx] = reward_result.get("score", self.default_reward)

        return scores_dict

    def compute_scores(self, data, generate_fn):
        total_size = data.batch.batch_size[0]
        translation_indices, ranking_indices, group_post_edit_indices, gqm_post_edit_indices = self._split_by_ability(data)

        final_scores: List[float] = [self.default_reward] * total_size

        translation_scores = self._process_translation_task(data, translation_indices, generate_fn)
        for idx, score in translation_scores.items():
            final_scores[idx] = score

        group_post_edit_scores = self._process_group_post_edit_task(data, group_post_edit_indices, generate_fn)
        for idx, score in group_post_edit_scores.items():
            final_scores[idx] = score

        gqm_post_edit_scores = self._process_gqm_post_edit_task(data, gqm_post_edit_indices, generate_fn)
        for idx, score in gqm_post_edit_scores.items():
            final_scores[idx] = score

        ranking_scores = self._process_ranking_task(data, ranking_indices)
        for idx, score in ranking_scores.items():
            final_scores[idx] = score

        return final_scores
